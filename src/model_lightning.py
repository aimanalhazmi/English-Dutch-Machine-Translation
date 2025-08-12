from typing import List
import torch 
from torch import nn 
import random 
import lightning

class Encoder(nn.Module):

    def __init__(self, src_vocab_size, embedding_dim, hidden_dim, n_layers, pretrained_embeddings=None, freeze_embeddings=False, pad_idx=0, p_dropout=0.1):

        super().__init__() 

        self.hidden_dim = hidden_dim 
        self.n_layers = n_layers 

        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=p_dropout)
        self.dropout = nn.Dropout(p_dropout)

        # projection layers to project from bidirectional encoder to unidirectional decoder
        self.hidden_proj = nn.Linear(2*hidden_dim, hidden_dim)
        self.cell_proj = nn.Linear(2*hidden_dim, hidden_dim)

    def forward(self, x):

        # shape x: (batch_size, seq_length)
        batch_size = x.shape[0]

        embedded = self.dropout(self.embedding(x))  # shape: (batch_size, seq_length, embedding_dim)

        _ , (hidden, cell) = self.rnn(embedded) # shape hidden/cell: (2*n_layers, batch_size, hidden_dim)

        ## project from bidirectional to unidirectional
        # separate forward and backward states
        hidden = hidden.reshape(self.n_layers, 2, batch_size, self.hidden_dim)
        cell = cell.reshape(self.n_layers, 2, batch_size, self.hidden_dim)

        # concatenate (in hidden_dim axis)
        hidden = torch.cat([hidden[:, 0], hidden[:, 1]], axis=2)
        cell = torch.cat([cell[:, 0], cell[:, 1]], axis=2)

        # project down to single hidden_dim (linear layer + tanh)
        hidden = torch.tanh(self.hidden_proj(hidden))
        cell = torch.tanh(self.cell_proj(cell))

        return hidden, cell
    

class Decoder(nn.Module):

    def __init__(self, tgt_vocab_size, embedding_dim, hidden_dim, n_layers, pretrained_embeddings=None, freeze_embeddings=False, pad_idx=0, p_dropout=0.1):

        super().__init__() 

        self.hidden_dim = hidden_dim 
        self.n_layers = n_layers 

        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=False, dropout=p_dropout)
        self.dropout = nn.Dropout(p_dropout)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, x, hidden, cell):

        # x: (batch_size)
        x = x[:, None]  # x: (batch_size, 1)

        embedded = self.dropout(self.embedding(x))  # embedded: (batch_size, 1, embedding_dim)

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: (batch_size, 1, hidden_dim)
        # hidden/cell: (n_layers, batch_size, hidden_dim)

        pred = self.fc(output.squeeze(1))   # pred: (batch_size, tgt_vocab_size)

        return pred, hidden, cell
    

class Seq2SeqModel(lightning.LightningModule):

    def __init__(self,
                 src_vocab,
                 tgt_vocab,
                 embedding_dim: int,
                 hidden_dim: int = 512,
                 n_layers: int = 2,
                 p_dropout: float = 0.1,
                 src_embeddings = None,
                 tgt_embeddings = None,
                 trainable_embeddings: List = [1,2,3],
                 teacher_forcing_ratio: float = 0.5,
                 teacher_forcing_decay: float = 0.95,
                 learning_rate: float = 1e-3):

        super().__init__()

        self.save_hyperparameters(ignore=["src_vocab", "tgt_vocab", "src_embeddings", "tgt_embeddings"])
        
        # init encoder and decoder 
        pad_idx = src_vocab.stoi["<pad>"]
        self.encoder = Encoder(len(src_vocab), embedding_dim, hidden_dim, n_layers, src_embeddings, False, pad_idx, p_dropout)
        self.decoder = Decoder(len(tgt_vocab), embedding_dim, hidden_dim, n_layers, tgt_embeddings, False, pad_idx, p_dropout)

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.src_vocab = src_vocab 
        self.tgt_vocab = tgt_vocab

        # define loss function 
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi["<pad>"])

        # gradient masking for pre-trained embeddings
        self.frozen_src_embeddings = torch.tensor([i for i in range(len(src_vocab)) if i not in trainable_embeddings], dtype=torch.long)
        self.frozen_tgt_embeddings = torch.tensor([i for i in range(len(tgt_vocab)) if i not in trainable_embeddings], dtype=torch.long)

        @self.encoder.embedding.weight.register_hook
        def mask_src_grad(grad):
            grad[self.frozen_src_embeddings] = 0
            return grad
   
        @self.decoder.embedding.weight.register_hook
        def mask_tgt_grad(grad):
            grad[self.frozen_tgt_embeddings] = 0
            return grad
   

    def forward(self, src_sent, tgt_sent, teacher_forcing_ratio=None):

        if teacher_forcing_ratio is None:
            teacher_forcing_ratio = self.teacher_forcing_ratio

        # src_sent: (batch_size, src_length)
        # tgt_sent: (batch_size, tgt_length)

        batch_size, tgt_length = tgt_sent.shape
        tgt_vocab_size = self.decoder.fc.out_features

        # decoder outputs
        outputs = torch.zeros(batch_size, tgt_length, tgt_vocab_size, device=self.device)

        # calculate context vector from encoder 
        hidden, cell = self.encoder(src_sent)   # hidden/cell: (n_layers, batch_size, hidden_dim)

        # get <sos> token as starting input
        input_tok = tgt_sent[:, 0]  # input_tok: (batch_size)

        for t in range(1, tgt_length):

            output, hidden, cell = self.decoder(input_tok, hidden, cell)
            # output: (batch_size, tgt_vocab_size)
            # hidden/cell: (n_layers, batch_size, hidden_dim) 

            # store output prediction
            outputs[:, t, :] = output 

            # use either ground truth or predicted token as next input
            # depending on teacher forcing
            use_teacher_force = random.random() < teacher_forcing_ratio 
            if use_teacher_force:
                input_tok = tgt_sent[:, t]
            else:
                input_tok = output.argmax(axis=1)

        return outputs
    
    def training_step(self, batch, batch_idx):

        src_sent, tgt_sent = batch 

        # forward pass 
        y_logit = self(src_sent, tgt_sent)  # y_logit: (batch_size, tgt_length, tgt_vocab_size)

        ## calculate loss
        # reshape and ignore first <sos> token
        y_logit = y_logit[:,1:].reshape(-1, y_logit.shape[-1])
        tgt_sent = tgt_sent[:, 1:].reshape(-1)

        loss = self.loss_fn(y_logit, tgt_sent)

        # logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("teacher_forcing_ratio", self.teacher_forcing_ratio, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):

        src_sent, tgt_sent = batch 

        # forward pass 
        y_logit = self(src_sent, tgt_sent, teacher_forcing_ratio=0.0)  # y_logit: (batch_size, tgt_length, tgt_vocab_size)

        ## calculate loss
        # reshape and ignore first <sos> token
        y_logit = y_logit[:,1:].reshape(-1, y_logit.shape[-1])
        tgt_sent = tgt_sent[:, 1:].reshape(-1)

        loss = self.loss_fn(y_logit, tgt_sent)

        # logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):

        src_sent, tgt_sent = batch 

        # forward pass 
        y_logit = self(src_sent, tgt_sent, teacher_forcing_ratio=0.0)  # y_logit: (batch_size, tgt_length, tgt_vocab_size)

        ## calculate loss
        # reshape and ignore first <sos> token
        y_logit = y_logit[:,1:].reshape(-1, y_logit.shape[-1])
        tgt_sent = tgt_sent[:, 1:].reshape(-1)

        loss = self.loss_fn(y_logit, tgt_sent)

        # logging
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

        return loss
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer
    
    def on_train_epoch_end(self):

        # teacher forcing ratio decay
        self.teacher_forcing_ratio = max(0.1, self.teacher_forcing_ratio * self.hparams.teacher_forcing_decay)


