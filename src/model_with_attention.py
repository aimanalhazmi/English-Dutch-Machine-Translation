from typing import List
import torch 
from torch import nn 
import random 
import lightning
from torchmetrics.text import BLEUScore
import wandb


class Encoder(nn.Module):

    def __init__(self, src_vocab_size, embedding_dim, hidden_dim, n_layers, pretrained_embeddings=None, freeze_embeddings=False, pad_idx=0, p_dropout=0.1):

        super().__init__() 

        self.hidden_dim = hidden_dim 
        self.n_layers = 1

        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(src_vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=True, dropout=p_dropout)
        self.dropout = nn.Dropout(p_dropout)

        self.fc_hidden = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_cell = nn.Linear(2*hidden_dim, hidden_dim)

    def forward(self, x):

        # shape x: (batch_size, seq_length)
        batch_size = x.shape[0]

        embedded = self.dropout(self.embedding(x))  # shape: (batch_size, seq_length, embedding_dim)

        outputs , (hidden, cell) = self.rnn(embedded) # shape hidden/cell: (2*n_layers, batch_size, hidden_dim) / outputs: (batch_size, seq_length, 2*hidden_dim)

        # map from bidirectional to unididrectional
        hidden = self.fc_hidden(torch.cat([hidden[0:self.n_layers], hidden[self.n_layers:2*self.n_layers]], axis=2))
        cell = self.fc_cell(torch.cat([cell[0:self.n_layers], cell[self.n_layers:2*self.n_layers]], axis=2))

        return outputs, hidden, cell
    
class Attention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()
        self.attn_fc = nn.Linear(3*hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, hidden, encoder_out):

        batch_size, seq_length, _ = encoder_out.shape

        hidden = hidden.repeat(1, seq_length, 1)

        energy = self.relu(self.attn_fc(torch.cat([hidden, encoder_out], axis=2)))

        attn = torch.softmax(energy, axis=1)

        return attn
    

class Decoder(nn.Module):

    def __init__(self, tgt_vocab_size, embedding_dim, hidden_dim, n_layers, attention, pretrained_embeddings=None, freeze_embeddings=False, pad_idx=0, p_dropout=0.1):

        super().__init__() 

        self.hidden_dim = hidden_dim 
        self.n_layers = 1

        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(tgt_vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=freeze_embeddings, padding_idx=pad_idx)

        self.rnn = nn.LSTM(2*hidden_dim + embedding_dim, hidden_dim, n_layers, batch_first=True, bidirectional=False, dropout=p_dropout)
        self.dropout = nn.Dropout(p_dropout)
        self.fc = nn.Linear(3*hidden_dim + embedding_dim, tgt_vocab_size)

        self.attention = attention

    def forward(self, x, hidden, cell, encoder_out):

        # x: (batch_size)
        x = x[:, None]  # x: (batch_size, 1)

        embedded = self.dropout(self.embedding(x))  # embedded: (batch_size, 1, embedding_dim)

        attn = self.attention(hidden, encoder_out)[:,None]  # attn: (batch_size, 1, seq_len)

        # multiply attn with encoder outputs
        weighted = torch.einsum("bol,blh->boh", attn, encoder_out)
        weighted = weighted.permute(1,0,2)  # weighted: (1, batch_size, 2* hidden_dim)
        rnn_in = torch.cat([embedded, weighted], axis=2)

        output, (hidden, cell) = self.rnn(rnn_in, (hidden, cell))
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
                 learning_rate: float = 1e-3,
                 class_weights = None,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):

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
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, weight=class_weights)
        # self.loss_fn = WeightedFocalLoss(class_weights=class_weights, alpha=focal_alpha, gamma=focal_gamma, ignore_idx=pad_idx)

        # BLEU metric
        self.bleu = BLEUScore()

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
        encoder_out, hidden, cell = self.encoder(src_sent)   # hidden/cell: (n_layers, batch_size, hidden_dim)

        # get <sos> token as starting input
        input_tok = tgt_sent[:, 0]  # input_tok: (batch_size)

        for t in range(1, tgt_length):

            output, hidden, cell = self.decoder(input_tok, hidden, cell, encoder_out)
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
        y_logit = y_logit[:,1:]
        tgt_sent = tgt_sent[:, 1:]

        y_logit_flat = y_logit.reshape(-1, y_logit.shape[-1])
        tgt_sent_flat = tgt_sent.reshape(-1)

        loss = self.loss_fn(y_logit_flat, tgt_sent_flat)

        # calculate BLEU score
        bleu_score = self._calculate_bleu_score(y_logit, tgt_sent)

        # logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_bleu", bleu_score, on_epoch=True, prog_bar=True)
        self.log("teacher_forcing_ratio", self.teacher_forcing_ratio, on_epoch=True, on_step=False)

        if batch_idx == 0:
            self._log_translations(src_sent[:,1:], y_logit, tgt_sent, mode="train", n_examples=5)

        return loss
    
    def validation_step(self, batch, batch_idx):

        src_sent, tgt_sent = batch 

        # forward pass 
        y_logit = self(src_sent, tgt_sent, teacher_forcing_ratio=0.0)  # y_logit: (batch_size, tgt_length, tgt_vocab_size)

        ## calculate loss
        # reshape and ignore first <sos> token
        y_logit = y_logit[:,1:]
        tgt_sent = tgt_sent[:, 1:]

        y_logit_flat = y_logit.reshape(-1, y_logit.shape[-1])
        tgt_sent_flat = tgt_sent.reshape(-1)

        loss = self.loss_fn(y_logit_flat, tgt_sent_flat)

        # calculate BLEU score
        bleu_score = self._calculate_bleu_score(y_logit, tgt_sent)

        # logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_bleu", bleu_score, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._log_translations(src_sent[:,1:], y_logit, tgt_sent, mode="val", n_examples=5)

        return loss
    
    def test_step(self, batch, batch_idx):

        src_sent, tgt_sent = batch 

        # forward pass 
        y_logit = self(src_sent, tgt_sent, teacher_forcing_ratio=0.0)  # y_logit: (batch_size, tgt_length, tgt_vocab_size)

        ## calculate loss
        # reshape and ignore first <sos> token
        y_logit = y_logit[:,1:]
        tgt_sent = tgt_sent[:, 1:]

        y_logit_flat = y_logit.reshape(-1, y_logit.shape[-1])
        tgt_sent_flat = tgt_sent.reshape(-1)

        loss = self.loss_fn(y_logit_flat, tgt_sent_flat)

        # calculate BLEU score
        bleu_score = self._calculate_bleu_score(y_logit, tgt_sent)

        # logging
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)
        self.log("test_bleu", bleu_score, on_epoch=True, prog_bar=True)

        if batch_idx == 0:
            self._log_translations(src_sent[:,1:], y_logit, tgt_sent, mode="test", n_examples=5)

        return loss
    
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer
    
    def on_train_epoch_end(self):

        # teacher forcing ratio decay
        self.teacher_forcing_ratio = max(0.1, self.teacher_forcing_ratio * self.hparams.teacher_forcing_decay)

    def _translate(self, idxs, vocab):
        """
        Translate from indices to vocab words. 
        """
        # idxs: (batch_size, seq_length)
        # return: List (batch_size) 

        texts = []

        # loop over examples in the batch
        for i in range(idxs.shape[0]):

            # predicted text (words)
            tokens = []
            for token_idx in idxs[i,:]:
                if token_idx.item() == vocab.stoi["<eos>"] or token_idx.item() == vocab.stoi["<pad>"]:
                    break
                tokens.append(vocab.itos[token_idx.item()])
            texts.append(" ".join(tokens))

        return texts

    def _calculate_bleu_score(self, y_logit, tgt_sent):

        pred = torch.argmax(y_logit, axis=2)    # argmax over vocab_size dimension

        # convert prediction and target to texts
        pred_texts = self._translate(pred, self.tgt_vocab)
        tgt_texts = self._translate(tgt_sent, self.tgt_vocab)

        # convert list to correct format for BLEUScore metric
        tgt_texts = [[txt] for txt in tgt_texts]
        bleu_score = self.bleu(pred_texts, tgt_texts)

        return bleu_score
    
    def _log_translations(self, src, y_logit, tgt, mode="train", n_examples=5):

        pred = torch.argmax(y_logit, axis=2)    # argmax over vocab_size dimension

        # convert to texts
        src_texts = self._translate(src[:n_examples], self.src_vocab)
        pred_texts = self._translate(pred[:n_examples], self.tgt_vocab)
        tgt_texts = self._translate(tgt[:n_examples], self.tgt_vocab)

        columns = ["Source", "Prediction", "Reference"]
        data = [[s, p, t] for s,p,t in zip(src_texts, pred_texts, tgt_texts)]
        table = wandb.Table(data=data, columns=columns)

        self.logger.experiment.log({f"sample_translations_{mode}_{self.global_step}": table})
