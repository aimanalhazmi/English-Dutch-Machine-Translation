import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, embeddings=None, dropout=0.5):
        super().__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)

        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, embeddings=None, dropout=0.5):
        super().__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)

        self.rnn = nn.GRU(emb_dim, hid_dim, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        return self.fc_out(output.squeeze(1)), hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        vocab_size = self.decoder.embedding.num_embeddings

        _, hidden = self.encoder(src)
        input = tgt[:, 0] #(batch_size, sequence_length)


        outputs = []

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden)  # (batch_size, vocab_size)
            outputs.append(output.unsqueeze(1))  # shape: (batch_size, 1, vocab_size)
            input = tgt[:, t] if torch.rand(1).item() < teacher_forcing_ratio else output.argmax(1)

        outputs = torch.cat(outputs, dim=1)  # shape: (batch_size, tgt_len - 1, vocab_size)
        return outputs

