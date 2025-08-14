import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Encoder (bi-LSTM) --------
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, pad_idx=0, embeddings=None, dropout=0.1):
        super().__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings.vectors, freeze=True)
            self.embedding.padding_idx = getattr(embeddings, "stoi", {}).get("<pad>", pad_idx)
        else:
            self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, lengths):
        # src: [B, S]  | lengths: [B] (cpu, descending not required due to enforce_sorted=False)
        emb = self.dropout(self.embedding(src))                                   # [B, S, E]
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, S, 2H]
        # last layer, concat directions -> [B, 2H]
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        c_cat = torch.cat([c[-2], c[-1]], dim=1)
        return enc_out, h_cat, c_cat  # enc_out unused in baseline (no attention)


# -------- Decoder (uni-LSTM) --------
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, pad_idx=0, embeddings=None, dropout=0.1):
        super().__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings.vectors, freeze=True)
            self.embedding.padding_idx = getattr(embeddings, "stoi", {}).get("<pad>", pad_idx)
        else:
            self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(emb_dim, dec_hid_dim, batch_first=True)
        self.fc_out = nn.Linear(dec_hid_dim, output_dim)  # This will be dec_hid_dim -> output_dim
        self.dropout = nn.Dropout(dropout)

        # Bridge from encoder (2 * enc_hid_dim because of bidirectional) -> decoder (dec_hid_dim)
        self.h_bridge = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.c_bridge = nn.Linear(enc_hid_dim * 2, dec_hid_dim)


    def init_state(self, h_enc, c_enc):
        # h_enc/c_enc: [B, 2H] -> [1, B, H]
        h0 = torch.tanh(self.h_bridge(h_enc)).unsqueeze(0)
        c0 = torch.tanh(self.c_bridge(c_enc)).unsqueeze(0)
        return (h0, c0)

    def forward(self, y_prev, state):
        # y_prev: [B], state: (h, c)
        emb = self.dropout(self.embedding(y_prev)).unsqueeze(1)                   # [B, 1, E]
        out, state = self.rnn(emb, state)                                         # out: [B, 1, H]
        logits = self.fc_out(out.squeeze(1))                                      # [B, V]
        return logits, state


# -------- Seq2Seq wrapper --------
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def forward(self, src, src_lengths, tgt, teacher_forcing_ratio=0.5):
        # tgt: [B, T] with <sos> at position 0
        B, T = tgt.size()
        V = self.decoder.embedding.num_embeddings

        _, h_enc, c_enc = self.encoder(src, src_lengths)
        dec_state = self.decoder.init_state(h_enc, c_enc)

        y = tgt[:, 0]  # <sos>
        logits_all = []
        for t in range(1, T):
            logits, dec_state = self.decoder(y, dec_state)     # [B, V]
            logits_all.append(logits.unsqueeze(1))
            use_tf = torch.rand(B, device=tgt.device) < teacher_forcing_ratio
            y = torch.where(use_tf, tgt[:, t], logits.argmax(dim=-1))

        return torch.cat(logits_all, dim=1)  # [B, T-1, V]