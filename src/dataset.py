import torch
from torch.utils.data import Dataset
from src.preprocessing import tokenizer_en, tokenizer_nl

class TranslationDataset(Dataset):
    def __init__(self, df, src_vocab, tgt_vocab, src_col='English', tgt_col='Dutch'):
        src = src_col.strip().lower()
        tgt = tgt_col.strip().lower()

        if src == 'english' and tgt == 'dutch':
            self.src_texts = df[src_col].apply(tokenizer_en).tolist()
            self.tgt_texts = df[tgt_col].apply(tokenizer_nl).tolist()
        elif src == 'dutch' and tgt == 'english':
            self.src_texts = df[src_col].apply(tokenizer_nl).tolist()
            self.tgt_texts = df[tgt_col].apply(tokenizer_en).tolist()
        else:
            raise ValueError(
                f"Unsupported language pair: {src_col} → {tgt_col}. "
                "Only English ↔ Dutch is supported.")

        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.src_vocab.numericalize(["<sos>"] + self.src_texts[idx] + ["<eos>"])
        tgt = self.tgt_vocab.numericalize(["<sos>"] + self.tgt_texts[idx] + ["<eos>"])
        return torch.tensor(src), torch.tensor(tgt)
