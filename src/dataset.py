import os
import torch
from torch.utils.data import Dataset, DataLoader
from src.preprocessing import tokenizer_en, tokenizer_nl
import lightning

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
    

class TranslationDataModule(lightning.LightningDataModule):

    def __init__(self, train_df, val_df, test_df, src_lang, tgt_lang, src_vocab, tgt_vocab, batch_size=128):

        super().__init__()

        self.train_df = train_df 
        self.val_df = val_df
        self.test_df = test_df 
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang 
        self.src_vocab = src_vocab 
        self.tgt_vocab = tgt_vocab 
        self.batch_size = batch_size

    def setup(self, stage=None):

        self.train_dataset = TranslationDataset(
            df = self.train_df,
            src_vocab = self.src_vocab,
            tgt_vocab = self.tgt_vocab,
            src_col = self.src_lang,
            tgt_col = self.tgt_lang
        )

        self.val_dataset = TranslationDataset(
            df = self.val_df,
            src_vocab = self.src_vocab,
            tgt_vocab = self.tgt_vocab,
            src_col = self.src_lang,
            tgt_col = self.tgt_lang
        )

        self.test_dataset = TranslationDataset(
            df = self.test_df,
            src_vocab = self.src_vocab,
            tgt_vocab = self.tgt_vocab,
            src_col = self.src_lang,
            tgt_col = self.tgt_lang
        )

    def collate_fn(self, batch):

        srcs, tgts = zip(*batch)
        srcs_padded = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=self.src_vocab.stoi["<pad>"])  # padding value should be equal to the id of the <pad> token!
        tgts_padded = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=self.tgt_vocab.stoi["<pad>"])
        return srcs_padded, tgts_padded
    
    def train_dataloader(self):

        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=os.cpu_count())
    
    def val_dataloader(self):

        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=os.cpu_count())
    
    def test_dataloader(self):

        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=os.cpu_count())
