import copy
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.dataset import TranslationDataset
from src.model import Encoder, Decoder, Seq2Seq
from src.utils import collate_fn
import config

def cutoff_eos(tokens, eos_idx):
    return tokens[:tokens.index(eos_idx)] if eos_idx in tokens else tokens

def evaluate_model(model, df_val, src_vocab, tgt_vocab, criterion, device="cpu"):
    model.eval()
    val_ds = TranslationDataset(df_val, src_vocab, tgt_vocab)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

    total_bleu, total_loss, count = 0, 0, 0

    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc="Evaluating", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()

            preds = output.argmax(dim=-1)
            for i in range(preds.size(0)):
                pred_seq = cutoff_eos(preds[i].tolist(), tgt_vocab.stoi["<eos>"])
                tgt_seq = cutoff_eos(tgt[i][1:].tolist(), tgt_vocab.stoi["<eos>"])
                pred_tokens = [tgt_vocab.itos[idx] for idx in pred_seq if idx != tgt_vocab.stoi["<pad>"]]
                tgt_tokens = [tgt_vocab.itos[idx] for idx in tgt_seq if idx != tgt_vocab.stoi["<pad>"]]
                bleu = sentence_bleu([tgt_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)
                total_bleu += bleu
                count += 1

    avg_bleu = total_bleu / count if count else 0
    avg_loss = total_loss / len(val_loader) if len(val_loader) else 0
    return avg_bleu, avg_loss



def train_evaluate(df_train, df_val, src_vocab, tgt_vocab, device="cpu"):
    print(f"[Info] Training on: {device}")
    train_ds = TranslationDataset(df=df_train, src_vocab=src_vocab, tgt_vocab=tgt_vocab, src_col=config.source_col, tgt_col=config.target_col)
    print(f"[Info] Model will be trained to translate from {config.source_col} → {config.target_col}\n")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    encoder = Encoder(len(src_vocab), config.embedding_dim, hid_dim=config.embedding_dim, embeddings=src_vocab)
    decoder = Decoder(len(tgt_vocab), config.embedding_dim, hid_dim=config.embedding_dim, embeddings=tgt_vocab)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi["<pad>"])

    train_losses, val_losses, bleu_scores = [], [], []
    best_val_loss = float("inf")

    for epoch in range(config.epoch):
        model.train()
        total_train_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epoch}", leave=False)
        for src, tgt in loop:
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        bleu, val_loss = evaluate_model(model, df_val, src_vocab, tgt_vocab, criterion, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, os.path.join("outputs", "model.pt"))
            print("[Info] Saving best model..")

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        bleu_scores.append(bleu)

        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu:.4f}")

    return train_losses, val_losses, bleu_scores
