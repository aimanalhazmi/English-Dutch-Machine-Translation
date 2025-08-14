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


def _infer_lengths(src, pad_idx: int):
    # src: [B, S] -> lengths: [B]
    return (src != pad_idx).sum(dim=1)


def evaluate_model(model, df_val, src_vocab, tgt_vocab, criterion, device="cpu", max_len=100):
    model.eval()
    val_ds = TranslationDataset(df_val, src_vocab, tgt_vocab)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

    total_bleu, total_loss, count = 0, 0, 0
    pad_idx_src = src_vocab.stoi.get("<pad>", 0)
    sos_idx =  tgt_vocab.stoi["<sos>"]
    eos_idx = tgt_vocab.stoi["<eos>"]
    pad_idx_tgt = tgt_vocab.stoi["<pad>"]

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                src, src_lengths, tgt = batch
            else:
                src, tgt = batch
                src_lengths = _infer_lengths(src, pad_idx_src)

            src, tgt = src.to(device), tgt.to(device)

            # ===== TEACHER FORCING LOSS (for monitoring training progress) =====
            tf_output = model(src, src_lengths, tgt, teacher_forcing_ratio=0.0)
            loss = criterion(tf_output.reshape(-1, tf_output.shape[-1]), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()

            # ===== TRUE INFERENCE (for BLEU score) =====
            batch_size = src.size(0)

            # Encode source
            _, h_enc, c_enc = model.encoder(src, src_lengths)
            dec_state = model.decoder.init_state(h_enc, c_enc)

            # Generate translations
            predictions = []
            for b in range(batch_size):
                generated = []
                y = torch.tensor([sos_idx], device=device)  # Start with <sos>
                current_state = (dec_state[0][:, b:b + 1, :], dec_state[1][:, b:b + 1, :])

                for _ in range(max_len):
                    logits, current_state = model.decoder(y, current_state)
                    pred_token = logits.argmax(dim=-1).item()

                    if pred_token == eos_idx:
                        break

                    generated.append(pred_token)
                    y = torch.tensor([pred_token], device=device)

                predictions.append(generated)

            # Calculate BLEU scores
            for i in range(batch_size):
                pred_seq = predictions[i]
                tgt_seq = cutoff_eos(tgt[i][1:].tolist(), eos_idx)

                pred_tokens = [tgt_vocab.itos[idx] for idx in pred_seq if idx != pad_idx_tgt]
                tgt_tokens = [tgt_vocab.itos[idx] for idx in tgt_seq if idx != pad_idx_tgt]

                if len(pred_tokens) > 0 and len(tgt_tokens) > 0:
                    bleu = sentence_bleu([tgt_tokens], pred_tokens, smoothing_function=SmoothingFunction().method1)
                    total_bleu += bleu
                count += 1

    avg_bleu = total_bleu / count if count else 0
    avg_loss = total_loss / len(val_loader) if len(val_loader) else 0
    return avg_bleu, avg_loss


def train_evaluate(df_train, df_val, src_vocab, tgt_vocab, device="cpu"):
    print(f"[Info] Training on: {device}")
    train_ds = TranslationDataset(
        df=df_train,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        src_col=config.source_col,
        tgt_col=config.target_col
    )
    print(f"[Info] Model will be trained to translate from {config.source_col} → {config.target_col}\n")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)

    # Build model
    encoder = Encoder(input_dim=len(src_vocab), emb_dim=config.embedding_dim, hid_dim=config.hidden_dim,
                      embeddings=src_vocab)
    decoder = Decoder(
        output_dim=len(tgt_vocab),
        emb_dim=config.embedding_dim,
        enc_hid_dim=config.hidden_dim,
        dec_hid_dim=config.hidden_dim,
        embeddings=tgt_vocab,
        dropout=config.dropout
    )
    model = Seq2Seq(encoder, decoder, pad_idx=tgt_vocab.stoi["<pad>"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab.stoi["<pad>"])

    train_losses, val_losses, bleu_scores = [], [], []
    best_val_loss = float("inf")
    pad_idx_src = src_vocab.stoi.get("<pad>", 0)

    for epoch in range(config.epoch):
        model.train()
        total_train_loss = 0
        teacher_forcing_ratio = max(0.3, 1.0 - epoch * 0.02)

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epoch}", leave=False)
        for batch in loop:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                src, src_lengths, tgt = batch
            else:
                src, tgt = batch
                src_lengths = _infer_lengths(src, pad_idx_src)

            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()
            output = model(src, src_lengths, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), tf_ratio=teacher_forcing_ratio)

        avg_train_loss = total_train_loss / len(train_loader)

        bleu, val_loss = evaluate_model(model, df_val, src_vocab, tgt_vocab, criterion, device)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            os.makedirs("outputs", exist_ok=True)
            torch.save(best_model_state, os.path.join("outputs", "model.pt"))
            print("[Info] Saving best model..")

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        bleu_scores.append(bleu)

        print(
            f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | BLEU: {bleu:.4f} | LR: {current_lr:.6f}")

    return train_losses, val_losses, bleu_scores