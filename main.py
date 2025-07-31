import os

import torch
from src.preprocessing import preprocess_dataframe, get_tokenized_vocab
from src.load import load_data_as_df
from src.utils import split_dataset, get_embedding_models_paths
from src.embedding import PretrainedEmbeddingVocab, EmbeddingVocab
from src import train
from src.visual import plot_loss, plot_bleu
import config
import time



if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

if __name__ == '__main__':
    cfg = config
    start = time.perf_counter()
    df = load_data_as_df(source_file=config.en_file, source_col=config.source_col, target_file=config.nl_file, target_col=config.target_col)

    df_clean = preprocess_dataframe(df=df, source_col=config.source_col, target_col=config.target_col)

    df_sampled = df_clean.sample(frac=config.sample_frac, random_state=config.random_state).reset_index(drop=True)
    print(f"[Info] Selected {len(df_sampled):,} rows out of {len(df_clean):,} ({config.sample_frac * 100:.1f}% of preprocessed data)")
    df_train, df_val, df_test = split_dataset(df_sampled, val_test_size=config.val_test_size, test_size=config.test_size, random_state=config.random_state)

    src_vocab_set = get_tokenized_vocab(df=df_sampled, lan=config.source_col)
    tgt_vocab_set = get_tokenized_vocab(df=df_sampled, lan=config.target_col)

    src_emb_path, tgt_emb_path = get_embedding_models_paths(source_col=config.source_col, target_col=config.target_col, method=config.embedding_method)

        # This is slower because it first loads the entire FastText file
    # src_vocab = EmbeddingVocab(embedding_path=src_emb_path, embedding_dim=config.embedding_dim, method=config.embedding_method, restrict_to_vocab=src_vocab_set, binary=False)
    # tgt_vocab = EmbeddingVocab(embedding_path=tgt_emb_path, embedding_dim=config.embedding_dim, method=config.embedding_method, restrict_to_vocab=tgt_vocab_set, binary=False)

    src_vocab = PretrainedEmbeddingVocab(embedding_path=src_emb_path, embedding_dim=config.embedding_dim, restrict_to_vocab=src_vocab_set)
    tgt_vocab = PretrainedEmbeddingVocab(embedding_path=tgt_emb_path, embedding_dim=config.embedding_dim, restrict_to_vocab=tgt_vocab_set)

    os.makedirs("outputs", exist_ok=True)
    train_loss, val_loss, bleu_scores = train.train_evaluate(df_train=df_train, df_val=df_val, src_vocab=src_vocab, tgt_vocab=tgt_vocab, device=DEVICE)

    # Plot
    plot_loss(train_loss, val_loss)
    plot_bleu(bleu_scores)

    end = time.perf_counter()
    elapsed = end - start
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"[Run Time] {minutes} min {seconds:.2f} sec")




