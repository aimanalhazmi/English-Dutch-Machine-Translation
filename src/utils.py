import os
from sklearn.model_selection import train_test_split
import torch


def split_dataset(df, val_test_size=0.3, test_size=0.66, random_state=42):
    """
    Splits a dataset into train, validation, and test sets.
        df (pd.DataFrame): Input dataset.
        val_test_size (float): Fraction for (validation + test) combined split
                               (e.g., 0.3 results in 70% train, 30% temp (val+test)).
        test_size (float): Fraction of the temp set to allocate for test
                           (e.g., 0.66 → splits 30% temp into ~10% val, ~20% test).
    """
    assert 0 < val_test_size < 1, "val_test_size must be between 0 and 1"
    assert 0 < test_size < 1, "test_size must be between 0 and 1"

    # Split train vs. (val+test)
    df_train, df_temp = train_test_split(
        df, test_size=val_test_size, random_state=random_state
    )

    # Split (val+test) into validation and test
    df_val, df_test = train_test_split(
        df_temp, test_size=test_size, random_state=random_state
    )

    total = len(df)
    print(f"[Split] Train: {len(df_train):,} ({len(df_train)/total:.1%})")
    print(f"[Split] Val:   {len(df_val):,} ({len(df_val)/total:.1%})")
    print(f"[Split] Test:  {len(df_test):,} ({len(df_test)/total:.1%})\n")

    return df_train, df_val, df_test

def get_embedding_models_paths(source_col, target_col, method):
    """
    Returns source and target embedding paths based on the given language columns and method.
    Automatically swaps embeddings if the source language is Dutch.
    """
    source_col = source_col.lower()
    target_col = target_col.lower()
    method = method.lower()

    if source_col == "english" and target_col == "dutch":
        src_emb_path = get_en_embedding_path(method)
        tgt_emb_path = get_nl_embedding_path(method)
    elif source_col == "dutch" and target_col == "english":
        src_emb_path = get_nl_embedding_path(method)
        tgt_emb_path = get_en_embedding_path(method)
    else:
        raise ValueError("Supported language pairs are only English ↔ Dutch.")

    return src_emb_path, tgt_emb_path


def get_en_embedding_path(method, folder="embedding_models"):
    """
    Returns the English embedding file path based on the method.
    """
    if method == "fasttext":
        emb_file = "cc.en.300.vec"  # English FastText
    elif method == "glove_fasttext":
        emb_file = "glove.6B.300d.txt"  # English GloVe
    else:
        raise ValueError("Invalid method. Use 'fasttext' or 'glove_fasttext'.")

    return _validate_embedding_path(folder, emb_file, lang="English")


def get_nl_embedding_path(method, folder="embedding_models"):
    """
    Returns the Dutch embedding file path based on the method.
    """
    if method in ["fasttext", "glove_fasttext"]:
        emb_file = "cc.nl.300.vec"  # Dutch FastText
    else:
        raise ValueError("Invalid method. Use 'fasttext' or 'glove_fasttext'.")

    return _validate_embedding_path(folder, emb_file, lang="Dutch")


def _validate_embedding_path(folder, filename, lang):
    """
    Helper to build full path and validate existence.
    """
    emb_path = os.path.join(folder, filename)
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"{lang} embedding file not found: {emb_path}")
    print(f"[Info] Selected {lang} embedding: {emb_path}")
    return emb_path



def collate_fn(batch):
    srcs, tgts = zip(*batch)
    srcs_padded = torch.nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=0)
    tgts_padded = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=0)
    return srcs_padded, tgts_padded
