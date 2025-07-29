import pandas as pd

def load_data(file):
    """Load data from a file"""
    with open(file, "r", encoding="utf-8") as data:
        return [line.strip() for line in data]


def load_data_as_df(source_file, source_col, target_file, target_col):
    print(f"Loading {source_col} Corpora from: {source_file} ...")
    en_corpora = load_data(source_file)
    print(f"Loading {target_col} Corpora from: {target_file} ...\n")
    nl_corpora = load_data(target_file)
    return pd.DataFrame({f"{source_col}": en_corpora, f"{target_col}": nl_corpora})