import pandas as pd

def load_data(file):
    """Load data from a file"""
    with open(file, "r", encoding="utf-8") as data:
        return [line.strip() for line in data]


def load_data_as_df(en_file, nl_file):
    print(f"Loading English Corpora from: {en_file} ...")
    en_corpora = load_data(en_file)
    print(f"Loading Dutch Corpora from: {nl_file} ...")
    nl_corpora = load_data(nl_file)
    return pd.DataFrame({"English": en_corpora, "Dutch": nl_corpora})