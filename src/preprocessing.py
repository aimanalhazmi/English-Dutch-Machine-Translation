import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import spacy
import config
from tqdm import tqdm

tqdm.pandas()
nl_nlp = spacy.load("nl_core_news_sm")
en_nlp = spacy.load("en_core_web_sm")


def remove_stop_words(text, stopWords) -> str:
    if not isinstance(text, str):
        return ""
    tokens = text.lower().split()
    tokens = [word for word in tokens if word not in stopWords]
    return " ".join(tokens)

def tokenizer_en(text):
    return [token.text for token in en_nlp.tokenizer(text)]

def tokenizer_nl(text):
    return [token.text for token in nl_nlp.tokenizer(text)]


def get_tokenized_vocab(df, lan, min_freq=1):
    lan = lan.strip().capitalize()
    if lan == "English":
        tokenizer = tokenizer_en
    elif lan == "Dutch":
        tokenizer = tokenizer_nl
    else:
        raise ValueError("Language must be 'English' or 'Dutch'")

    tokens = [token for sent in df[lan] for token in tokenizer(sent)]
    counter = Counter(tokens)
    vocab = [word for word, freq in counter.items() if freq >= min_freq]
    return set(vocab)

def get_stopwords(lan: str) -> set:
    """
    Returns a set of stopwords for the given language.
    Supports only 'English' and 'Dutch'.
    """
    lan = lan.strip().capitalize()
    if lan == "English":
        return set(stopwords.words("english"))
    elif lan == "Dutch":
        nl_stop = set(stopwords.words("dutch"))
        nl_stop.update(config.custom_nl_stopwords)
        return nl_stop
    else:
        raise ValueError("Language must be 'English' or 'Dutch'")

def preprocess_text(text: str, lan: str) -> str:
    """
    Preprocesses a given text string by applying normalization, punctuation removal,
    stopword removal, and number removal based on configuration flags.

    Steps performed:
    1. Validates input: returns an empty string if input is not a valid non-empty string.
    2. Returns an empty string if the line starts with '<'.
    3. Converts text to lowercase and trims whitespace.
    4. Removes punctuation if `config.remove_punct` is True.
    5. Removes stopwords using the language-specific stopword list if `config.remove_stopwords` is True.
    6. Removes numeric digits if `config.remove_nums` is True.

    Args:
        text (str): The input text to preprocess.
        lan (str): The language for stopword removal ('English' or 'Dutch').

    Returns:
        str: The preprocessed text string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.strip()
    if text.startswith("<"):
        return ""
    text = text.lower()

    if config.remove_punct:
        text = "".join(char for char in text if char not in string.punctuation)

    if config.remove_stopwords:
        stop_words = get_stopwords(lan)
        tokens = word_tokenize(text)
        text = " ".join(word for word in tokens if word not in stop_words)

    if config.remove_nums:
        text = re.sub(r"\d+", "", text)

    return text

def preprocess_dataframe(df, source_col, target_col):

    print(f"[Start Preprocessing] Total raw rows: {len(df):,}")

    # Remove non-strings and nulls
    df = df[df[source_col].apply(lambda x: isinstance(x, str))]
    df = df[df[target_col].apply(lambda x: isinstance(x, str))]
    df = df.dropna(subset=[source_col, target_col])
    print(f"[Step 1] Rows after removing nulls/non-strings: {len(df):,}")

    # Preprocess text
    print("[Step 2] Preprocessing each row ...")
    df[source_col] = df[source_col].progress_apply(lambda x: preprocess_text(x, source_col))
    df[target_col] = df[target_col].progress_apply(lambda x: preprocess_text(x, target_col))

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(keep="first")
    print(f"[Step 3] Removed duplicates: {before - len(df):,} rows dropped ({len(df):,} remain)")

    # Remove rows with too short strings
    before = len(df)
    df = df[(df[source_col].str.strip() != "") & (df[target_col].str.strip() != "")]
    df = df[(df[source_col].str.len() >= config.min_len_chars) &
            (df[target_col].str.len() >= config.min_len_chars)]
    print(f"[Step 4] Removed empty/short rows: {before - len(df):,} rows dropped ({len(df):,} remain)")

    # Remove overly long sentences
    before = len(df)
    df = df[df.apply(
        lambda row: len(row[source_col].split()) <= config.max_len_tokens and
                    len(row[target_col].split()) <= config.max_len_tokens, axis=1)]
    print(f"[Step 5] Removed overly long rows: {before - len(df):,} rows dropped ({len(df):,} remain)")

    print(f"[Done] Final preprocessed rows: {len(df):,}\n")
    return df