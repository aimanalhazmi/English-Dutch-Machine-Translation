


en_file = "data/europarl-v7.nl-en.en"
source_col = "English"
nl_file = "data/europarl-v7.nl-en.nl"
target_col = "Dutch"


embedding_dim = 300
embedding_method = "fasttext"


remove_punct = False # Keep punctuation for translation context
remove_stopwords = False # Keep all words since stopwords matter in translation
remove_nums = True
min_len_chars = 3
max_len_tokens = 100


sample_frac = 0.01
random_state = 42
# ~70% train
val_test_size = 0.3
# Split from remaining 30% into ~10% val and (0.66) ~20% test
test_size = 0.66

custom_nl_stopwords = ["we", "wij", "onze"]

batch_size = 32

# MODEL
epoch = 5
hidden_dim = 512