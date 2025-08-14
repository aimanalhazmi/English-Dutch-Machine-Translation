


en_file = "data/europarl-v7.nl-en.en"
source_col = "Dutch" #English
nl_file = "data/europarl-v7.nl-en.nl"
target_col = "English" # Dutch


embedding_dim = 300
embedding_method = "fasttext" #fasttext, glove_fasttext


remove_punct = False # Keep punctuation for translation context
remove_stopwords = False # Keep all words since stopwords matter in translation
remove_nums = True
min_len_chars = 2
max_len_tokens = 200


sample_frac = 0.01
random_state = 42
# ~60% train
val_test_size = 0.4
# Split from remaining 40% into 20% val and 20% test
test_size = 0.5

custom_nl_stopwords = ["we", "wij", "onze"]

batch_size = 64

# MODEL
lr = 0.01
freeze = True
dropout = 0.3
epoch = 10
hidden_dim = 512