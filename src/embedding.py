import numpy as np
import torch
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

class PretrainedEmbeddingVocab:
    def __init__(self, embedding_path, embedding_dim, restrict_to_vocab=None):
        """Loads pretrained embeddings and builds vocab mappings."""
        self.itos = []  # index-to-string
        self.stoi = {}  # string-to-index
        self.vectors = []

        print(f"[info] Loading pretrained embedding from {embedding_path}]")

        with open(embedding_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                values = line.strip().split()

                if line_num == 0 and len(values) == 2 and values[0].isdigit():
                    print(f"[Info] Skipping header line: vocab_size={values[0]}, dim={values[1]}")
                    continue

                word = values[0]
                vector = list(map(float, values[1:]))

                if len(vector) != embedding_dim:
                    continue

                # Restrict vocabulary if needed
                if restrict_to_vocab and word not in restrict_to_vocab:
                    continue

                self.stoi[word] = len(self.itos)
                self.itos.append(word)
                self.vectors.append(vector)

        # Add special tokens
        for token in ["<pad>", "<sos>", "<eos>", "<unk>"]:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
                self.vectors.append([0.0] * embedding_dim)

        # Convert vectors to tensor
        self.vectors = torch.tensor(np.array(self.vectors), dtype=torch.float32)
        print(f"[Done] Loaded {len(self.itos):,} tokens with {embedding_dim}-dim embeddings\n")

    def numericalize(self, tokens):
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]

    def __len__(self):
        return len(self.itos)



class EmbeddingVocab:
    def __init__(self, embedding_path, embedding_dim, method="fasttext", restrict_to_vocab=None, binary=False):
        """Loads pretrained embeddings (GloVe, FastText, or Word2Vec) and builds vocab mappings."""
        self.itos = []  # index-to-string
        self.stoi = {}  # string-to-index
        self.vectors = []

        print(f"[info] Loading {method} embeddings from {embedding_path}")

        #  GloVe (convert to Word2Vec format first)
        if method.lower() == "glove":
            w2v_temp = embedding_path.replace(".txt", ".w2v.txt")
            glove2word2vec(embedding_path, w2v_temp)
            model = KeyedVectors.load_word2vec_format(w2v_temp, binary=False)

        #  FastText (.vec text format)
        elif method.lower() == "fasttext":
            model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)

        #  Word2Vec (.bin or .txt)
        elif method.lower() == "word2vec":
            model = KeyedVectors.load_word2vec_format(embedding_path, binary=binary)

        else:
            raise ValueError("method must be one of ['glove', 'fasttext', 'word2vec']")

        # Build vocab (restricted if needed)
        words = restrict_to_vocab if restrict_to_vocab else model.key_to_index.keys()
        for word in words:
            if word in model:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
                self.vectors.append(model[word])

        # Add special tokens
        for token in ["<pad>", "<sos>", "<eos>", "<unk>"]:
            if token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
                self.vectors.append([0.0] * embedding_dim)

        self.vectors = torch.tensor(np.array(self.vectors), dtype=torch.float32)
        print(f"[Done] Loaded {len(self.itos):,} tokens with {embedding_dim}-dim {method} embeddings")

    def numericalize(self, tokens):
        return [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]

    def __len__(self):
        return len(self.itos)
