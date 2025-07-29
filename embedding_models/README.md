# Embedding Models

This folder is used to store **pretrained embedding vectors** (e.g., FastText, GloVe) required for the project.

##  Required Downloads

Before training or running the models, you **must download the following embeddings** and place them in this folder.

---

### GloVe (English)
- **Source:** [GloVe](https://nlp.stanford.edu/projects/glove/)
- **File needed:** `glove.6B.300d.txt`
- **Download instructions:**
  1. Download the file from the GloVe project page.
  2. Extract and place `glove.6B.300d.txt` here:
     ```
     embedding_models/glove.6B.300d.txt
     ```
---

### FastText (Dutch or Aligned Multilingual)
- **Source:** [FastText](https://fasttext.cc/docs/en/crawl-vectors.html)
- **Files needed:**
  - `cc.nl.300.vec` (Dutch)
  - `cc.en.300.vec` (English)
- **Download instructions:**
  1. For Dutch:
     - Download **`cc.nl.300.vec`**.
  2. For English:
     - Download **`cc.en.300.vec`**
  3. Place them here:
     ```
     embedding_models/cc.nl.300.vec
     embedding_models/cc.en.300.vec
     ```

---
