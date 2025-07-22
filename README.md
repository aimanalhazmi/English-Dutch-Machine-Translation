# English-Dutch-Machine-Translation
_Natural Language Processing – Technische Universität Berlin (Master’s in Computer Science)_

## Overview
This project was developed for the Natural Language Processing course at Technische Universität Berlin. The goal is to implement a neural machine translation (NMT) system capable of translating between English and Dutch using RNN-based sequence-to-sequence models with and without attention mechanisms. A character-level model and a pivot translation extension are also explored.

---
## Dataset
We use the Europarl v7 English-Dutch parallel corpus, which consists of aligned parliamentary proceedings. It includes sentence pairs for 21 European languages; we extract and process only the English-Dutch portions.

- **Source**: [Europarl Parallel Corpus](https://www.statmt.org/europarl/)
- **Download Link**: [nl-en.tgz](https://www.statmt.org/europarl/v7/nl-en.tgz)

The corpus is too large for training, so 10% of the data is randomly sampled for all model training and evaluation.

---
## Objectives
- Analyze and visualize sentence-level statistics of the corpus
- Preprocess text to clean and normalize the dataset
- Build and evaluate:
    - Word-based RNN encoder-decoder models (EN→NL and NL→EN)
    - Character-based encoder-decoder models 
    - Enhanced models using attention mechanism

- Compare performance with different word embedding strategies (GloVe, Word2Vec, etc.)
- Optional: Build a pivot translation system (NL → EN → SV)
---
## Project Structure

```
english-dutch-machine-translation/
│
├── data/                     # Raw and sampled parallel corpus
├── src/                      # Python scripts for training, preprocessing, evaluation
├── outputs/                  # Saved model weights and configurations, Sample translations, plots, etc.
├── requirements.txt          # Python package dependencies
└── README.md                 # Project overview

```

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/aimanalhazmi/English-Dutch-Machine-Translation.git
cd English-Dutch-Machine-Translation
```

### 2. Set up the environment using `make`
```bash
make
```

This will:
- Create a virtual environment in `.venv/`
- Register a Jupyter kernel as **'translator'**
- Install all required packages from `requirements.txt`

### 3. Activate the environment
```bash
source .venv/bin/activate
```

## ⚙️ Makefile Commands

| Command             | Description                                                |
|---------------------|------------------------------------------------------------|
| `make install`      | Set up virtual environment and install dependencies        |
| `make activate`     | Print the command to activate the environment              |
| `make jupyter-kernel` | Register Jupyter kernel as `translator`          |
| `make remove-kernel`  | Unregister existing kernel (if needed)                  |
| `make clean`        | Delete the virtual environment folder                      |


---