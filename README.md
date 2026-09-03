# Applied NLP Projects

A compact collection of natural language processing projects spanning probabilistic language modeling, spelling correction, text classification, feature engineering, neural networks, and transformer fine-tuning.

## Projects

| Project | Focus | Main Methods |
|---|---|---|
| [Probabilistic Spell Checker](./01-probabilistic-spell-checker/) | Language modeling & spelling correction | Noisy channel, n-grams, confusion matrices, edit distance, candidate indexing |
| [Authorship Attribution](./02-authorship-attribution/) | Text classification | TF-IDF, SVM, FFNN, POS features, metadata, BERTweet |

## Repository Structure

```text
.
├── 01-probabilistic-spell-checker/
│   ├── README.md
│   ├── probabilistic_spell_checker.py
│   └── requirements.txt
│
├── 02-authorship-attribution/
│   ├── README.md
│   ├── authorship_attribution.ipynb
│   └── requirements.txt
│
├── README.md
├── .gitignore
└── GITHUB_SETUP.md
```

## Highlights

- Implemented a context-sensitive noisy-channel spell checker from scratch.
- Used bounded dynamic programming and indexed candidate retrieval for efficient correction.
- Compared classical, neural, syntactic, hybrid, and transformer-based text classifiers.
- Fine-tuned BERTweet and evaluated models with stratified cross-validation.
- Explored lexical, character-level, syntactic, stylistic, and metadata-based representations.

## Data and Privacy

Datasets are intentionally not bundled with the repository. Credentials, API keys, and access tokens are excluded from the public version.
