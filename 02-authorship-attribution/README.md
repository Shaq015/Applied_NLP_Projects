# Authorship Attribution from Social Media Text

A binary text-classification study comparing classical machine learning, neural networks, syntactic features, metadata, and transformer fine-tuning for authorship attribution.

## Problem

The task is to infer whether a tweet from a public account was authored by Donald Trump or by staff, using historical tweet data and device-derived labels.

The implementation filters and preprocesses the corpus, then evaluates multiple representations and classifier families under stratified cross-validation.

## Models

Six approaches are implemented:

1. **Logistic Regression** with word- and character-level TF-IDF features
2. **Linear SVM** with TF-IDF features
3. **PyTorch FFNN** with three hidden layers on TF-IDF vectors
4. **POS Random Forest** using POS unigrams, bigrams, trigrams, and grammar-oriented features
5. **Hybrid RBF SVM** combining lexical, syntactic, stylistic, and metadata features
6. **BERTweet** fine-tuned for binary authorship classification

## Selected Cross-Validation Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| BERTweet | **0.9154** | 0.8570 | **0.8693** | **0.8626** |
| Hybrid RBF SVM | 0.9034 | **0.8785** | 0.7938 | 0.8337 |
| Logistic Regression | 0.8812 | 0.8425 | 0.7536 | 0.7950 |
| Linear SVM | 0.8754 | 0.8261 | 0.7536 | 0.7869 |
| PyTorch FFNN | 0.8609 | 0.7976 | 0.7344 | 0.7631 |
| POS Random Forest | 0.8480 | 0.8449 | 0.6158 | 0.7116 |

BERTweet achieved the strongest mean accuracy and F1 score in the reported 5-fold evaluation.

## Feature Engineering

The project explores several complementary signals:

- word and character n-grams;
- POS-tag sequences and local syntactic patterns;
- capitalization and punctuation style;
- tweet metadata and posting-time features;
- contextual transformer representations.

## Tech Stack

- Python
- Pandas / NumPy / SciPy
- NLTK
- Scikit-learn
- PyTorch
- Hugging Face Transformers / Datasets
- BERTweet

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

The tweet datasets are not included. Update the training and test paths in the notebook before running:

```text
authorship_attribution.ipynb
```
