# Probabilistic Spell Checker

A context-sensitive spell checker based on the **noisy-channel model**, combining an n-gram language model with character-level error probabilities.

## Overview

The system handles both non-word and real-word spelling errors by balancing:

- the probability of the intended correction under an n-gram language model;
- character-level insertion, deletion, substitution, and transposition probabilities;
- edit distance and candidate frequency;
- sentence-level contextual likelihood.

## Implementation Highlights

- Unigram and general n-gram language models with Laplace smoothing
- Noisy-channel scoring with confusion matrices
- Bounded edit distance with adjacent transpositions
- Efficient candidate retrieval using a delete-index
- Context-sensitive sentence correction
- Model serialization and loading
- Support for both single-word and sentence-level correction

A bounded dynamic-programming edit-distance implementation and an indexed candidate search are used to avoid brute-force vocabulary comparisons.

## Tech Stack

- Python
- NumPy
- Standard-library data structures and serialization

## Run

Install the dependency:

```bash
pip install -r requirements.txt
```

The implementation expects an external text corpus and, optionally, error/confusion tables. These data files are not included in the repository.

Main file:

```text
probabilistic_spell_checker.py
```

## Privacy

Student ID and university email fields required by the original submission API were removed from the public version.
