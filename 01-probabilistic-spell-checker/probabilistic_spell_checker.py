import numpy as np
import re
import sys
import random
import math
import collections
import pickle


class Spell_Checker:

    def __init__(self, lm=None, n=1, alpha = 0.95, edits = 1, errors_in_text = 1, smoothing_method = 'laplace'):
        """
        Initializes the spell checker.
        Note: In testing we will use the default values of the edits, errors_in_text, smoothing_method parameters.
              However, you can experiment with different values in order to experience the impact on performance.
        Args:
            lm: (obj) a LM object to initiate the Spell Checker with (optional).
                    If not None, assume it is an object of the same type returned by build_model().
            n: (int) the n of the language model object passed or the one to be built. Defaults to 1 (unigram LM).
            alpha: (float) the alpha parameter of the speccl checker (likelihood of 'no error' in the considered word).
            edits: (int) the max edit distance to consider in a word (consider sub and transpose as one edit)
            errors_in_text: (int) the max number of erronous word to consider in a sentence.
            smoothing_method: (str) the smothing method to be used.
                              (Laplace must be supported. You are not required to support other smoothing methods)
        """
        self.n = n
        self.alpha = alpha
        self.edits = edits
        self.errors_in_text = errors_in_text
        self.smoothing_method = smoothing_method

        # I build later a delete-index up to 2 because it is used as a fast approximation before verifying candidates
        self.max_index_edits = 2

        # Characters used by the error model smoothing
        self.alphabet = "abcdefghijklmnopqrstuvwxyz'"

        # Language model containers
        self.lm = None
        self.unigram_counts = collections.Counter()
        self.ngram_counts = collections.Counter()
        self.context_counts = collections.Counter()
        self.vocab = set()
        self.total_tokens = 0

        # Auxiliary lookup structures
        self.words_by_len = collections.defaultdict(set)
        self.delete_index = collections.defaultdict(set)

        # Error model tables, loaded from provided confusion-matrix file
        self.error_tables = {
            "insertion": collections.Counter(),
            "deletion": collections.Counter(),
            "substitution": collections.Counter(),
            "transposition": collections.Counter()
        }

        # Character statistics are used as denominators in the noisy-channel model
        self.char_unigrams = collections.Counter()
        self.char_bigrams = collections.Counter()

        # If a language model object was passed directly to the constructor, copy it into the current object
        if lm is not None:
            self._load_lm_into_state(lm)
            self.n = lm.get("n", n)

    def build_model(self, text_corpus_path, n=1):
        """
        Returns a language model object of the specified n  built from a text file.
        Note 1: you can implement the LM the way you like (e.g., implement an inner class, use a sictionary, etc.), as it is
                transparent to us. However, the implementation may effect efficiancy.


        Args:
            text_corpus_path: (str) full path to a single corpus file.
            n: (int) the n value of the Markovian n-gram LM. Defaults to 1 (a unigram model)
        """
        self.n = n

        # Reset previous model state so multiple build_model calls do not mix counts
        self.unigram_counts = collections.Counter()
        self.ngram_counts = collections.Counter()
        self.context_counts = collections.Counter()
        self.vocab = set()
        self.total_tokens = 0
        self.words_by_len = collections.defaultdict(set)
        self.delete_index = collections.defaultdict(set)

        with open(text_corpus_path, "r", encoding="utf8", errors="ignore") as f:
            raw_text = f.read()

        sentences = []

        # If the marker <s> exists, use it as a sentence boundary
        if "<s>" in raw_text:
            raw_text = raw_text.lower()
            raw_sentences = raw_text.split("<s>")

            for sent in raw_sentences:
                clean_sent = normalize_text(sent)
                tokens = clean_sent.split()
                if tokens:
                    sentences.append(tokens)
        else:
            # For regular raw text, create rough sentence boundaries before normalization
            raw_text = str(raw_text).lower()
            raw_text = re.sub(r"[.!?]+", " <eos> ", raw_text)
            raw_sentences = raw_text.split("<eos>")

            for sent in raw_sentences:
                clean_sent = normalize_text(sent)
                tokens = clean_sent.split()
                if tokens:
                    sentences.append(tokens)

        # Count unigrams and n-grams
        for tokens in sentences:
            for word in tokens:
                self.unigram_counts[word] += 1
                self.vocab.add(word)
                self.total_tokens += 1

            # Padding
            padded_tokens = ["<s>"] * (n - 1) + tokens + ["</s>"]

            for i in range(len(padded_tokens) - n + 1):
                ngram = tuple(padded_tokens[i:i + n])

                if n == 1:
                    context = ()
                else:
                    context = tuple(padded_tokens[i:i + n - 1])

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        # Build lookup helpers after the full vocabulary is known
        for word in self.vocab:
            self.words_by_len[len(word)].add(word)

        self._build_char_stats()
        self._build_delete_index()

        # Store the LM as a dictionary
        self.lm = {
            "n": self.n,
            "unigram_counts": self.unigram_counts,
            "ngram_counts": self.ngram_counts,
            "context_counts": self.context_counts,
            "vocab": self.vocab,
            "total_tokens": self.total_tokens,
            "delete_index": dict(self.delete_index),
            "max_index_edits": self.max_index_edits
        }

        return self.lm

    def save_model(self, path, model_name):
        """
        Saves the model as a picke file to the specified path with the specified name.
        Args:
            path: the path to save the LM model to.
            model_name: the file name to use, with out the pkl suffix (so the file will be <name>.pkl).
        """
        if path.endswith("/") or path.endswith("\\"):
            full_path = path + model_name + ".pkl"
        else:
            full_path = path + "/" + model_name + ".pkl"

        with open(full_path, "wb") as f:
            pickle.dump(self.lm, f)

    def load_model(self, full_path):
        """
        Loads a LM model from a pickle file in the specified location. Tassume it is an object of the same type
        returned by build_model().
        Args:
            full_path: (str) the full path of the location of the LM pickle object.
        """
        with open(full_path, "rb") as f:
            lm = pickle.load(f)

        self._load_lm_into_state(lm)

    def load_error_tables(self, full_path):
        """
        Loads the error tables file (the file should be in the same format and semantics of the provided error table).
        Note: If you'd like to experiment you can updte the tables provided, create new (better?) ones or create toy
        ones tailored for testing.
        """
        # Reset tables first, so loading a new file replaces the old one
        self.error_tables = {
            "insertion": collections.Counter(),
            "deletion": collections.Counter(),
            "substitution": collections.Counter(),
            "transposition": collections.Counter()
        }

        local_scope = {}

        with open(full_path, "r", encoding="utf8", errors="ignore") as f:
            file_text = f.read()

        exec(file_text, {}, local_scope)

        if "error_tables" not in local_scope:
            raise ValueError("The loaded file does not contain 'error_tables'.")

        loaded_tables = local_scope["error_tables"]

        # Convert each dictionary into a Counter for convenient missing-key behavior
        for error_type in self.error_tables:
            if error_type in loaded_tables:
                self.error_tables[error_type] = collections.Counter(loaded_tables[error_type])

        # Character statistics depend on the LM vocabulary, so refresh them if needed
        if self.vocab:
            self._build_char_stats()

    def spellcheck_file(self, input_full_path, output_input_full_path):
        """
        Reads a .txt 'input_file' (one sentence per line), corrects it, and writes the result
        to a .txt 'output_file' (one sentence per line, lines in the input and output files should correspond).


        CRITICAL: This function must run in < 100 seconds for 100 sentences (of length <12).


        Args:
            input_full_path: (str) the full path to the input file.
            output_input_full_path: (str) the full path to the output file.
        """
        with open(input_full_path, "r", encoding="utf8", errors="ignore") as fin, \
             open(output_input_full_path, "w", encoding="utf8") as fout:

            for line in fin:
                sentence = line.rstrip("\n")
                corrected_sentence = self.spellcheck_sentence(sentence)
                fout.write(corrected_sentence + "\n")

    def correction(self, word):
        """
        Returns the most probable correction for a single word
        (without context).
        """
        word = normalize_text(word)

        if not word:
            return word

        word = word.split()[0]
        candidates = self._collect_candidates(word, self.edits)

        # If the word is not in the vocabulary, keeping it unchanged is usually not useful when there are alternatives
        if word not in self.vocab and len(candidates) > 1 and word in candidates:
            candidates.remove(word)

        best_candidate = word
        best_score = -float("inf")

        total = self.total_tokens if self.total_tokens > 0 else 1
        vocab_size = len(self.vocab) + 1

        for candidate in candidates:
            # Prior probability from the unigram model
            prior = (self.unigram_counts[candidate] + 1) / (total + vocab_size)

            # Small frequency bonus
            freq_bonus = math.log(self.unigram_counts[candidate] + 1)

            # Error/noisy-channel components
            typo_prob = self._score_error(word, candidate)
            dist_bonus = self._distance_bonus(word, candidate)
            len_bonus = self._length_bonus(word, candidate)
            ins_bonus = self._insertion_bonus(word, candidate)

            score = (
                math.log(max(prior, 1e-12))
                + math.log(max(typo_prob, 1e-12))
                + math.log(max(dist_bonus, 1e-12))
                + math.log(max(len_bonus, 1e-12))
                + math.log(max(ins_bonus, 1e-12))
                + 0.3 * freq_bonus
            )

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def spellcheck_sentence(self, sentence):
        """
        Returns the corrected sentence (string).
        """
        tokens = normalize_text(sentence).split()

        if not tokens:
            return ""

        # I assume at most one erroneous word in a sentence
        # Instead of searching all combinations, I try changing exactly one word at a time and select the best resulting sentence (strategy similar to Beam Search)
        original_lm_score = self.evaluate_text(" ".join(tokens))
        best_tokens = tokens[:]

        # Score threshold for keeping the sentence unchanged
        # If the sentence has an unknown word, keeping it is less reliable
        if any(tok not in self.vocab for tok in tokens):
            best_score = -3.0
        else:
            best_score = math.log(max(self.alpha, 1e-12))

        for i in range(len(tokens)):
            original_word = tokens[i]
            candidates = self._collect_candidates(original_word, self.edits)

            # For an unknown word, do not keep it unchanged if real alternatives exist
            if original_word not in self.vocab and len(candidates) > 1 and original_word in candidates:
                candidates.remove(original_word)

            for candidate in candidates:
                if candidate == original_word:
                    continue

                new_tokens = tokens[:]
                new_tokens[i] = candidate

                # Score the new sentence using the LM, then compare it to the original sentence by using the LM gain
                # This avoids depending only on absolute log-likelihood values, which are always negative
                new_lm_score = self.evaluate_text(" ".join(new_tokens))
                lm_gain = new_lm_score - original_lm_score

                # Error model and additional simple preferences
                typo_prob = self._score_error(original_word, candidate)
                dist_bonus = self._distance_bonus(original_word, candidate)
                len_bonus = self._length_bonus(original_word, candidate)
                ins_bonus = self._insertion_bonus(original_word, candidate)

                # Replacing an unknown word is more plausible than replacing a known word
                if original_word not in self.vocab:
                    oov_bonus = 80.0
                else:
                    oov_bonus = 1.0

                # Small advantage for common words
                freq_bonus = math.log(self.unigram_counts[candidate] + 1)

                # If both words are known, use only positive context evidence from the LM
                if original_word in self.vocab and candidate in self.vocab:
                    context_bonus = max(0.0, lm_gain) * 0.7
                else:
                    context_bonus = 0.0

                score = (
                    lm_gain
                    + math.log(max(typo_prob, 1e-12))
                    + math.log(max(dist_bonus, 1e-12))
                    + math.log(max(len_bonus, 1e-12))
                    + math.log(max(ins_bonus, 1e-12))
                    + math.log(max(oov_bonus, 1e-12))
                    + 0.3 * freq_bonus
                    + context_bonus
                )

                # Prevent weak corrections from changing already-correct sentences
                # This reduced over-correction in my experiments
                if score > best_score + 1.0:
                    best_score = score
                    best_tokens = new_tokens

        return " ".join(best_tokens)

    def evaluate_text(self, text):
        """
        Returns the log-likelihood of the specified text to be a product of the model used.
        """
        tokens = normalize_text(text).split()

        if not tokens:
            return 0.0

        vocab_size = len(self.vocab) + 1

        # Unigram model with Laplace smoothing
        if self.n == 1:
            log_likelihood = 0.0

            for word in tokens:
                prob = (self.unigram_counts[word] + 1) / (self.total_tokens + vocab_size)
                log_likelihood += math.log(prob)

            return log_likelihood

        # General n-gram model with Laplace smoothing
        padded_tokens = ["<s>"] * (self.n - 1) + tokens + ["</s>"]
        log_likelihood = 0.0

        for i in range(len(padded_tokens) - self.n + 1):
            ngram = tuple(padded_tokens[i:i + self.n])
            context = tuple(padded_tokens[i:i + self.n - 1])

            prob = (self.ngram_counts[ngram] + 1) / (self.context_counts[context] + vocab_size)
            log_likelihood += math.log(prob)

        return log_likelihood


    ########################### HELPERS ###########################

    def _load_lm_into_state(self, lm):
        """
        Copies a loaded language model object into the current checker state
        """
        self.lm = lm
        self.n = lm.get("n", 1)
        self.unigram_counts = lm.get("unigram_counts", collections.Counter())
        self.ngram_counts = lm.get("ngram_counts", collections.Counter())
        self.context_counts = lm.get("context_counts", collections.Counter())
        self.vocab = lm.get("vocab", set())
        self.total_tokens = lm.get("total_tokens", 0)

        # Rebuild length lookup from vocabulary
        self.words_by_len = collections.defaultdict(set)
        for word in self.vocab:
            self.words_by_len[len(word)].add(word)

        # Reuse stored delete-index if available, otherwise rebuild it
        self.delete_index = collections.defaultdict(set)
        loaded_delete_index = lm.get("delete_index", None)

        if loaded_delete_index is not None:
            for key in loaded_delete_index:
                self.delete_index[key] = set(loaded_delete_index[key])
        else:
            self._build_delete_index()

        self.max_index_edits = lm.get("max_index_edits", 2)
        self._build_char_stats()

    def _build_char_stats(self):
        """
        Builds character unigram and bigram counts from the current vocabulary
        """
        self.char_unigrams = collections.Counter()
        self.char_bigrams = collections.Counter()

        for word in self.vocab:
            freq = self.unigram_counts[word]

            if freq <= 0:
                freq = 1

            for ch in word:
                self.char_unigrams[ch] += freq

            # Marks word start, useful for first-character insertions/deletions
            extended = "#" + word
            for i in range(len(extended) - 1):
                self.char_bigrams[extended[i:i + 2]] += freq

    def _build_delete_index(self):
        """
        Builds a delete-key index for fast candidate lookup
        """
        self.delete_index = collections.defaultdict(set)

        for word in self.vocab:
            # Store the original word as a key too
            self.delete_index[word].add(word)

            # Store all keys made by deleting up to max_index_edits characters
            # This helps retrieve nearby words without scanning the full vocabulary
            delete_keys = self._generate_delete_keys(word, self.max_index_edits)
            for key in delete_keys:
                self.delete_index[key].add(word)

    def _generate_delete_keys(self, word, max_deletes):
        """
        Generates all strings formed by deleting up to max_deletes characters
        """
        results = set()
        queue = [(word, 0)]

        while queue:
            current_word, depth = queue.pop()

            if depth == max_deletes:
                continue

            for i in range(len(current_word)):
                new_word = current_word[:i] + current_word[i + 1:]

                if new_word not in results:
                    results.add(new_word)
                    queue.append((new_word, depth + 1))

        return results

    def _collect_candidates(self, word, max_dist):
        """
        Collects correction candidates for a word using the delete index
        and then verifies them with bounded edit distance
        """
        candidates = {word}

        # Retrieve a rough candidate pool by comparing delete keys
        rough_pool = set()
        lookup_keys = {word} | self._generate_delete_keys(word, min(max_dist, self.max_index_edits))

        for key in lookup_keys:
            if key in self.delete_index:
                rough_pool |= self.delete_index[key]

        # Remove impossible candidates by length before DP
        min_len = len(word) - max_dist
        max_len = len(word) + max_dist

        for candidate in rough_pool:
            cand_len = len(candidate)

            if cand_len < min_len or cand_len > max_len:
                continue

            # Final verification uses bounded dynamic programming
            dist = self._bounded_edit_distance(word, candidate, max_dist)

            if dist <= max_dist:
                candidates.add(candidate)

        return candidates

    def _bounded_edit_distance(self, a, b, max_dist):
        """
        Computes bounded edit distance with adjacent transposition counted as one edit.
        Returns max_dist + 1 if the distance is larger
        """
        if a == b:
            return 0

        if abs(len(a) - len(b)) > max_dist:
            return max_dist + 1

        # Use the shorter word as the column dimension to save memory
        if len(b) > len(a):
            a, b = b, a

        m, n = len(a), len(b)
        INF = max_dist + 1

        # prev_prev is needed for adjacent transposition
        prev_prev = [INF] * (n + 1)
        prev = [INF] * (n + 1)

        # Initialize only the relevant bounded band
        for j in range(0, min(n, max_dist) + 1):
            prev[j] = j

        for i in range(1, m + 1):
            curr = [INF] * (n + 1)

            if i <= max_dist:
                curr[0] = i

            # Compute only cells close to the diagonal because max_dist is small
            j_start = max(1, i - max_dist)
            j_end = min(n, i + max_dist)

            if j_start > j_end:
                return max_dist + 1

            row_min = INF
            ai = a[i - 1]

            for j in range(j_start, j_end + 1):
                bj = b[j - 1]
                cost = 0 if ai == bj else 1

                delete_cost = prev[j] + 1
                insert_cost = curr[j - 1] + 1
                sub_cost = prev[j - 1] + cost

                best = min(delete_cost, insert_cost, sub_cost)

                if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                    best = min(best, prev_prev[j - 2] + 1)

                curr[j] = best

                if best < row_min:
                    row_min = best

            # Stop early if even the best value in this row exceeds the allowed distance
            if row_min > max_dist:
                return max_dist + 1

            prev_prev, prev = prev, curr

        return prev[n]

    def _score_error(self, error_word, correct_word):
        """
        Estimates the probability of observing error_word given correct_word
        """
        tables_empty = True
        for key in self.error_tables:
            if len(self.error_tables[key]) > 0:
                tables_empty = False
                break

        # Fallback if no error tables were loaded
        if tables_empty:
            if error_word == correct_word:
                if error_word in self.vocab:
                    return self.alpha
                return 1e-12

            dist = self._bounded_edit_distance(error_word, correct_word, max(2, self.edits))
            if dist == 1:
                return 1e-4
            if dist == 2:
                return 1e-8
            return 1e-15

        # If the word is known, exact match means no error
        if error_word == correct_word:
            if error_word in self.vocab:
                return self.alpha
            return 1e-12

        # Same length can be substitution or transposition
        if len(error_word) == len(correct_word):
            diff_positions = []

            for i in range(len(correct_word)):
                if correct_word[i] != error_word[i]:
                    diff_positions.append(i)

            if len(diff_positions) == 1:
                i = diff_positions[0]
                key = correct_word[i] + error_word[i]
                numerator = self.error_tables["substitution"][key] + 1
                denominator = self.char_unigrams[correct_word[i]] + len(self.alphabet)
                return numerator / max(denominator, 1)

            if len(diff_positions) == 2:
                i = diff_positions[0]
                j = diff_positions[1]

                if j == i + 1 and correct_word[i] == error_word[j] and correct_word[j] == error_word[i]:
                    key = correct_word[i] + correct_word[j]
                    numerator = self.error_tables["transposition"][key] + 1
                    denominator = self.char_bigrams[key] + len(self.alphabet)
                    return numerator / max(denominator, 1)

        # Deletion: the correct word is longer by one character
        if len(correct_word) == len(error_word) + 1:
            i = 0
            while i < len(error_word) and correct_word[i] == error_word[i]:
                i += 1

            prev_char = correct_word[i - 1] if i > 0 else "#"
            deleted_char = correct_word[i]
            key = prev_char + deleted_char
            numerator = self.error_tables["deletion"][key] + 1
            denominator = self.char_bigrams[key] + len(self.alphabet)
            return numerator / max(denominator, 1)

        # Insertion: the observed word is longer by one character
        if len(error_word) == len(correct_word) + 1:
            i = 0
            while i < len(correct_word) and correct_word[i] == error_word[i]:
                i += 1

            prev_char = correct_word[i - 1] if i > 0 else "#"
            inserted_char = error_word[i]
            key = prev_char + inserted_char
            numerator = self.error_tables["insertion"][key] + 1
            denominator = self.char_bigrams[key] + len(self.alphabet)
            return numerator / max(denominator, 1)

        # Multi-edit fallback
        dist = self._bounded_edit_distance(error_word, correct_word, max(2, self.edits))
        if dist == 1:
            return 1e-6
        if dist == 2:
            return 1e-12
        return 1e-15

    def _distance_bonus(self, observed_word, candidate_word):
        """
        Returns a probability-like bonus based on edit distance.
        Closer candidates are usually more likely corrections
        """
        dist = self._bounded_edit_distance(observed_word, candidate_word, max(2, self.edits))

        if dist == 0:
            return 1.0
        if dist == 1:
            return 2.0
        if dist == 2:
            return 0.001

        return 1e-12

    def _length_bonus(self, observed_word, candidate_word):
        """
        Returns a small preference based on length difference.
        The bonus is asymmetric:
        - If the candidate is longer by 1-2 letters, it may mean the observed word missed letters, which is a common spelling error.
        - If the candidate is shorter, we are more careful because it can create bad corrections like choosing very short frequent words.
        """
        diff = len(candidate_word) - len(observed_word)

        if diff == 0:
            return 1.0
        if diff == 1:
            return 0.8
        if diff == 2:
            return 0.35
        if diff == -1:
            return 0.15
        if diff == -2:
            return 0.05

        return 1e-12

    def _insertion_bonus(self, observed, candidate):
        """
        Gives a bonus if candidate can be formed by inserting letters into observed
        """
        if len(candidate) <= len(observed):
            return 1.0

        i, j = 0, 0
        mismatches = 0

        while i < len(observed) and j < len(candidate):
            if observed[i] == candidate[j]:
                i += 1
                j += 1
            else:
                j += 1
                mismatches += 1

            if mismatches > 2:
                return 1.0

        return 3.0


def normalize_text(text):
    """
    Returns a normalized version of the specified text (string).
    Decisions:
    - Lowercasing to map semantic equivalents (e.g., "The" -> "the").
    - Keep alphabetic strings to support basic spell-checking and remove stray punctuation.
    - Added padding <s> based on larger corpus hints, but omitted here to avoid breaking length limits unless explicit.
    """
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return " ".join(text.split())

