"""Build word to sentence mappings."""

import pickle
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from qwen3_static_embeddings.data.corpus import iter_corpus_sentences
from qwen3_static_embeddings.data.vocabulary import (
    PUNCTUATION,
    Vocabulary,
    get_word_tokenizer,
)


@dataclass
class Word2Sent:
    """
    Mapping from words to example sentences containing them.

    This is the core data structure for extracting contextual embeddings.
    For each word in the vocabulary, we store N example sentences that
    contain that word.
    """

    mapping: dict[str, list[str]]

    def __len__(self) -> int:
        return len(self.mapping)

    def __getitem__(self, word: str) -> list[str]:
        return self.mapping.get(word, [])

    def __contains__(self, word: str) -> bool:
        return word in self.mapping

    @property
    def words(self) -> list[str]:
        """Return list of words with sentences."""
        return list(self.mapping.keys())

    def get_sentences(self, word: str, n: int | None = None) -> list[str]:
        """
        Get example sentences for a word.

        Args:
            word: Target word
            n: Maximum number of sentences (None for all)

        Returns:
            List of sentences containing the word
        """
        sentences = self.mapping.get(word, [])
        if n is not None:
            sentences = sentences[:n]
        return sentences

    def save(self, path: Path) -> None:
        """Save word2sent mapping to pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.mapping, f)

    @classmethod
    def load(cls, path: Path) -> "Word2Sent":
        """Load word2sent mapping from pickle file."""
        with open(path, "rb") as f:
            mapping = pickle.load(f)
        return cls(mapping=mapping)

    def filter_by_vocab(self, vocab: Vocabulary) -> "Word2Sent":
        """Return a new Word2Sent filtered to vocabulary words only."""
        filtered = {w: sents for w, sents in self.mapping.items() if w in vocab}
        return Word2Sent(mapping=filtered)


def build_word2sent(
    corpus_path: Path,
    vocab: Vocabulary,
    n_sentences: int = 100,
    tokenizer_name: str = "bert-base-uncased",
    lowercase: bool = True,
    max_sentence_len: int = 500,
    show_progress: bool = True,
    seed: int = 42,
) -> Word2Sent:
    """
    Build word to sentence mapping from corpus.

    For each word in the vocabulary, collects up to n_sentences example
    sentences that contain that word.

    Args:
        corpus_path: Path to corpus file
        vocab: Vocabulary to collect sentences for
        n_sentences: Maximum sentences per word
        tokenizer_name: HuggingFace model for word tokenization
        lowercase: Whether to lowercase text
        max_sentence_len: Skip sentences longer than this
        show_progress: Whether to show progress bar
        seed: Random seed for reproducibility

    Returns:
        Word2Sent mapping
    """
    random.seed(seed)
    word_tokenize = get_word_tokenizer(tokenizer_name)

    # Initialize mapping with empty lists
    word2sents: dict[str, list[str]] = defaultdict(list)

    # Track which words still need more sentences
    words_needed = set(vocab.words)

    sentences = iter_corpus_sentences(corpus_path, show_progress=show_progress)

    for sentence in sentences:
        # Skip very long sentences
        if len(sentence) > max_sentence_len:
            continue

        original_sentence = sentence
        if lowercase:
            sentence = sentence.lower()

        # Tokenize into words
        word_spans = word_tokenize(sentence)
        words_in_sent = {word for word, _ in word_spans if word not in PUNCTUATION}

        # Check which needed words appear in this sentence
        matched_words = words_needed & words_in_sent

        for word in matched_words:
            # Store original sentence (before lowercasing for matching)
            word2sents[word].append(original_sentence if not lowercase else sentence)

            # Check if word has enough sentences
            if len(word2sents[word]) >= n_sentences:
                words_needed.discard(word)

        # Stop if all words have enough sentences
        if not words_needed:
            break

    # Log statistics
    total_words = len(vocab)
    words_with_sents = len(word2sents)
    avg_sents = sum(len(s) for s in word2sents.values()) / max(words_with_sents, 1)

    if show_progress:
        print("Word2Sent statistics:")
        print(f"  Words in vocab: {total_words}")
        print(f"  Words with sentences: {words_with_sents}")
        print(f"  Average sentences per word: {avg_sents:.1f}")
        print(f"  Words still needing sentences: {len(words_needed)}")

    return Word2Sent(mapping=dict(word2sents))
