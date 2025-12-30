"""Vocabulary building from corpus."""

import pickle
import string
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from transformers import AutoTokenizer

from qwen3_static_embeddings.data.corpus import iter_corpus_sentences

# Punctuation set including CJK punctuation
PUNCTUATION = set(
    list(string.punctuation) + ["。", "、", "？", "！", "「", "」", "（", "）", "：", "・", "，"]
)


@dataclass
class Vocabulary:
    """
    A vocabulary of words with their frequencies.

    Attributes:
        words: List of words in frequency order (most frequent first)
        word2freq: Mapping from word to frequency count
        word2idx: Mapping from word to index
    """

    words: list[str]
    word2freq: dict[str, int]
    word2idx: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.word2idx = {w: i for i, w in enumerate(self.words)}

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.word2idx

    def __getitem__(self, idx: int) -> str:
        return self.words[idx]

    def save(self, path: Path) -> None:
        """Save vocabulary to pickle file."""
        with open(path, "wb") as f:
            pickle.dump({"words": self.words, "word2freq": self.word2freq}, f)

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        """Load vocabulary from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(words=data["words"], word2freq=data["word2freq"])

    def filter(
        self,
        min_freq: int = 10,
        min_len: int = 3,
        max_size: int | None = None,
    ) -> "Vocabulary":
        """
        Return a filtered vocabulary.

        Args:
            min_freq: Minimum word frequency
            min_len: Minimum word length
            max_size: Maximum vocabulary size

        Returns:
            New filtered Vocabulary instance
        """
        filtered_words = []
        filtered_freqs = {}

        for word in self.words:
            freq = self.word2freq[word]

            # Apply filters
            if freq < min_freq:
                continue
            if len(word) < min_len:
                continue
            if not _is_valid_word(word):
                continue

            filtered_words.append(word)
            filtered_freqs[word] = freq

            if max_size and len(filtered_words) >= max_size:
                break

        return Vocabulary(words=filtered_words, word2freq=filtered_freqs)


def _is_valid_word(word: str) -> bool:
    """Check if word is valid (alphabetic, not punctuation, etc.)."""
    # Must contain at least one letter
    if not any(c.isalpha() for c in word):
        return False
    # Should not be pure punctuation
    if word in PUNCTUATION:
        return False
    # Should not start with special characters
    return not word.startswith(("#", "@", "http"))


def get_word_tokenizer(model_name: str = "bert-base-uncased"):
    """
    Get a word-level tokenizer (BERT's pre-tokenizer).

    This tokenizes text into words (not subwords), which is what we want
    for building word-level static embeddings.

    Args:
        model_name: HuggingFace model name for the tokenizer

    Returns:
        Callable that takes text and returns list of (word, span) tuples
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str


def build_vocabulary(
    corpus_path: Path,
    tokenizer_name: str = "bert-base-uncased",
    lowercase: bool = True,
    show_progress: bool = True,
) -> Vocabulary:
    """
    Build vocabulary from a corpus file.

    Args:
        corpus_path: Path to corpus file
        tokenizer_name: HuggingFace model name for word tokenizer
        lowercase: Whether to lowercase words
        show_progress: Whether to show progress bar

    Returns:
        Vocabulary instance with word frequencies
    """
    word_tokenize = get_word_tokenizer(tokenizer_name)
    word_counts: Counter = Counter()

    sentences = iter_corpus_sentences(corpus_path, show_progress=show_progress)

    for sentence in sentences:
        if lowercase:
            sentence = sentence.lower()

        # Tokenize into words
        word_spans = word_tokenize(sentence)
        words = [word for word, _ in word_spans]

        # Filter punctuation
        words = [w for w in words if w not in PUNCTUATION]

        word_counts.update(words)

    # Sort by frequency (most common first)
    sorted_words = [word for word, _ in word_counts.most_common()]
    word2freq = dict(word_counts)

    return Vocabulary(words=sorted_words, word2freq=word2freq)
