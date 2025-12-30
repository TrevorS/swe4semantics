"""Tests for vocabulary module."""

import tempfile
from pathlib import Path

from qwen3_static_embeddings.data.vocabulary import Vocabulary, _is_valid_word


def test_vocabulary_basic():
    """Test basic Vocabulary functionality."""
    words = ["hello", "world", "test"]
    word2freq = {"hello": 100, "world": 50, "test": 25}

    vocab = Vocabulary(words=words, word2freq=word2freq)

    assert len(vocab) == 3
    assert "hello" in vocab
    assert "unknown" not in vocab
    assert vocab[0] == "hello"


def test_vocabulary_save_load():
    """Test vocabulary serialization."""
    words = ["alpha", "beta", "gamma"]
    word2freq = {"alpha": 10, "beta": 20, "gamma": 30}

    vocab = Vocabulary(words=words, word2freq=word2freq)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vocab.pkl"

        vocab.save(path)
        loaded = Vocabulary.load(path)

        assert len(loaded) == len(vocab)
        assert loaded.words == vocab.words
        assert loaded.word2freq == vocab.word2freq


def test_vocabulary_filter():
    """Test vocabulary filtering."""
    words = ["a", "ab", "abc", "abcd", "abcde"]
    word2freq = {"a": 100, "ab": 50, "abc": 25, "abcd": 10, "abcde": 5}

    vocab = Vocabulary(words=words, word2freq=word2freq)

    # Filter by min_len
    filtered = vocab.filter(min_freq=1, min_len=3)
    assert "a" not in filtered
    assert "ab" not in filtered
    assert "abc" in filtered

    # Filter by min_freq
    filtered = vocab.filter(min_freq=20, min_len=1)
    assert "abcde" not in filtered
    assert "a" in filtered

    # Filter by max_size
    filtered = vocab.filter(min_freq=1, min_len=1, max_size=2)
    assert len(filtered) == 2


def test_is_valid_word():
    """Test word validation."""
    assert _is_valid_word("hello")
    assert _is_valid_word("Hello123")
    assert not _is_valid_word("...")
    assert not _is_valid_word("123")
    assert not _is_valid_word("#hashtag")
    assert not _is_valid_word("http://url")
