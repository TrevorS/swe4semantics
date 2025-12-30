"""Tests for static encoder module."""

import numpy as np
import pytest

from qwen3_static_embeddings.encode.encoder import StaticEncoder


@pytest.fixture
def simple_embeddings():
    """Create simple test embeddings."""
    return {
        "hello": np.array([1.0, 0.0, 0.0]),
        "world": np.array([0.0, 1.0, 0.0]),
        "test": np.array([0.0, 0.0, 1.0]),
        "the": np.array([0.5, 0.5, 0.0]),
        "quick": np.array([0.3, 0.3, 0.4]),
    }


@pytest.fixture
def encoder(simple_embeddings):
    """Create encoder with simple embeddings."""

    # Simple word tokenizer
    def word_tokenizer(text):
        return [(w, (0, len(w))) for w in text.lower().split()]

    return StaticEncoder(
        word2vec=simple_embeddings,
        dim=3,
        word_tokenizer=word_tokenizer,
        normalize=True,
    )


def test_encoder_single_word(encoder):
    """Test encoding a single word."""
    emb = encoder.encode("hello")

    assert emb.shape == (3,)
    assert np.isclose(np.linalg.norm(emb), 1.0)  # Normalized


def test_encoder_sentence(encoder):
    """Test encoding a sentence."""
    emb = encoder.encode("hello world")

    assert emb.shape == (3,)
    assert np.isclose(np.linalg.norm(emb), 1.0)


def test_encoder_unknown_words(encoder):
    """Test handling unknown words."""
    # "xyz" is not in vocabulary
    emb = encoder.encode("hello xyz world")

    # Should still produce valid embedding from known words
    assert emb.shape == (3,)
    assert not np.allclose(emb, 0)  # Not all zeros


def test_encoder_all_unknown(encoder):
    """Test handling all unknown words."""
    emb = encoder.encode("xyz abc def")

    # Should return zero vector when no words found
    assert emb.shape == (3,)
    # Normalization of zero vector should still give something
    assert np.allclose(emb, 0)


def test_encoder_batch(encoder):
    """Test batch encoding."""
    texts = ["hello world", "test the quick", "world"]
    embeddings = encoder.encode_batch(texts)

    assert embeddings.shape == (3, 3)


def test_encoder_similarity(encoder):
    """Test similarity computation."""
    sim = encoder.similarity("hello", "hello")
    assert np.isclose(sim, 1.0)  # Same text should have similarity 1

    sim = encoder.similarity("hello", "world")
    # Different words should have lower similarity
    assert sim < 1.0


def test_encoder_contains(encoder):
    """Test vocabulary membership."""
    assert "hello" in encoder
    assert "xyz" not in encoder


def test_encoder_len(encoder):
    """Test vocabulary size."""
    assert len(encoder) == 5
