"""Tests for evaluation module."""

import numpy as np
import pytest

from qwen3_static_embeddings.encode.encoder import StaticEncoder
from qwen3_static_embeddings.evaluate import (
    create_synthetic_sentence_pairs,
    create_synthetic_word_pairs,
    evaluate_sts,
    evaluate_word_similarity,
)


@pytest.fixture
def simple_embeddings():
    """Create simple test embeddings with semantic structure."""
    # Create embeddings where similar words have similar vectors
    return {
        # Similar cluster 1: vehicles
        "car": np.array([0.9, 0.1, 0.0]),
        "automobile": np.array([0.85, 0.15, 0.0]),
        "vehicle": np.array([0.8, 0.2, 0.0]),
        # Similar cluster 2: animals
        "dog": np.array([0.1, 0.9, 0.0]),
        "animal": np.array([0.15, 0.85, 0.0]),
        "cat": np.array([0.1, 0.8, 0.1]),
        # Different: food
        "banana": np.array([0.0, 0.1, 0.9]),
        # Common words
        "the": np.array([0.3, 0.3, 0.4]),
        "is": np.array([0.35, 0.35, 0.3]),
        "a": np.array([0.33, 0.33, 0.34]),
    }


def test_evaluate_word_similarity(simple_embeddings):
    """Test word similarity evaluation."""
    word_pairs = [
        ("car", "automobile", 0.9),  # High similarity
        ("car", "vehicle", 0.7),  # Medium-high
        ("dog", "animal", 0.6),  # Medium
        ("car", "banana", 0.1),  # Low
    ]

    result = evaluate_word_similarity(simple_embeddings, word_pairs, "test")

    assert result.dataset == "test"
    assert result.n_pairs == 4
    assert result.n_found == 4
    assert result.coverage == 1.0
    # With our structured embeddings, correlation should be positive
    assert result.spearman_rho > 0


def test_evaluate_word_similarity_missing_words(simple_embeddings):
    """Test word similarity with OOV words."""
    word_pairs = [
        ("car", "automobile", 0.9),
        ("xyz", "unknown", 0.5),  # OOV pair
    ]

    result = evaluate_word_similarity(simple_embeddings, word_pairs, "test")

    assert result.n_pairs == 2
    assert result.n_found == 1
    assert result.coverage == 0.5


def test_evaluate_sts(simple_embeddings):
    """Test sentence similarity evaluation."""

    # Simple word tokenizer
    def word_tokenizer(text):
        return [(w, (0, len(w))) for w in text.lower().split()]

    encoder = StaticEncoder(
        word2vec=simple_embeddings,
        dim=3,
        word_tokenizer=word_tokenizer,
        normalize=True,
    )

    sentence_pairs = [
        ("the car is fast", "the automobile is quick", 0.9),
        ("the dog is cute", "the cat is cute", 0.7),
        ("car vehicle", "banana fruit", 0.1),
    ]

    result = evaluate_sts(encoder, sentence_pairs, "test", show_progress=False)

    assert result.dataset == "test"
    assert result.n_pairs == 3


def test_synthetic_word_pairs():
    """Test synthetic word pair creation."""
    pairs = create_synthetic_word_pairs()

    assert len(pairs) > 0
    assert all(len(p) == 3 for p in pairs)
    assert all(isinstance(p[0], str) for p in pairs)
    assert all(isinstance(p[2], float) for p in pairs)


def test_synthetic_sentence_pairs():
    """Test synthetic sentence pair creation."""
    pairs = create_synthetic_sentence_pairs()

    assert len(pairs) > 0
    assert all(len(p) == 3 for p in pairs)
    assert all(isinstance(p[0], str) for p in pairs)
    assert all(isinstance(p[2], float) for p in pairs)
