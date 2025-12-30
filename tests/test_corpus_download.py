"""Tests for corpus download module."""

import pytest

from qwen3_static_embeddings.data.corpus_download import (
    CORPUS_CONFIGS,
    estimate_corpus_size,
)


def test_corpus_configs_exist():
    """Test that corpus configs are defined."""
    assert "cc100" in CORPUS_CONFIGS
    assert CORPUS_CONFIGS["cc100"].name == "cc100"
    assert CORPUS_CONFIGS["cc100"].language == "en"


def test_estimate_corpus_size_english():
    """Test corpus size estimation for English."""
    estimate = estimate_corpus_size("cc100", "en")

    assert "sentences" in estimate
    assert "size_gb" in estimate
    # English CC-100 should be large
    assert "300M" in estimate["sentences"] or "M" in estimate["sentences"]


def test_estimate_corpus_size_unknown():
    """Test corpus size estimation for unknown language."""
    estimate = estimate_corpus_size("cc100", "xyz")

    assert estimate.get("sentences") == "unknown"
    assert estimate.get("size_gb") == "unknown"


# Note: Actual download tests are skipped by default as they require
# network access and can be slow. Run with --run-slow to include them.
@pytest.mark.skip(reason="Requires network access and is slow")
def test_download_cc100_sample():
    """Test downloading a small sample of CC-100."""
    from qwen3_static_embeddings.data import get_corpus_sample

    sentences = get_corpus_sample(
        corpus="cc100",
        language="en",
        n_sentences=10,
    )

    assert len(sentences) == 10
    assert all(isinstance(s, str) for s in sentences)
    assert all(len(s) > 0 for s in sentences)
