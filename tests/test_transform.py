"""Tests for transform module."""

import numpy as np

from qwen3_static_embeddings.transform.normalize import l2_normalize, mean_center
from qwen3_static_embeddings.transform.pca import (
    apply_pca_reduction,
    fit_sentence_pca,
    remove_principal_components,
)


def test_l2_normalize_1d():
    """Test L2 normalization of 1D array."""
    vec = np.array([3.0, 4.0])
    normalized = l2_normalize(vec)

    assert np.isclose(np.linalg.norm(normalized), 1.0)
    assert np.allclose(normalized, [0.6, 0.8])


def test_l2_normalize_2d():
    """Test L2 normalization of 2D array."""
    vecs = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]])
    normalized = l2_normalize(vecs)

    # Each row should have unit norm
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, [1.0, 1.0, 1.0])


def test_mean_center():
    """Test mean centering."""
    vecs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    centered, mean = mean_center(vecs)

    assert np.allclose(mean, [3.0, 4.0])
    assert np.allclose(centered.mean(axis=0), [0.0, 0.0])


def test_fit_sentence_pca():
    """Test fitting PCA on sentence embeddings."""
    np.random.seed(42)

    # Create some embeddings with clear structure
    n_samples = 100
    dim = 64
    embeddings = np.random.randn(n_samples, dim)

    pca = fit_sentence_pca(embeddings, n_components=7)

    assert pca.n_components_ == 7
    assert pca.components_.shape == (7, dim)


def test_remove_principal_components():
    """Test removing principal components."""
    np.random.seed(42)

    # Create embeddings
    n_words = 50
    dim = 32
    word_embeddings = np.random.randn(n_words, dim)

    # Create sentence embeddings for PCA
    n_sentences = 100
    sentence_embeddings = np.random.randn(n_sentences, dim)

    # Fit PCA
    pca = fit_sentence_pca(sentence_embeddings, n_components=3)

    # Remove components
    transformed = remove_principal_components(word_embeddings, pca, n_remove=3)

    assert transformed.shape == word_embeddings.shape

    # Check that projections onto removed components are near zero
    for i in range(3):
        projections = transformed @ pca.components_[i]
        assert np.allclose(projections, 0, atol=1e-10)


def test_apply_pca_reduction():
    """Test dimensionality reduction."""
    np.random.seed(42)

    n_samples = 500  # Need n_samples > output_dim for PCA
    input_dim = 768
    output_dim = 256

    embeddings = np.random.randn(n_samples, input_dim)
    reduced, pca = apply_pca_reduction(embeddings, output_dim=output_dim)

    assert reduced.shape == (n_samples, output_dim)
    assert pca.n_components_ == output_dim
