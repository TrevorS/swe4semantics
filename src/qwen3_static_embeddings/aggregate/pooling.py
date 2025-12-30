"""Pooling methods for aggregating contextual embeddings."""

import numpy as np


def mean_pooling(embeddings: np.ndarray) -> np.ndarray:
    """
    Simple mean pooling across contexts.

    Args:
        embeddings: Array of shape (n_contexts, hidden_dim)

    Returns:
        Static embedding of shape (hidden_dim,)
    """
    return embeddings.mean(axis=0)


def weighted_mean_pooling(
    embeddings: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """
    Weighted mean pooling across contexts.

    Args:
        embeddings: Array of shape (n_contexts, hidden_dim)
        weights: Array of shape (n_contexts,), if None uses uniform weights

    Returns:
        Static embedding of shape (hidden_dim,)
    """
    if weights is None:
        return mean_pooling(embeddings)

    # Normalize weights
    weights = weights / weights.sum()

    # Weighted average
    return np.average(embeddings, axis=0, weights=weights)


def variance_weighted_pooling(embeddings: np.ndarray) -> np.ndarray:
    """
    Variance-weighted pooling - downweight high-variance contexts.

    Contexts that are very different from others get lower weight,
    as they may be outliers or unusual usages.

    Args:
        embeddings: Array of shape (n_contexts, hidden_dim)

    Returns:
        Static embedding of shape (hidden_dim,)
    """
    if len(embeddings) <= 1:
        return mean_pooling(embeddings)

    # Compute centroid
    centroid = embeddings.mean(axis=0)

    # Compute distances from centroid
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    # Convert distances to weights (inverse distance)
    # Add small epsilon to avoid division by zero
    weights = 1.0 / (distances + 1e-6)

    return weighted_mean_pooling(embeddings, weights)
