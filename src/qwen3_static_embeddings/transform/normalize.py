"""Normalization utilities for embeddings."""

import numpy as np


def l2_normalize(embeddings: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    L2 normalize embeddings.

    Args:
        embeddings: Array of shape (n_samples, dim) or (dim,)
        eps: Small constant to avoid division by zero

    Returns:
        L2 normalized embeddings with same shape
    """
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        return embeddings / (norm + eps)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + eps)


def mean_center(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean-center embeddings.

    Args:
        embeddings: Array of shape (n_samples, dim)

    Returns:
        Tuple of (centered embeddings, mean vector)
    """
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    return centered, mean
