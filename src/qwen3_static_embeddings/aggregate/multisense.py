"""Multi-sense embedding extraction for polysemous words."""

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


@dataclass
class MultiSenseEmbedding:
    """Multiple sense embeddings for a polysemous word."""

    word: str
    sense_embeddings: list[np.ndarray]  # One embedding per sense
    sense_counts: list[int]  # Number of contexts per sense
    cluster_labels: np.ndarray  # Which sense each context belongs to


def extract_multi_sense(
    word: str,
    context_embeddings: np.ndarray,
    n_senses: int = 3,
    min_contexts_per_sense: int = 5,
) -> MultiSenseEmbedding:
    """
    Extract multiple sense embeddings via K-means clustering.

    Based on "Word Sense Induction with Knowledge Distillation from BERT" (2023).

    Args:
        word: The target word
        context_embeddings: Array of shape (n_contexts, hidden_dim)
        n_senses: Number of senses to extract
        min_contexts_per_sense: Minimum contexts required per sense

    Returns:
        MultiSenseEmbedding with sense vectors
    """
    n_contexts = len(context_embeddings)

    # Adjust n_senses if not enough contexts
    max_senses = n_contexts // min_contexts_per_sense
    n_senses = min(n_senses, max_senses)
    n_senses = max(n_senses, 1)  # At least 1 sense

    if n_senses == 1:
        # Single sense - just average
        return MultiSenseEmbedding(
            word=word,
            sense_embeddings=[context_embeddings.mean(axis=0)],
            sense_counts=[n_contexts],
            cluster_labels=np.zeros(n_contexts, dtype=int),
        )

    # K-means clustering
    kmeans = KMeans(n_clusters=n_senses, random_state=42, n_init=10)
    labels = kmeans.fit_predict(context_embeddings)

    # Compute sense embeddings (centroid of each cluster)
    sense_embeddings = []
    sense_counts = []

    for sense_id in range(n_senses):
        mask = labels == sense_id
        sense_contexts = context_embeddings[mask]

        if len(sense_contexts) > 0:
            sense_embeddings.append(sense_contexts.mean(axis=0))
            sense_counts.append(len(sense_contexts))
        else:
            # Empty cluster - shouldn't happen but handle gracefully
            sense_embeddings.append(np.zeros(context_embeddings.shape[1]))
            sense_counts.append(0)

    return MultiSenseEmbedding(
        word=word,
        sense_embeddings=sense_embeddings,
        sense_counts=sense_counts,
        cluster_labels=labels,
    )


def find_optimal_n_senses(
    context_embeddings: np.ndarray,
    max_senses: int = 5,
    min_contexts_per_sense: int = 5,
) -> int:
    """
    Find optimal number of senses using silhouette score.

    Args:
        context_embeddings: Array of shape (n_contexts, hidden_dim)
        max_senses: Maximum number of senses to try
        min_contexts_per_sense: Minimum contexts per sense

    Returns:
        Optimal number of senses
    """
    n_contexts = len(context_embeddings)

    # Limit max_senses based on available contexts
    max_possible = n_contexts // min_contexts_per_sense
    max_senses = min(max_senses, max_possible)

    if max_senses < 2:
        return 1

    best_score = -1
    best_k = 1

    for k in range(2, max_senses + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(context_embeddings)

        # Check if all clusters have minimum contexts
        unique, counts = np.unique(labels, return_counts=True)
        if min(counts) < min_contexts_per_sense:
            continue

        score = silhouette_score(context_embeddings, labels)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k
