"""PCA-based transformations for embeddings.

This module implements the sentence-level PCA post-processing from SWE4Semantics,
based on the "All-but-the-Top" technique (Mu & Viswanath 2018).
"""

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def fit_sentence_pca(
    sentence_embeddings: np.ndarray,
    n_components: int = 7,
) -> PCA:
    """
    Fit PCA on sentence embeddings to find dominant directions.

    These directions often encode non-semantic information like
    sentence length, frequency bias, or positional patterns.

    Args:
        sentence_embeddings: Array of shape (n_sentences, dim)
        n_components: Number of principal components to compute

    Returns:
        Fitted PCA object
    """
    pca = PCA(n_components=n_components)
    pca.fit(sentence_embeddings)
    return pca


def remove_principal_components(
    word_embeddings: np.ndarray,
    pca: PCA,
    n_remove: int | None = None,
) -> np.ndarray:
    """
    Remove top principal components from word embeddings.

    This projects out the dominant directions found in sentence embeddings,
    which often improves downstream task performance.

    Args:
        word_embeddings: Array of shape (n_words, dim)
        pca: Fitted PCA from fit_sentence_pca
        n_remove: Number of components to remove (default: all in PCA)

    Returns:
        Transformed embeddings with same shape
    """
    if n_remove is None:
        n_remove = pca.n_components_

    # Get components to remove
    components = pca.components_[:n_remove]  # (n_remove, dim)

    # Project embeddings onto components and subtract
    result = word_embeddings.copy()

    for i in range(n_remove):
        component = components[i]  # (dim,)
        # Compute projection: (emb · component) * component
        projections = (result @ component)[:, np.newaxis] * component
        result = result - projections

    return result


def apply_pca_reduction(
    embeddings: np.ndarray,
    output_dim: int = 256,
) -> tuple[np.ndarray, PCA]:
    """
    Reduce embedding dimensionality using PCA.

    Args:
        embeddings: Array of shape (n_samples, input_dim)
        output_dim: Target dimensionality

    Returns:
        Tuple of (reduced embeddings, fitted PCA)
    """
    pca = PCA(n_components=output_dim)
    reduced = pca.fit_transform(embeddings)
    return reduced, pca


def transform_embeddings(
    word_embeddings: dict[str, np.ndarray],
    sentence_embeddings: np.ndarray,
    n_components_remove: int = 7,
    output_dim: int = 256,
) -> dict[str, np.ndarray]:
    """
    Full transformation pipeline: remove PCs, reduce dimensions, normalize.

    This implements the post-processing from SWE4Semantics:
    1. Fit PCA on sentence embeddings
    2. Remove top-k principal components from word embeddings
    3. Reduce dimensionality with PCA
    4. L2 normalize

    Args:
        word_embeddings: Dict mapping words to raw embeddings
        sentence_embeddings: Array of sentence embeddings for fitting
        n_components_remove: Number of PCs to remove
        output_dim: Final embedding dimension

    Returns:
        Dict mapping words to transformed embeddings
    """
    from qwen3_static_embeddings.transform.normalize import l2_normalize

    # Convert to array for batch processing
    words = list(word_embeddings.keys())
    embeddings = np.array([word_embeddings[w] for w in words])

    # Step 1: Fit sentence PCA
    print(f"Fitting PCA on {len(sentence_embeddings)} sentence embeddings...")
    sent_pca = fit_sentence_pca(sentence_embeddings, n_components=n_components_remove)

    # Step 2: Remove principal components
    print(f"Removing top {n_components_remove} principal components...")
    embeddings = remove_principal_components(embeddings, sent_pca, n_remove=n_components_remove)

    # Step 3: Dimensionality reduction
    print(f"Reducing to {output_dim} dimensions...")
    embeddings, _ = apply_pca_reduction(embeddings, output_dim=output_dim)

    # Step 4: L2 normalize
    print("L2 normalizing...")
    embeddings = l2_normalize(embeddings)

    # Convert back to dict
    return {w: embeddings[i] for i, w in enumerate(words)}


def save_embeddings(
    embeddings: dict[str, np.ndarray],
    path: Path,
) -> None:
    """
    Save embeddings in word2vec text format.

    Args:
        embeddings: Dict mapping words to embeddings
        path: Output file path
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        for word, emb in embeddings.items():
            vec_str = " ".join(str(x) for x in emb)
            f.write(f"{word} {vec_str}\n")

    print(f"Saved {len(embeddings)} embeddings to {path}")


def load_embeddings(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """
    Load embeddings from word2vec text format.

    Args:
        path: Path to embeddings file

    Returns:
        Tuple of (embeddings dict, dimension)

    Raises:
        ValueError: If the file is empty or contains no valid embeddings
    """
    embeddings: dict[str, np.ndarray] = {}
    dim: int = 0

    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split(" ")

            # Skip header line if present
            if len(parts) == 2:
                continue

            word = parts[0]
            vec = [float(x) for x in parts[1:]]

            if dim == 0:
                dim = len(vec)

            embeddings[word] = np.array(vec)

    if not embeddings:
        raise ValueError(f"No embeddings found in {path}")

    return embeddings, dim
