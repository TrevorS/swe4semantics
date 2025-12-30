"""Word similarity evaluation on standard benchmarks."""

from dataclasses import dataclass

import numpy as np
from scipy import stats

from qwen3_static_embeddings.transform.normalize import l2_normalize


@dataclass
class WordSimResult:
    """Result of word similarity evaluation."""

    dataset: str
    spearman_rho: float
    pearson_r: float
    n_pairs: int
    n_found: int
    coverage: float


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    vec1 = l2_normalize(vec1)
    vec2 = l2_normalize(vec2)
    return float(np.dot(vec1, vec2))


def evaluate_word_similarity(
    word2vec: dict[str, np.ndarray],
    word_pairs: list[tuple[str, str, float]],
    dataset_name: str = "custom",
) -> WordSimResult:
    """
    Evaluate word embeddings on word similarity task.

    Args:
        word2vec: Dictionary mapping words to embeddings
        word_pairs: List of (word1, word2, gold_score) tuples
        dataset_name: Name of the dataset for reporting

    Returns:
        WordSimResult with correlation scores
    """
    predictions = []
    gold_scores = []
    n_found = 0

    for word1, word2, gold in word_pairs:
        # Try exact match and lowercase
        vec1 = word2vec.get(word1)
        if vec1 is None:
            vec1 = word2vec.get(word1.lower())

        vec2 = word2vec.get(word2)
        if vec2 is None:
            vec2 = word2vec.get(word2.lower())

        if vec1 is not None and vec2 is not None:
            sim = cosine_similarity(vec1, vec2)
            predictions.append(sim)
            gold_scores.append(gold)
            n_found += 1

    if len(predictions) < 2:
        return WordSimResult(
            dataset=dataset_name,
            spearman_rho=0.0,
            pearson_r=0.0,
            n_pairs=len(word_pairs),
            n_found=n_found,
            coverage=n_found / len(word_pairs) if word_pairs else 0.0,
        )

    spearman_rho, _ = stats.spearmanr(predictions, gold_scores)
    pearson_r, _ = stats.pearsonr(predictions, gold_scores)

    return WordSimResult(
        dataset=dataset_name,
        spearman_rho=spearman_rho,
        pearson_r=pearson_r,
        n_pairs=len(word_pairs),
        n_found=n_found,
        coverage=n_found / len(word_pairs),
    )


def load_simlex999() -> list[tuple[str, str, float]]:
    """
    Load SimLex-999 dataset from HuggingFace datasets.

    Returns:
        List of (word1, word2, similarity_score) tuples
    """
    try:
        from datasets import load_dataset

        # tasksource/simlex has SimLex999 column with scores 0-10
        dataset = load_dataset("tasksource/simlex", split="train")
        pairs = []
        for item in dataset:
            # Normalize score from 0-10 to 0-1
            score = item["SimLex999"] / 10.0
            pairs.append((item["word1"], item["word2"], score))
        return pairs
    except Exception as e:
        print(f"Warning: Could not load SimLex-999: {e}")
        return []


def load_wordsim353() -> list[tuple[str, str, float]]:
    """
    Load WordSim-353 dataset from HuggingFace datasets.

    Uses StephanAkkerman/semantic-similarity which includes WordSim-353.

    Returns:
        List of (word1, word2, similarity_score) tuples
    """
    try:
        from datasets import load_dataset

        # This dataset has word1, word2, similarity (0-1), dataset columns
        dataset = load_dataset("StephanAkkerman/semantic-similarity", split="train")
        pairs = []
        for item in dataset:
            # Filter to only WordSim-353 entries
            if "wordsim" in item["dataset"].lower():
                pairs.append((item["word1"], item["word2"], item["similarity"]))
        return pairs
    except Exception as e:
        print(f"Warning: Could not load WordSim-353: {e}")
        return []


def create_synthetic_word_pairs() -> list[tuple[str, str, float]]:
    """
    Create synthetic word pairs for testing when datasets unavailable.

    Returns:
        List of (word1, word2, similarity_score) tuples
    """
    # High similarity pairs
    pairs = [
        ("car", "automobile", 0.9),
        ("happy", "joyful", 0.9),
        ("big", "large", 0.85),
        ("fast", "quick", 0.85),
        ("smart", "intelligent", 0.9),
        # Medium similarity
        ("car", "vehicle", 0.7),
        ("dog", "animal", 0.6),
        ("book", "read", 0.5),
        ("water", "drink", 0.5),
        # Low similarity
        ("car", "banana", 0.1),
        ("happy", "table", 0.05),
        ("dog", "computer", 0.1),
        ("book", "mountain", 0.1),
        ("water", "philosophy", 0.05),
    ]
    return pairs
