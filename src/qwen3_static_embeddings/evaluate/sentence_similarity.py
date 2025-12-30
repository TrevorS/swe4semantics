"""Sentence similarity evaluation on STS benchmarks."""

from dataclasses import dataclass

from scipy import stats
from tqdm import tqdm

from qwen3_static_embeddings.encode.encoder import StaticEncoder


@dataclass
class STSResult:
    """Result of sentence similarity evaluation."""

    dataset: str
    spearman_rho: float
    pearson_r: float
    n_pairs: int


def evaluate_sts(
    encoder: StaticEncoder,
    sentence_pairs: list[tuple[str, str, float]],
    dataset_name: str = "custom",
    batch_size: int = 256,
    show_progress: bool = True,
) -> STSResult:
    """
    Evaluate sentence encoder on STS task.

    Args:
        encoder: StaticEncoder instance
        sentence_pairs: List of (sent1, sent2, gold_score) tuples
        dataset_name: Name of the dataset for reporting
        batch_size: Batch size for encoding
        show_progress: Whether to show progress bar

    Returns:
        STSResult with correlation scores
    """
    predictions = []
    gold_scores = []

    iterator = range(0, len(sentence_pairs), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc=f"Evaluating {dataset_name}")

    for i in iterator:
        batch = sentence_pairs[i : i + batch_size]

        for sent1, sent2, gold in batch:
            sim = encoder.similarity(sent1, sent2)
            predictions.append(sim)
            gold_scores.append(gold)

    if len(predictions) < 2:
        return STSResult(
            dataset=dataset_name,
            spearman_rho=0.0,
            pearson_r=0.0,
            n_pairs=len(sentence_pairs),
        )

    spearman_rho, _ = stats.spearmanr(predictions, gold_scores)
    pearson_r, _ = stats.pearsonr(predictions, gold_scores)

    return STSResult(
        dataset=dataset_name,
        spearman_rho=spearman_rho,
        pearson_r=pearson_r,
        n_pairs=len(sentence_pairs),
    )


def load_stsb(split: str = "test") -> list[tuple[str, str, float]]:
    """
    Load STS-Benchmark dataset from HuggingFace.

    Args:
        split: Dataset split ("train", "validation", or "test")

    Returns:
        List of (sentence1, sentence2, similarity_score) tuples
    """
    try:
        from datasets import load_dataset

        dataset = load_dataset("sentence-transformers/stsb", split=split)
        pairs = []
        for item in dataset:
            # Normalize score to 0-1 range (original is 0-5)
            score = item["score"] / 5.0
            pairs.append((item["sentence1"], item["sentence2"], score))
        return pairs
    except Exception as e:
        print(f"Warning: Could not load STS-B: {e}")
        return []


def create_synthetic_sentence_pairs() -> list[tuple[str, str, float]]:
    """
    Create synthetic sentence pairs for testing.

    Returns:
        List of (sentence1, sentence2, similarity_score) tuples
    """
    pairs = [
        # High similarity
        ("The cat sat on the mat.", "A cat was sitting on a mat.", 0.9),
        ("I love programming.", "I enjoy coding.", 0.85),
        ("The weather is nice today.", "It's a beautiful day.", 0.8),
        # Medium similarity
        ("The dog is running.", "The cat is sleeping.", 0.3),
        ("I went to the store.", "She visited the market.", 0.5),
        ("He plays guitar.", "She sings songs.", 0.4),
        # Low similarity
        ("The sun is shining.", "Mathematics is difficult.", 0.1),
        ("I like pizza.", "The stock market crashed.", 0.05),
        ("Trees are green.", "Computers process data.", 0.1),
    ]
    return pairs
