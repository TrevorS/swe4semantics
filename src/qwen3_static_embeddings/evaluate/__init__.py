"""Evaluation utilities for word and sentence embeddings."""

from qwen3_static_embeddings.evaluate.sentence_similarity import (
    STSResult,
    create_synthetic_sentence_pairs,
    evaluate_sts,
    load_stsb,
)
from qwen3_static_embeddings.evaluate.word_similarity import (
    WordSimResult,
    create_synthetic_word_pairs,
    evaluate_word_similarity,
    load_simlex999,
    load_wordsim353,
)

__all__ = [
    "WordSimResult",
    "evaluate_word_similarity",
    "load_simlex999",
    "load_wordsim353",
    "create_synthetic_word_pairs",
    "STSResult",
    "evaluate_sts",
    "load_stsb",
    "create_synthetic_sentence_pairs",
]
