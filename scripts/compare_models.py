#!/usr/bin/env python3
"""Compare static embeddings against transformer and baseline models.

This script evaluates models on the same benchmarks to measure quality.

Usage:
    # Compare test embeddings vs 0.6B model
    uv run python scripts/compare_models.py \
        --static outputs/test/exports/qwen3_static.w2v.txt \
        --model Qwen/Qwen3-Embedding-0.6B

    # Compare against Model2Vec baseline
    uv run python scripts/compare_models.py \
        --static outputs/test/exports/qwen3_static.w2v.txt \
        --baseline minishlab/potion-base-8M

    # Compare all: static vs transformer vs baseline
    uv run python scripts/compare_models.py \
        --static outputs/test/exports/qwen3_static.w2v.txt \
        --model Qwen/Qwen3-Embedding-0.6B \
        --baseline minishlab/potion-base-8M
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from qwen3_static_embeddings.encode.encoder import StaticEncoder
from qwen3_static_embeddings.evaluate import (
    load_simlex999,
    load_stsb,
    load_wordsim353,
)
from qwen3_static_embeddings.transform.pca import load_embeddings


@dataclass
class ComparisonResult:
    """Results comparing two models."""

    dataset: str
    static_spearman: float
    transformer_spearman: float
    delta: float
    retention: float  # % of transformer performance retained


class TransformerEncoder:
    """Wrapper for transformer model to match StaticEncoder interface."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            outputs = self.model(**inputs)

            # Mean pooling over sequence
            attention_mask = inputs["attention_mask"]
            hidden_states = outputs.last_hidden_state
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze().cpu().numpy()

            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            return embedding

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity."""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1, emb2))

    def get_word_embedding(self, word: str) -> np.ndarray:
        """Get embedding for a single word."""
        return self.encode(word)


class Model2VecEncoder:
    """Wrapper for Model2Vec static embedding models."""

    def __init__(self, model_name: str):
        try:
            from model2vec import StaticModel
        except ImportError as e:
            raise ImportError(
                "model2vec not installed. Install with: uv pip install model2vec"
            ) from e

        self.model = StaticModel.from_pretrained(model_name)
        self.model_name = model_name

    def encode(self, text: str) -> np.ndarray:
        """Encode a single text."""
        embedding = self.model.encode(text)
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity."""
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1, emb2))


def evaluate_word_similarity(
    encoder,
    word_pairs: list[tuple[str, str, float]],
    is_transformer: bool = False,
) -> tuple[float, float, int]:
    """
    Evaluate on word similarity.

    Returns:
        Tuple of (spearman, coverage_ratio, n_found)
    """
    predictions = []
    gold_scores = []

    for word1, word2, gold in word_pairs:
        try:
            if is_transformer:
                sim = encoder.similarity(word1, word2)
            else:
                # For static, check if words exist
                if word1 not in encoder and word1.lower() not in encoder:
                    continue
                if word2 not in encoder and word2.lower() not in encoder:
                    continue
                sim = encoder.similarity(word1, word2)

            predictions.append(sim)
            gold_scores.append(gold)
        except Exception:
            continue

    if len(predictions) < 2:
        return 0.0, 0.0, 0

    spearman, _ = stats.spearmanr(predictions, gold_scores)
    coverage = len(predictions) / len(word_pairs)
    return spearman, coverage, len(predictions)


def evaluate_sentence_similarity(
    encoder,
    sentence_pairs: list[tuple[str, str, float]],
    show_progress: bool = True,
) -> float:
    """Evaluate on sentence similarity."""
    predictions = []
    gold_scores = []

    iterator = sentence_pairs
    if show_progress:
        iterator = tqdm(sentence_pairs, desc="Evaluating")

    for sent1, sent2, gold in iterator:
        try:
            sim = encoder.similarity(sent1, sent2)
            predictions.append(sim)
            gold_scores.append(gold)
        except Exception:
            continue

    if len(predictions) < 2:
        return 0.0

    spearman, _ = stats.spearmanr(predictions, gold_scores)
    return spearman


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare static embeddings vs transformer and baseline models",
    )
    parser.add_argument(
        "--static",
        type=Path,
        required=True,
        help="Path to static embeddings file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Transformer model to compare against (e.g., Qwen/Qwen3-Embedding-0.6B)",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Model2Vec baseline model (e.g., minishlab/potion-base-8M)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for transformer model",
    )
    parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Skip transformer evaluation (just show static results)",
    )
    parser.add_argument(
        "--word-only",
        action="store_true",
        help="Only run word similarity benchmarks",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Model Comparison: Static Embeddings vs Baselines")
    print("=" * 70)

    # Load static encoder
    print(f"\nLoading static embeddings from {args.static}...")
    word2vec, dim = load_embeddings(args.static)
    static_encoder = StaticEncoder(word2vec=word2vec, dim=dim, normalize=True)
    print(f"  Vocab size: {len(static_encoder):,}")
    print(f"  Dimension: {dim}")

    # Load transformer encoder
    transformer_encoder = None
    if args.model and not args.skip_transformer:
        print(f"\nLoading transformer model: {args.model}...")
        print(f"  Device: {args.device}")
        transformer_encoder = TransformerEncoder(args.model, args.device)
        print("  Model loaded!")

    # Load baseline encoder (Model2Vec)
    baseline_encoder = None
    if args.baseline:
        print(f"\nLoading baseline model: {args.baseline}...")
        baseline_encoder = Model2VecEncoder(args.baseline)
        print("  Baseline loaded!")

    # Word similarity benchmarks
    print("\n" + "-" * 70)
    print("Word Similarity Benchmarks")
    print("-" * 70)

    word_datasets = {
        "SimLex-999": load_simlex999(),
        "WordSim-353": load_wordsim353(),
    }

    word_results = {}  # {dataset: {model: (spearman, coverage, n)}}

    for name, pairs in word_datasets.items():
        if not pairs:
            print(f"\n{name}: Dataset not available")
            continue

        print(f"\n{name}:")
        word_results[name] = {}

        # Static evaluation
        static_spearman, static_cov, static_n = evaluate_word_similarity(
            static_encoder, pairs, is_transformer=False
        )
        word_results[name]["Ours"] = (static_spearman, static_cov, static_n)
        print(f"  Ours:        ρ={static_spearman:.4f} (coverage={static_cov:.1%}, n={static_n})")

        # Baseline evaluation
        if baseline_encoder:
            base_spearman, base_cov, base_n = evaluate_word_similarity(
                baseline_encoder, pairs, is_transformer=True
            )
            word_results[name]["Baseline"] = (base_spearman, base_cov, base_n)
            print(f"  Baseline:    ρ={base_spearman:.4f} (coverage={base_cov:.1%}, n={base_n})")

        # Transformer evaluation
        if transformer_encoder:
            trans_spearman, trans_cov, trans_n = evaluate_word_similarity(
                transformer_encoder, pairs, is_transformer=True
            )
            word_results[name]["Transformer"] = (trans_spearman, trans_cov, trans_n)
            print(f"  Transformer: ρ={trans_spearman:.4f} (coverage={trans_cov:.1%}, n={trans_n})")

    sent_results = {}  # {model: spearman}

    # Sentence similarity benchmarks
    if not args.word_only:
        print("\n" + "-" * 70)
        print("Sentence Similarity Benchmarks")
        print("-" * 70)

        stsb_pairs = load_stsb("test")
        if stsb_pairs:
            print("\nSTS-Benchmark (test):")

            # Static evaluation
            print("  Evaluating ours...")
            static_spearman = evaluate_sentence_similarity(
                static_encoder, stsb_pairs, show_progress=True
            )
            sent_results["Ours"] = static_spearman
            print(f"  Ours:        ρ={static_spearman:.4f}")

            # Baseline evaluation
            if baseline_encoder:
                print("  Evaluating baseline...")
                base_spearman = evaluate_sentence_similarity(
                    baseline_encoder, stsb_pairs, show_progress=True
                )
                sent_results["Baseline"] = base_spearman
                print(f"  Baseline:    ρ={base_spearman:.4f}")

            # Transformer evaluation
            if transformer_encoder:
                print("  Evaluating transformer...")
                trans_spearman = evaluate_sentence_similarity(
                    transformer_encoder, stsb_pairs, show_progress=True
                )
                sent_results["Transformer"] = trans_spearman
                print(f"  Transformer: ρ={trans_spearman:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("Summary: Word Similarity (Spearman ρ)")
    print("=" * 70)

    # Build header
    models = ["Ours"]
    if baseline_encoder:
        models.append("Baseline")
    if transformer_encoder:
        models.append("Transformer")

    header = f"{'Dataset':<15}"
    for m in models:
        header += f" {m:>12}"
    print(header)
    print("-" * (15 + 13 * len(models)))

    for dataset, model_results in word_results.items():
        row = f"{dataset:<15}"
        for m in models:
            if m in model_results:
                rho, cov, n = model_results[m]
                row += f" {rho:>12.4f}"
            else:
                row += f" {'N/A':>12}"
        print(row)

    if sent_results:
        print(f"\n{'STS-B':<15}", end="")
        for m in models:
            if m in sent_results:
                print(f" {sent_results[m]:>12.4f}", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
