#!/usr/bin/env python3
"""Evaluate static embeddings on word and sentence similarity benchmarks.

Usage:
    uv run python scripts/evaluate.py \
        --embeddings outputs/embeddings_256d.txt \
        --benchmarks wordsim simlex stsb
"""

import argparse
from pathlib import Path

from qwen3_static_embeddings.encode.encoder import StaticEncoder
from qwen3_static_embeddings.evaluate import (
    create_synthetic_sentence_pairs,
    create_synthetic_word_pairs,
    evaluate_sts,
    evaluate_word_similarity,
    load_simlex999,
    load_stsb,
    load_wordsim353,
)
from qwen3_static_embeddings.transform.pca import load_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate static embeddings")
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to embeddings file",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["synthetic"],
        choices=["wordsim", "simlex", "stsb", "synthetic", "all"],
        help="Benchmarks to evaluate on",
    )
    parser.add_argument(
        "--model-tokenizer",
        type=str,
        default=None,
        help="Model tokenizer for subword fallback",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading embeddings from {args.embeddings}")
    word2vec, dim = load_embeddings(args.embeddings)
    print(f"Loaded {len(word2vec)} embeddings of dimension {dim}")

    # Load model tokenizer for subword fallback if specified
    model_tokenizer = None
    if args.model_tokenizer:
        from transformers import AutoTokenizer

        model_tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer)

    # Create encoder
    encoder = StaticEncoder(
        word2vec=word2vec,
        dim=dim,
        model_tokenizer=model_tokenizer,
        normalize=True,
    )

    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["wordsim", "simlex", "stsb", "synthetic"]

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Word similarity benchmarks
    if "wordsim" in benchmarks:
        print("\n--- WordSim-353 ---")
        pairs = load_wordsim353()
        if pairs:
            result = evaluate_word_similarity(word2vec, pairs, "WordSim-353")
            print(f"Spearman ρ: {result.spearman_rho:.4f}")
            print(f"Pearson r: {result.pearson_r:.4f}")
            print(f"Coverage: {result.coverage:.1%} ({result.n_found}/{result.n_pairs})")
        else:
            print("Could not load WordSim-353")

    if "simlex" in benchmarks:
        print("\n--- SimLex-999 ---")
        pairs = load_simlex999()
        if pairs:
            result = evaluate_word_similarity(word2vec, pairs, "SimLex-999")
            print(f"Spearman ρ: {result.spearman_rho:.4f}")
            print(f"Pearson r: {result.pearson_r:.4f}")
            print(f"Coverage: {result.coverage:.1%} ({result.n_found}/{result.n_pairs})")
        else:
            print("Could not load SimLex-999")

    # Sentence similarity benchmarks
    if "stsb" in benchmarks:
        print("\n--- STS-Benchmark (test) ---")
        pairs = load_stsb("test")
        if pairs:
            result = evaluate_sts(encoder, pairs, "STS-B", show_progress=True)
            print(f"Spearman ρ: {result.spearman_rho:.4f}")
            print(f"Pearson r: {result.pearson_r:.4f}")
            print(f"Pairs: {result.n_pairs}")
        else:
            print("Could not load STS-B")

    # Synthetic benchmarks (always available)
    if "synthetic" in benchmarks:
        print("\n--- Synthetic Word Pairs ---")
        pairs = create_synthetic_word_pairs()
        result = evaluate_word_similarity(word2vec, pairs, "Synthetic")
        print(f"Spearman ρ: {result.spearman_rho:.4f}")
        print(f"Coverage: {result.coverage:.1%} ({result.n_found}/{result.n_pairs})")

        print("\n--- Synthetic Sentence Pairs ---")
        pairs = create_synthetic_sentence_pairs()
        result = evaluate_sts(encoder, pairs, "Synthetic", show_progress=False)
        print(f"Spearman ρ: {result.spearman_rho:.4f}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
