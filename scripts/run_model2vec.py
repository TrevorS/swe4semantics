#!/usr/bin/env python3
"""Run Model2Vec distillation and evaluation pipeline.

This script performs POTION-style static embedding distillation:
1. Distill static embeddings from a transformer model
2. Apply post-processing (remove PC, normalize)
3. Evaluate on benchmarks
4. Compare against POTION baselines

Usage:
    # Step 1: Distill Qwen3-0.6B to static embeddings
    uv run python scripts/run_model2vec.py distill \
        --model Qwen/Qwen3-Embedding-0.6B \
        --output outputs/qwen3_static \
        --device cpu

    # Step 2: Post-process (remove principal component)
    uv run python scripts/run_model2vec.py post-process \
        --model outputs/qwen3_static \
        --output outputs/qwen3_static_pp \
        --n-pc-remove 1

    # Step 3: Evaluate the model
    uv run python scripts/run_model2vec.py evaluate \
        --model outputs/qwen3_static_pp

    # Step 4: Compare against POTION baseline
    uv run python scripts/run_model2vec.py compare \
        --model outputs/qwen3_static_pp \
        --baseline minishlab/potion-base-8M

    # Or run full pipeline: distill + evaluate + compare
    uv run python scripts/run_model2vec.py full \
        --model Qwen/Qwen3-Embedding-0.6B \
        --output outputs/qwen3_static \
        --baseline minishlab/potion-base-8M
"""

import argparse
from pathlib import Path

import numpy as np


def distill(args):
    """Distill static embeddings from transformer model."""
    from model2vec.distill import distill as m2v_distill

    print(f"Distilling static embeddings from {args.model}")
    print(f"  PCA dims: {args.pca_dims}")
    print(f"  SIF coefficient: {args.sif_coef}")
    print(f"  Device: {args.device}")

    # Distill the model
    static_model = m2v_distill(
        model_name=args.model,
        pca_dims=args.pca_dims,
        sif_coefficient=args.sif_coef,
        device=args.device,
        trust_remote_code=True,
        quantize_to="float32",  # Keep full precision for now
    )

    # Save the model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    static_model.save_pretrained(str(output_path))
    print(f"\nSaved static model to {output_path}")

    # Print model stats
    print("\nModel statistics:")
    print(f"  Vocabulary size: {len(static_model.tokenizer.get_vocab()):,}")
    print(f"  Embedding dim: {static_model.dim}")

    return static_model


def evaluate(args):
    """Evaluate a static model on benchmarks."""
    from model2vec import StaticModel
    from scipy import stats
    from tqdm import tqdm

    from qwen3_static_embeddings.evaluate import (
        load_simlex999,
        load_stsb,
        load_wordsim353,
    )

    print(f"Loading model from {args.model}")
    model = StaticModel.from_pretrained(str(args.model))
    print(f"  Vocabulary size: {len(model.tokenizer.get_vocab()):,}")
    print(f"  Embedding dim: {model.dim}")

    results = {}

    # Word similarity benchmarks
    print("\n" + "=" * 60)
    print("Word Similarity Benchmarks")
    print("=" * 60)

    for name, loader in [("SimLex-999", load_simlex999), ("WordSim-353", load_wordsim353)]:
        pairs = loader()
        if not pairs:
            print(f"\n{name}: Dataset not available")
            continue

        predictions = []
        gold_scores = []

        for word1, word2, gold in pairs:
            try:
                emb1 = model.encode(word1)
                emb2 = model.encode(word2)
                emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
                emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
                sim = float(np.dot(emb1, emb2))
                predictions.append(sim)
                gold_scores.append(gold)
            except Exception:
                continue

        if len(predictions) >= 2:
            spearman, _ = stats.spearmanr(predictions, gold_scores)
            results[name] = {"spearman": spearman, "n": len(predictions)}
            print(f"\n{name}:")
            print(f"  Spearman ρ: {spearman:.4f}")
            print(f"  Pairs evaluated: {len(predictions)}/{len(pairs)}")

    # Sentence similarity benchmark
    print("\n" + "=" * 60)
    print("Sentence Similarity Benchmarks")
    print("=" * 60)

    stsb_pairs = load_stsb("test")
    if stsb_pairs:
        predictions = []
        gold_scores = []

        for sent1, sent2, gold in tqdm(stsb_pairs, desc="STS-B"):
            try:
                emb1 = model.encode(sent1)
                emb2 = model.encode(sent2)
                emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
                emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
                sim = float(np.dot(emb1, emb2))
                predictions.append(sim)
                gold_scores.append(gold)
            except Exception:
                continue

        if len(predictions) >= 2:
            spearman, _ = stats.spearmanr(predictions, gold_scores)
            results["STS-B"] = {"spearman": spearman, "n": len(predictions)}
            print("\nSTS-B (test):")
            print(f"  Spearman ρ: {spearman:.4f}")
            print(f"  Pairs evaluated: {len(predictions)}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, res in results.items():
        print(f"  {name}: ρ={res['spearman']:.4f} (n={res['n']})")

    return results


def compare(args):
    """Compare our model against a baseline."""
    from model2vec import StaticModel
    from scipy import stats
    from tqdm import tqdm

    from qwen3_static_embeddings.evaluate import (
        load_simlex999,
        load_stsb,
        load_wordsim353,
    )

    print(f"Loading our model from {args.model}")
    our_model = StaticModel.from_pretrained(str(args.model))

    print(f"Loading baseline from {args.baseline}")
    baseline_model = StaticModel.from_pretrained(args.baseline)

    print("\nModel comparison:")
    print(f"  Ours:     vocab={len(our_model.tokenizer.get_vocab()):,}, dim={our_model.dim}")
    print(
        f"  Baseline: vocab={len(baseline_model.tokenizer.get_vocab()):,}, dim={baseline_model.dim}"
    )

    def eval_model(model, pairs, is_sentence=False):
        predictions = []
        gold_scores = []
        for item in pairs:
            if is_sentence:
                text1, text2, gold = item
            else:
                text1, text2, gold = item
            try:
                emb1 = model.encode(text1)
                emb2 = model.encode(text2)
                emb1 = emb1 / (np.linalg.norm(emb1) + 1e-8)
                emb2 = emb2 / (np.linalg.norm(emb2) + 1e-8)
                sim = float(np.dot(emb1, emb2))
                predictions.append(sim)
                gold_scores.append(gold)
            except Exception:
                continue
        if len(predictions) >= 2:
            spearman, _ = stats.spearmanr(predictions, gold_scores)
            return spearman, len(predictions)
        return 0.0, 0

    results = {"ours": {}, "baseline": {}}

    # Word similarity
    print("\n" + "=" * 70)
    print("Comparison Results")
    print("=" * 70)
    print(f"\n{'Dataset':<15} {'Ours':>12} {'Baseline':>12} {'Delta':>10}")
    print("-" * 50)

    for name, loader in [("SimLex-999", load_simlex999), ("WordSim-353", load_wordsim353)]:
        pairs = loader()
        if not pairs:
            continue

        our_rho, our_n = eval_model(our_model, pairs)
        base_rho, base_n = eval_model(baseline_model, pairs)
        delta = our_rho - base_rho
        results["ours"][name] = our_rho
        results["baseline"][name] = base_rho

        print(f"{name:<15} {our_rho:>12.4f} {base_rho:>12.4f} {delta:>+10.4f}")

    # STS-B
    stsb_pairs = load_stsb("test")
    if stsb_pairs:
        our_rho, _ = eval_model(our_model, tqdm(stsb_pairs, desc="Ours STS-B"), is_sentence=True)
        base_rho, _ = eval_model(
            baseline_model, tqdm(stsb_pairs, desc="Baseline STS-B"), is_sentence=True
        )
        delta = our_rho - base_rho
        results["ours"]["STS-B"] = our_rho
        results["baseline"]["STS-B"] = base_rho
        print(f"{'STS-B':<15} {our_rho:>12.4f} {base_rho:>12.4f} {delta:>+10.4f}")

    # Summary
    print("\n" + "=" * 70)
    avg_ours = np.mean(list(results["ours"].values()))
    avg_base = np.mean(list(results["baseline"].values()))
    print(f"Average:        {avg_ours:>12.4f} {avg_base:>12.4f} {avg_ours - avg_base:>+10.4f}")
    print("=" * 70)

    return results


def post_process(args):
    """Apply POTION-style post-processing to a Model2Vec model."""
    from model2vec import StaticModel

    from qwen3_static_embeddings.train.tokenlearn import (
        remove_principal_component,
    )

    print(f"Loading model from {args.model}")
    model = StaticModel.from_pretrained(str(args.model))
    print(f"  Vocabulary size: {len(model.tokenizer.get_vocab()):,}")
    print(f"  Embedding dim: {model.dim}")

    # Get embeddings as numpy array
    embeddings = model.embedding
    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Apply post-processing
    print("\nApplying POTION-style post-processing...")

    # Step 1: Remove principal components (All-but-the-Top)
    if args.remove_pc:
        print(f"  Removing {args.n_pc_remove} principal component(s)...")
        embeddings = remove_principal_component(embeddings, args.n_pc_remove)

    # Step 2: L2 normalize
    print("  L2 normalizing...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    print(f"  Processed embeddings shape: {embeddings.shape}")

    # Update model embeddings (model2vec expects numpy array)
    model.embedding = embeddings.astype(np.float32)

    # Save the model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_path))
    print(f"\nSaved post-processed model to {output_path}")

    return model


def full_pipeline(args):
    """Run full pipeline: distill, evaluate, compare."""
    print("=" * 70)
    print("Full Model2Vec Pipeline")
    print("=" * 70)

    # Step 1: Distill
    print("\n[1/3] Distilling model...")
    distill_args = argparse.Namespace(
        model=args.model,
        output=args.output,
        pca_dims=args.pca_dims,
        sif_coef=args.sif_coef,
        device=args.device,
    )
    distill(distill_args)

    # Step 2: Evaluate
    print("\n[2/3] Evaluating model...")
    eval_args = argparse.Namespace(model=args.output)
    evaluate(eval_args)

    # Step 3: Compare
    if args.baseline:
        print("\n[3/3] Comparing against baseline...")
        compare_args = argparse.Namespace(model=args.output, baseline=args.baseline)
        compare(compare_args)
    else:
        print("\n[3/3] Skipping comparison (no baseline specified)")

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Model2Vec distillation and evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Distill command
    distill_parser = subparsers.add_parser("distill", help="Distill static model from transformer")
    distill_parser.add_argument("--model", type=str, required=True, help="Source transformer model")
    distill_parser.add_argument("--output", type=Path, required=True, help="Output directory")
    distill_parser.add_argument("--pca-dims", type=int, default=256, help="PCA dimensions")
    distill_parser.add_argument("--sif-coef", type=float, default=1e-4, help="SIF coefficient")
    distill_parser.add_argument("--device", type=str, default="cuda", help="Device")
    distill_parser.set_defaults(func=distill)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate static model")
    eval_parser.add_argument("--model", type=Path, required=True, help="Model path")
    eval_parser.set_defaults(func=evaluate)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare against baseline")
    compare_parser.add_argument("--model", type=Path, required=True, help="Our model path")
    compare_parser.add_argument(
        "--baseline", type=str, required=True, help="Baseline model (HuggingFace)"
    )
    compare_parser.set_defaults(func=compare)

    # Post-process command
    pp_parser = subparsers.add_parser(
        "post-process", help="Apply POTION-style post-processing (remove PC, normalize)"
    )
    pp_parser.add_argument("--model", type=Path, required=True, help="Input model path")
    pp_parser.add_argument("--output", type=Path, required=True, help="Output model path")
    pp_parser.add_argument(
        "--remove-pc", action="store_true", default=True, help="Remove principal components"
    )
    pp_parser.add_argument(
        "--no-remove-pc", dest="remove_pc", action="store_false", help="Skip removing PC"
    )
    pp_parser.add_argument("--n-pc-remove", type=int, default=1, help="Number of PCs to remove")
    pp_parser.set_defaults(func=post_process)

    # Full pipeline command
    full_parser = subparsers.add_parser("full", help="Full pipeline: distill + evaluate + compare")
    full_parser.add_argument("--model", type=str, required=True, help="Source transformer model")
    full_parser.add_argument("--output", type=Path, required=True, help="Output directory")
    full_parser.add_argument("--pca-dims", type=int, default=256, help="PCA dimensions")
    full_parser.add_argument("--sif-coef", type=float, default=1e-4, help="SIF coefficient")
    full_parser.add_argument("--device", type=str, default="cuda", help="Device")
    full_parser.add_argument(
        "--baseline", type=str, default=None, help="Baseline model for comparison"
    )
    full_parser.set_defaults(func=full_pipeline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
