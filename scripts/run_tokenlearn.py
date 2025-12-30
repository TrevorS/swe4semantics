#!/usr/bin/env python3
"""Run Tokenlearn training pipeline (POTION-style).

This script implements the full Tokenlearn/POTION pipeline:
1. Featurize corpus with teacher model
2. Train static model to match teacher embeddings
3. Post-process: Apply SIF weighting and remove principal components
4. Export trained model

Usage:
    # Step 1a: Featurize using C4 dataset (recommended, like original Tokenlearn)
    uv run python scripts/run_tokenlearn.py featurize \
        --use-c4 \
        --output outputs/tokenlearn/features \
        --model Qwen/Qwen3-Embedding-0.6B \
        --max-sentences 1000000

    # Step 1b: Or use your own corpus
    uv run python scripts/run_tokenlearn.py featurize \
        --corpus data/corpus.txt \
        --output outputs/tokenlearn/features \
        --model Qwen/Qwen3-Embedding-0.6B \
        --max-sentences 100000

    # Step 2: Train (with checkpointing, eval, and wandb logging)
    uv run python scripts/run_tokenlearn.py train \
        --features outputs/tokenlearn/features \
        --base-model outputs/model2vec_qwen3_0.6b \
        --output outputs/tokenlearn/trained \
        --epochs 20 \
        --checkpoint-every 5 \
        --eval-every 1 \
        --eval-benchmarks simlex wordsim \
        --use-wandb \
        --compile

    # Step 2b: Resume training from checkpoint
    uv run python scripts/run_tokenlearn.py train \
        --features outputs/tokenlearn/features \
        --base-model outputs/model2vec_qwen3_0.6b \
        --output outputs/tokenlearn/trained \
        --resume-from outputs/tokenlearn/trained

    # Step 3: Post-process (SIF weighting + remove principal component)
    uv run python scripts/run_tokenlearn.py post-process \
        --model outputs/tokenlearn/trained \
        --features outputs/tokenlearn/features \
        --output outputs/tokenlearn/processed

    # Step 4: Export to Model2Vec format
    uv run python scripts/run_tokenlearn.py export \
        --model outputs/tokenlearn/processed \
        --base-model outputs/model2vec_qwen3_0.6b \
        --output outputs/tokenlearn/final
"""

import argparse
from pathlib import Path


def featurize(args):
    """Generate features from teacher model."""
    from qwen3_static_embeddings.train.tokenlearn import featurize_c4, featurize_corpus

    if not args.use_c4 and not args.corpus:
        raise ValueError("Must provide either --corpus or --use-c4")

    if args.use_c4:
        # Use C4 dataset from HuggingFace (like original Tokenlearn)
        featurize_c4(
            output_dir=args.output,
            model_name=args.model,
            max_sentences=args.max_sentences,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        featurize_corpus(
            corpus_path=args.corpus,
            output_dir=args.output,
            model_name=args.model,
            max_sentences=args.max_sentences,
            batch_size=args.batch_size,
            device=args.device,
        )


def train(args):
    """Train static model with Tokenlearn."""
    from qwen3_static_embeddings.train.tokenlearn import (
        StaticModelForTraining,
        TokenlearnConfig,
        load_features,
        train_tokenlearn,
    )

    # Load features
    print(f"Loading features from {args.features}")
    texts, embeddings = load_features(args.features)
    print(f"Loaded {len(texts)} samples, embeddings shape: {embeddings.shape}")

    # Create trainable model from Model2Vec
    print(f"Loading base model from {args.base_model}")
    model = StaticModelForTraining.from_model2vec(
        args.base_model,
        output_dim=args.pca_dims,
    )
    print(f"Model vocab size: {len(model.token_to_idx)}")

    # Training config
    config = TokenlearnConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        pca_dims=args.pca_dims,
        device=args.device,
        checkpoint_every=args.checkpoint_every,
        resume_from=str(args.resume_from) if args.resume_from else None,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        compile_model=args.compile,
        eval_every=args.eval_every,
        eval_benchmarks=tuple(args.eval_benchmarks),
    )

    # Train
    train_tokenlearn(
        model=model,
        texts=texts,
        target_embeddings=embeddings,
        config=config,
        output_path=args.output,
    )


def export(args):
    """Export trained model to Model2Vec format."""
    import numpy as np
    import torch
    from model2vec import StaticModel

    print(f"Loading trained model from {args.model}")
    checkpoint = torch.load(
        Path(args.model) / "tokenlearn_model.pt",
        weights_only=False,
    )

    # Get embeddings
    embeddings = checkpoint["embeddings"]
    if hasattr(embeddings, "numpy"):
        embeddings = embeddings.numpy()
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Load base model to get tokenizer
    print(f"Loading base model from {args.base_model}")
    base_model = StaticModel.from_pretrained(args.base_model)

    # Update base model with trained embeddings
    base_model.embedding = embeddings

    # Save as Model2Vec model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(str(output_path))

    print(f"Exported Model2Vec model to {output_path}")


def post_process(args):
    """Apply post-training re-regularization (POTION-style)."""
    import torch
    from model2vec import StaticModel

    from qwen3_static_embeddings.train.tokenlearn import (
        collect_token_frequencies,
        load_features,
        post_process_embeddings,
    )

    print(f"Loading trained model from {args.model}")
    checkpoint = torch.load(
        Path(args.model) / "tokenlearn_model.pt",
        weights_only=False,
    )

    embeddings = checkpoint["embeddings"].numpy()
    token_to_idx = checkpoint["token_to_idx"]
    print(f"Loaded embeddings: {embeddings.shape}")

    # Load features to get texts for token frequency computation
    token_frequencies = None
    if args.features:
        print(f"Loading features from {args.features}")
        texts, _ = load_features(args.features)
        print(f"Loaded {len(texts)} texts for token frequency computation")

        # Load tokenizer from base model
        if args.base_model:
            base_model = StaticModel.from_pretrained(args.base_model)
            tokenizer = base_model.tokenizer
        else:
            # Try to use tokenizer from a well-known model
            print("Warning: No base model specified, using default tokenizer")
            tokenizer = None

        if tokenizer:
            print("Collecting token frequencies from training corpus...")
            token_frequencies = collect_token_frequencies(texts, tokenizer)
            print(f"Collected frequencies for {len(token_frequencies)} unique tokens")

    # Apply post-processing
    print("\nApplying post-training re-regularization...")
    processed = post_process_embeddings(
        embeddings=embeddings,
        token_frequencies=token_frequencies,
        apply_sif=args.apply_sif,
        apply_pca=args.apply_pca,
        pca_dims=args.pca_dims,
        remove_pc=args.remove_pc,
        n_pc_remove=args.n_pc_remove,
        sif_coefficient=args.sif_coef,
    )
    print(f"Processed embeddings shape: {processed.shape}")

    # Save processed model
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings": torch.tensor(processed),
            "token_to_idx": token_to_idx,
            "projection": checkpoint.get("projection"),
            "pca_components": checkpoint.get("pca_components"),
            "pca_mean": checkpoint.get("pca_mean"),
            "token_frequencies": token_frequencies,
        },
        output_path / "tokenlearn_model.pt",
    )

    print(f"Saved post-processed model to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tokenlearn training pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Featurize command
    feat_parser = subparsers.add_parser("featurize", help="Generate features from teacher")
    feat_parser.add_argument("--corpus", type=Path, help="Corpus file (optional if using --use-c4)")
    feat_parser.add_argument(
        "--use-c4", action="store_true", help="Use C4 dataset from HuggingFace"
    )
    feat_parser.add_argument("--output", type=Path, required=True, help="Output directory")
    feat_parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    feat_parser.add_argument("--max-sentences", type=int, default=100_000)
    feat_parser.add_argument("--batch-size", type=int, default=32)
    feat_parser.add_argument("--device", type=str, default="cuda")
    feat_parser.set_defaults(func=featurize)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train with Tokenlearn")
    train_parser.add_argument("--features", type=Path, required=True, help="Features directory")
    train_parser.add_argument("--base-model", type=str, required=True, help="Base Model2Vec model")
    train_parser.add_argument("--output", type=Path, required=True, help="Output directory")
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--pca-dims", type=int, default=256)
    train_parser.add_argument("--device", type=str, default="cuda")
    train_parser.add_argument(
        "--checkpoint-every", type=int, default=1, help="Save checkpoint every N epochs"
    )
    train_parser.add_argument("--resume-from", type=Path, help="Resume from checkpoint directory")
    train_parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")
    train_parser.add_argument("--wandb-project", type=str, default="tokenlearn")
    train_parser.add_argument("--wandb-run-name", type=str, default=None)
    train_parser.add_argument(
        "--compile", action="store_true", help="Use torch.compile for faster training"
    )
    train_parser.add_argument(
        "--eval-every", type=int, default=1, help="Evaluate every N epochs (0 to disable)"
    )
    train_parser.add_argument(
        "--eval-benchmarks",
        type=str,
        nargs="+",
        default=["simlex", "wordsim"],
        choices=["simlex", "wordsim", "stsb"],
        help="Benchmarks to run during training",
    )
    train_parser.set_defaults(func=train)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export trained model")
    export_parser.add_argument("--model", type=Path, required=True, help="Trained model path")
    export_parser.add_argument("--base-model", type=str, required=True, help="Base Model2Vec model")
    export_parser.add_argument("--output", type=Path, required=True, help="Output path")
    export_parser.set_defaults(func=export)

    # Post-process command (POTION-style re-regularization)
    pp_parser = subparsers.add_parser(
        "post-process", help="Apply post-training re-regularization (SIF weighting, remove PC)"
    )
    pp_parser.add_argument("--model", type=Path, required=True, help="Trained model path")
    pp_parser.add_argument(
        "--features", type=Path, help="Features directory (for token frequencies)"
    )
    pp_parser.add_argument("--base-model", type=str, help="Base Model2Vec model (for tokenizer)")
    pp_parser.add_argument("--output", type=Path, required=True, help="Output path")
    pp_parser.add_argument(
        "--apply-sif", action="store_true", default=True, help="Apply SIF weighting"
    )
    pp_parser.add_argument(
        "--no-sif", dest="apply_sif", action="store_false", help="Skip SIF weighting"
    )
    pp_parser.add_argument(
        "--sif-coef", type=float, default=1e-3, help="SIF coefficient (default: 1e-3)"
    )
    pp_parser.add_argument(
        "--apply-pca", action="store_true", default=False, help="Apply PCA reduction"
    )
    pp_parser.add_argument("--pca-dims", type=int, default=None, help="PCA target dimensions")
    pp_parser.add_argument(
        "--remove-pc", action="store_true", default=True, help="Remove principal component"
    )
    pp_parser.add_argument(
        "--no-remove-pc", dest="remove_pc", action="store_false", help="Skip removing PC"
    )
    pp_parser.add_argument("--n-pc-remove", type=int, default=1, help="Number of PCs to remove")
    pp_parser.set_defaults(func=post_process)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
