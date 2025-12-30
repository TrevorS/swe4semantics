#!/usr/bin/env python3
"""Train static embeddings via knowledge distillation.

Usage:
    uv run python scripts/train_distillation.py \
        --embeddings outputs/embeddings_256d.txt \
        --word2sent outputs/word2sent.pkl \
        --output outputs/embeddings_distilled.txt \
        --teacher Qwen/Qwen3-Embedding-0.6B \
        --epochs 5
"""

import argparse
from pathlib import Path

import numpy as np

from qwen3_static_embeddings.data.word2sent import Word2Sent
from qwen3_static_embeddings.train import (
    TrainConfig,
    sample_training_sentences,
    train_distillation,
)
from qwen3_static_embeddings.transform.pca import load_embeddings, save_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Train via knowledge distillation")
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to input embeddings",
    )
    parser.add_argument(
        "--word2sent",
        type=Path,
        required=True,
        help="Path to word2sent pickle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for distilled embeddings",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="Teacher model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Load embeddings
    print(f"Loading embeddings from {args.embeddings}")
    word2vec, dim = load_embeddings(args.embeddings)
    print(f"Loaded {len(word2vec)} embeddings of dimension {dim}")

    # Load word2sent
    print(f"Loading word2sent from {args.word2sent}")
    word2sent = Word2Sent.load(args.word2sent)

    # Sample training sentences
    print("Sampling training sentences...")
    train_sents, val_sents = sample_training_sentences(
        word2sent,
        n_per_word=3,
        train_ratio=0.8,
        seed=args.seed,
    )
    print(f"Train sentences: {len(train_sents)}")
    print(f"Val sentences: {len(val_sents)}")

    # Training config
    config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stop_patience=5,
    )

    # Train
    print("\nStarting distillation training...")
    distilled = train_distillation(
        word2vec=word2vec,
        train_sentences=train_sents,
        val_sentences=val_sents,
        teacher_model_name=args.teacher,
        config=config,
        device=args.device,
    )

    # Save
    save_embeddings(distilled, args.output)
    print(f"\nSaved distilled embeddings to {args.output}")


if __name__ == "__main__":
    main()
