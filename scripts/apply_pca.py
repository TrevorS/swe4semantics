#!/usr/bin/env python3
"""Apply PCA post-processing to embeddings.

Usage:
    uv run python scripts/apply_pca.py \
        --embeddings outputs/embeddings_raw.txt \
        --word2sent outputs/word2sent.pkl \
        --output outputs/embeddings_256d.txt \
        --output-dim 256 \
        --n-remove 7
"""

import argparse
from pathlib import Path

import numpy as np

from qwen3_static_embeddings.data.word2sent import Word2Sent
from qwen3_static_embeddings.encode.encoder import StaticEncoder
from qwen3_static_embeddings.transform.pca import (
    load_embeddings,
    save_embeddings,
    transform_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Apply PCA post-processing")
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to raw embeddings file",
    )
    parser.add_argument(
        "--word2sent",
        type=Path,
        required=True,
        help="Path to word2sent pickle (for sentence sampling)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for processed embeddings",
    )
    parser.add_argument(
        "--output-dim",
        type=int,
        default=256,
        help="Output embedding dimension (default: 256)",
    )
    parser.add_argument(
        "--n-remove",
        type=int,
        default=7,
        help="Number of principal components to remove (default: 7)",
    )
    parser.add_argument(
        "--n-sentences",
        type=int,
        default=10000,
        help="Number of sentences for fitting PCA (default: 10000)",
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

    # Load word2sent for sampling sentences
    print(f"Loading word2sent from {args.word2sent}")
    word2sent = Word2Sent.load(args.word2sent)

    # Sample sentences for PCA fitting
    print(f"Sampling {args.n_sentences} sentences for PCA fitting...")
    all_sentences = []
    for word in word2sent.words:
        all_sentences.extend(word2sent.get_sentences(word))

    np.random.shuffle(all_sentences)
    sample_sentences = all_sentences[: args.n_sentences]
    print(f"Sampled {len(sample_sentences)} sentences")

    # Create temporary encoder for sentence embeddings
    encoder = StaticEncoder(word2vec=word2vec, dim=dim, normalize=False)

    # Encode sentences
    print("Encoding sentences for PCA...")
    sentence_embeddings = encoder.encode_batch(sample_sentences)
    print(f"Sentence embeddings shape: {sentence_embeddings.shape}")

    # Apply transformation pipeline
    print("Applying PCA transformation...")
    transformed = transform_embeddings(
        word_embeddings=word2vec,
        sentence_embeddings=sentence_embeddings,
        n_components_remove=args.n_remove,
        output_dim=args.output_dim,
    )

    # Save
    save_embeddings(transformed, args.output)
    print("Done!")


if __name__ == "__main__":
    main()
