#!/usr/bin/env python3
"""Extract static word embeddings from Qwen3.

Usage:
    uv run python scripts/extract_embeddings.py \
        --word2sent outputs/word2sent.pkl \
        --output outputs/embeddings_raw.txt \
        --model Qwen/Qwen3-Embedding-0.6B \
        --n-contexts 100
"""

import argparse
from pathlib import Path

from qwen3_static_embeddings.config import ModelConfig
from qwen3_static_embeddings.data.word2sent import Word2Sent
from qwen3_static_embeddings.extract.extractor import EmbeddingExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Extract word embeddings from Qwen3")
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
        help="Output path for embeddings (word2vec format)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--n-contexts",
        type=int,
        default=100,
        help="Number of contexts per word (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for model",
    )
    parser.add_argument(
        "--subword",
        action="store_true",
        help="Enable subword matching mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading word2sent from {args.word2sent}")
    word2sent = Word2Sent.load(args.word2sent)
    print(f"Words to process: {len(word2sent)}")

    # Configure model
    config = ModelConfig(
        name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        dtype=args.dtype,
    )

    # Initialize extractor
    print(f"Initializing extractor with {args.model}")
    extractor = EmbeddingExtractor(config=config)

    # Extract embeddings
    print("Extracting embeddings...")
    extractor.extract_vocabulary(
        word2sent=word2sent,
        output_path=args.output,
        n_contexts=args.n_contexts,
        subword_mode=args.subword,
    )

    print("Done!")


if __name__ == "__main__":
    main()
