#!/usr/bin/env python3
"""Build vocabulary from a corpus file.

Usage:
    uv run python scripts/build_vocabulary.py \
        --corpus data/corpus.txt \
        --output outputs/vocab.pkl \
        --vocab-size 150000
"""

import argparse
from pathlib import Path

from qwen3_static_embeddings.data.vocabulary import build_vocabulary


def parse_args():
    parser = argparse.ArgumentParser(description="Build vocabulary from corpus")
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus file (text, one document per line)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for vocabulary pickle",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=150_000,
        help="Maximum vocabulary size (default: 150000)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=10,
        help="Minimum word frequency (default: 10)",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=3,
        help="Minimum word length (default: 3)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="bert-base-uncased",
        help="HuggingFace tokenizer for word tokenization",
    )
    parser.add_argument(
        "--no-lowercase",
        action="store_true",
        help="Do not lowercase words",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Building vocabulary from {args.corpus}")

    # Build full vocabulary
    vocab = build_vocabulary(
        corpus_path=args.corpus,
        tokenizer_name=args.tokenizer,
        lowercase=not args.no_lowercase,
    )

    print(f"Raw vocabulary size: {len(vocab)}")

    # Filter vocabulary
    vocab = vocab.filter(
        min_freq=args.min_freq,
        min_len=args.min_len,
        max_size=args.vocab_size,
    )

    print(f"Filtered vocabulary size: {len(vocab)}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    vocab.save(args.output)
    print(f"Saved vocabulary to {args.output}")


if __name__ == "__main__":
    main()
