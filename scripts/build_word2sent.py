#!/usr/bin/env python3
"""Build word-to-sentence mapping from corpus.

Usage:
    uv run python scripts/build_word2sent.py \
        --corpus data/corpus.txt \
        --vocab outputs/vocab.pkl \
        --output outputs/word2sent.pkl \
        --n-sentences 100
"""

import argparse
from pathlib import Path

from qwen3_static_embeddings.data.vocabulary import Vocabulary
from qwen3_static_embeddings.data.word2sent import build_word2sent


def parse_args():
    parser = argparse.ArgumentParser(description="Build word to sentence mapping")
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus file",
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        required=True,
        help="Path to vocabulary pickle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for word2sent pickle",
    )
    parser.add_argument(
        "--n-sentences",
        type=int,
        default=100,
        help="Number of sentences per word (default: 100)",
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
        help="Do not lowercase text",
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

    print(f"Loading vocabulary from {args.vocab}")
    vocab = Vocabulary.load(args.vocab)
    print(f"Vocabulary size: {len(vocab)}")

    print(f"Building word2sent from {args.corpus}")
    word2sent = build_word2sent(
        corpus_path=args.corpus,
        vocab=vocab,
        n_sentences=args.n_sentences,
        tokenizer_name=args.tokenizer,
        lowercase=not args.no_lowercase,
        seed=args.seed,
    )

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    word2sent.save(args.output)
    print(f"Saved word2sent to {args.output}")


if __name__ == "__main__":
    main()
