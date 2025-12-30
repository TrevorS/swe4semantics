#!/usr/bin/env python3
"""Download corpus data for SWE4Semantics replication.

The original paper uses:
- CC-100 for English (https://data.statmt.org/cc-100/)
- CCMatrix for cross-lingual (https://opus.nlpl.eu/CCMatrix/)

Usage:
    # Download CC-100 English sample (1M sentences for testing)
    uv run python scripts/download_corpus.py --corpus cc100 --language en --max-sentences 1000000 --output data/cc100_en_1M.txt

    # Download full CC-100 English (WARNING: ~82GB)
    uv run python scripts/download_corpus.py --corpus cc100 --language en --output data/cc100_en.txt

    # Estimate corpus size
    uv run python scripts/download_corpus.py --corpus cc100 --language en --estimate
"""

import argparse
from pathlib import Path

from qwen3_static_embeddings.data import (
    download_cc100,
    download_ccmatrix,
    estimate_corpus_size,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download corpus for SWE4Semantics replication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (1000 sentences)
  %(prog)s --corpus cc100 --language en --max-sentences 1000 --output data/cc100_test.txt

  # Medium sample for development (1M sentences, ~200MB)
  %(prog)s --corpus cc100 --language en --max-sentences 1000000 --output data/cc100_1M.txt

  # Full corpus for paper replication (WARNING: 82GB for English)
  %(prog)s --corpus cc100 --language en --output data/cc100_full.txt

Paper Reference:
  The SWE4Semantics paper (EMNLP 2025) uses CC-100 with:
  - 150k vocabulary
  - 100 sentences per word
  - BlingFire for sentence splitting
        """,
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="cc100",
        choices=["cc100", "ccmatrix"],
        help="Corpus to download (default: cc100)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code: 'en', 'de', 'zh', 'ja' for cc100; 'en-de', 'en-zh', 'en-ja' for ccmatrix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (required unless --estimate)",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="Maximum sentences to download (None = full corpus)",
    )
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Only estimate corpus size, don't download",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Estimate mode
    if args.estimate:
        print(f"\nCorpus size estimate for {args.corpus} ({args.language}):")
        estimate = estimate_corpus_size(args.corpus, args.language)
        print(f"  Sentences: {estimate.get('sentences', 'unknown')}")
        print(f"  Size: {estimate.get('size_gb', 'unknown')}")
        print("\nNote: These are approximate values.")
        return

    # Download mode
    if args.output is None:
        print("Error: --output is required for download")
        return

    print(f"\n{'=' * 60}")
    print(f"Downloading {args.corpus} ({args.language})")
    print(f"{'=' * 60}")

    if args.max_sentences:
        print(f"Limiting to {args.max_sentences:,} sentences")
    else:
        print("WARNING: Downloading FULL corpus. This may be very large!")
        estimate = estimate_corpus_size(args.corpus, args.language)
        print(f"Estimated size: {estimate.get('size_gb', 'unknown')}")

        # Confirm for large downloads
        response = input("\nProceed? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled.")
            return

    print(f"Output: {args.output}")
    print()

    if args.corpus == "cc100":
        download_cc100(
            language=args.language,
            output_path=args.output,
            max_sentences=args.max_sentences,
            streaming=True,
            show_progress=True,
        )
    elif args.corpus == "ccmatrix":
        download_ccmatrix(
            lang_pair=args.language,
            output_path=args.output,
            max_pairs=args.max_sentences,
            show_progress=True,
        )

    print(f"\n{'=' * 60}")
    print("Download complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
