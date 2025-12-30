#!/usr/bin/env python3
"""Download and cache all data needed for SWE4Semantics.

This script ensures all required data is available before running the pipeline:
1. Evaluation datasets (SimLex-999, WordSim-353, STS-B)
2. Pre-trained model (Qwen3-Embedding)
3. Corpus sample (optional)

Usage:
    # Download everything needed for testing
    uv run python scripts/prepare_data.py --mode test

    # Download everything for production
    uv run python scripts/prepare_data.py --mode production

    # Just download evaluation datasets
    uv run python scripts/prepare_data.py --eval-only

    # Just download the model
    uv run python scripts/prepare_data.py --model-only
"""

import argparse
from pathlib import Path


def download_eval_datasets() -> dict[str, bool]:
    """Download and cache evaluation datasets from HuggingFace.

    Returns:
        Dict mapping dataset name to success status
    """
    from datasets import load_dataset

    results = {}

    # SimLex-999
    print("\n[1/3] Downloading SimLex-999...")
    try:
        dataset = load_dataset("tasksource/simlex", split="train")
        print(f"  ✓ SimLex-999: {len(dataset)} word pairs")
        results["SimLex-999"] = True
    except Exception as e:
        print(f"  ✗ SimLex-999: {e}")
        results["SimLex-999"] = False

    # WordSim-353 (via semantic-similarity dataset)
    print("\n[2/3] Downloading WordSim-353...")
    try:
        dataset = load_dataset("StephanAkkerman/semantic-similarity", split="train")
        wordsim_count = sum(1 for item in dataset if "wordsim" in item["dataset"].lower())
        print(f"  ✓ WordSim-353: {wordsim_count} word pairs")
        results["WordSim-353"] = True
    except Exception as e:
        print(f"  ✗ WordSim-353: {e}")
        results["WordSim-353"] = False

    # STS-Benchmark
    print("\n[3/3] Downloading STS-Benchmark...")
    try:
        dataset = load_dataset("sentence-transformers/stsb", split="test")
        print(f"  ✓ STS-Benchmark: {len(dataset)} sentence pairs")
        results["STS-B"] = True
    except Exception as e:
        print(f"  ✗ STS-Benchmark: {e}")
        results["STS-B"] = False

    return results


def download_model(model_name: str) -> bool:
    """Download and cache the embedding model.

    Args:
        model_name: HuggingFace model name

    Returns:
        True if successful
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"\nDownloading model: {model_name}")
    try:
        print("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"  ✓ Tokenizer: vocab_size={tokenizer.vocab_size}")

        print("  Downloading model weights...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"  ✓ Model: {n_params:.1f}M parameters")

        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_corpus_sample(
    language: str = "en",
    n_sentences: int = 10000,
    output_dir: Path = Path("data"),
) -> bool:
    """Download a small corpus sample for testing.

    Args:
        language: Language code
        n_sentences: Number of sentences to download
        output_dir: Directory to save corpus

    Returns:
        True if successful
    """
    from qwen3_static_embeddings.data import download_cc100

    output_path = output_dir / f"cc100_{language}_sample.txt"

    if output_path.exists():
        print(f"\n✓ Corpus sample already exists: {output_path}")
        return True

    print(f"\nDownloading CC-100 sample ({n_sentences} sentences)...")
    try:
        download_cc100(
            language=language,
            output_path=output_path,
            max_sentences=n_sentences,
            streaming=True,
            show_progress=True,
        )
        print(f"  ✓ Saved to {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and cache data for SWE4Semantics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything for test mode
  %(prog)s --mode test

  # Download everything for production
  %(prog)s --mode production

  # Just evaluation datasets
  %(prog)s --eval-only

  # Just the model
  %(prog)s --model-only --model Qwen/Qwen3-Embedding-0.6B
        """,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "production"],
        help="Download data for test or production mode",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only download evaluation datasets",
    )
    parser.add_argument(
        "--model-only",
        action="store_true",
        help="Only download the model",
    )
    parser.add_argument(
        "--corpus-only",
        action="store_true",
        help="Only download corpus sample",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to download (default: based on mode)",
    )
    parser.add_argument(
        "--corpus-sentences",
        type=int,
        default=None,
        help="Number of corpus sentences to download",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for downloaded data",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SWE4Semantics Data Preparation")
    print("=" * 60)

    # Determine what to download
    if args.eval_only:
        do_eval = True
        do_model = False
        do_corpus = False
    elif args.model_only:
        do_eval = False
        do_model = True
        do_corpus = False
    elif args.corpus_only:
        do_eval = False
        do_model = False
        do_corpus = True
    elif args.mode:
        do_eval = True
        do_model = True
        do_corpus = True
    else:
        # Default: just eval datasets
        do_eval = True
        do_model = False
        do_corpus = False

    # Set model based on mode
    if args.model:
        model_name = args.model
    elif args.mode == "production":
        model_name = "Qwen/Qwen3-Embedding-8B"
    else:
        model_name = "Qwen/Qwen3-Embedding-0.6B"

    # Set corpus size based on mode
    if args.corpus_sentences:
        corpus_sentences = args.corpus_sentences
    elif args.mode == "production":
        corpus_sentences = 1_000_000
    else:
        corpus_sentences = 10_000

    results = {}

    # Download evaluation datasets
    if do_eval:
        print("\n" + "-" * 40)
        print("Evaluation Datasets")
        print("-" * 40)
        eval_results = download_eval_datasets()
        results.update(eval_results)

    # Download model
    if do_model:
        print("\n" + "-" * 40)
        print("Model")
        print("-" * 40)
        results["Model"] = download_model(model_name)

    # Download corpus
    if do_corpus:
        print("\n" + "-" * 40)
        print("Corpus")
        print("-" * 40)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results["Corpus"] = download_corpus_sample(
            n_sentences=corpus_sentences,
            output_dir=args.output_dir,
        )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    success = 0
    failed = 0
    for name, status in results.items():
        if status:
            print(f"  ✓ {name}")
            success += 1
        else:
            print(f"  ✗ {name}")
            failed += 1

    print(f"\nTotal: {success} succeeded, {failed} failed")

    if failed > 0:
        print("\nSome downloads failed. Check the errors above.")
        return 1

    print("\nAll data ready!")
    return 0


if __name__ == "__main__":
    exit(main())
