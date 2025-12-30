#!/usr/bin/env python3
"""Full pipeline for training static word embeddings from Qwen3.

This script orchestrates the complete SWE4Semantics pipeline:
1. Download corpus (or use existing)
2. Build vocabulary
3. Build word2sent mapping
4. Extract embeddings from Qwen3
5. Apply PCA post-processing
6. (Optional) Train with knowledge distillation
7. Evaluate on benchmarks
8. Export to multiple formats

Usage:
    # Quick test run (small vocab, 0.6B model)
    uv run python scripts/run_pipeline.py --mode test --output-dir outputs/test

    # Production run (150k vocab, 8B model)
    uv run python scripts/run_pipeline.py --mode production --output-dir outputs/prod

    # Resume from checkpoint
    uv run python scripts/run_pipeline.py --resume outputs/prod
"""

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from qwen3_static_embeddings.config import Config
from qwen3_static_embeddings.data import build_vocabulary, build_word2sent, download_cc100
from qwen3_static_embeddings.encode.encoder import StaticEncoder
from qwen3_static_embeddings.evaluate import (
    create_synthetic_word_pairs,
    evaluate_word_similarity,
    load_simlex999,
    load_wordsim353,
)
from qwen3_static_embeddings.export import export_embeddings
from qwen3_static_embeddings.extract.extractor import EmbeddingExtractor
from qwen3_static_embeddings.logging import (
    RunConfig,
    finish_wandb,
    init_wandb,
    log_artifact,
    log_embedding_stats,
    log_evaluation_results,
)
from qwen3_static_embeddings.transform.pca import save_embeddings, transform_embeddings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full SWE4Semantics pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "production"],
        default="test",
        help="Run mode: 'test' (0.6B, small vocab) or 'production' (8B, full vocab)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for all outputs (checkpoints, embeddings, logs)",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Path to corpus file (if not provided, downloads CC-100)",
    )
    parser.add_argument(
        "--corpus-sentences",
        type=int,
        default=None,
        help="Max sentences to download from CC-100 (None = all)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Maximum vocabulary size (default: 100 for test, 150000 for production)",
    )
    parser.add_argument(
        "--sentences-per-word",
        type=int,
        default=None,
        help="Sentences per word for context (default: 10 for test, 100 for production)",
    )
    parser.add_argument(
        "--skip-distillation",
        action="store_true",
        help="Skip the knowledge distillation step",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from existing output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="qwen3-static-embeddings",
        help="W&B project name (default: qwen3-static-embeddings)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team/user)",
    )

    return parser.parse_args()


def setup_directories(output_dir: Path) -> dict[str, Path]:
    """Create output directory structure."""
    dirs = {
        "root": output_dir,
        "data": output_dir / "data",
        "checkpoints": output_dir / "checkpoints",
        "embeddings": output_dir / "embeddings",
        "exports": output_dir / "exports",
        "logs": output_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_config(args) -> Config:
    """Get configuration based on mode."""
    if args.mode == "test":
        config = Config.for_testing()
        if args.vocab_size is None:
            args.vocab_size = 1000
        if args.sentences_per_word is None:
            args.sentences_per_word = 10
        if args.corpus_sentences is None:
            args.corpus_sentences = 100_000  # 100k for quick test
    else:
        config = Config.for_production()
        if args.vocab_size is None:
            args.vocab_size = 150_000
        if args.sentences_per_word is None:
            args.sentences_per_word = 100
        # No limit for production

    # Override device if specified
    if args.device:
        config.model.device = args.device
    elif not torch.cuda.is_available():
        config.model.device = "cpu"
        config.model.dtype = "float32"

    return config


def load_state(output_dir: Path) -> dict:
    """Load pipeline state from disk."""
    state_path = output_dir / "pipeline_state.json"
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"completed_steps": []}


def save_state(output_dir: Path, state: dict):
    """Save pipeline state to disk."""
    state_path = output_dir / "pipeline_state.json"
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def main():
    args = parse_args()

    # Handle resume
    if args.resume:
        args.output_dir = args.resume
        print(f"Resuming from {args.output_dir}")

    # Setup
    dirs = setup_directories(args.output_dir)
    config = get_config(args)
    state = load_state(dirs["root"])

    print("=" * 60)
    print("SWE4Semantics Pipeline")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print(f"Model: {config.model.name}")
    print(f"Device: {config.model.device}")
    print(f"Output: {dirs['root']}")
    print(f"Vocab size: {args.vocab_size:,}")
    print(f"Sentences per word: {args.sentences_per_word}")
    print(f"W&B logging: {'enabled' if args.wandb else 'disabled'}")
    print("=" * 60)

    # Initialize wandb if enabled
    wandb_run = None
    if args.wandb:
        run_config = RunConfig(
            model_name=config.model.name,
            model_type="0.6B" if "0.6B" in config.model.name else "8B",
            corpus="cc100",
            language="en",
            vocab_size=args.vocab_size,
            sentences_per_word=args.sentences_per_word,
            pca_components_remove=config.pca.n_components_remove,
            output_dim=config.pca.output_dim,
            run_name=f"{args.mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            tags=[args.mode, config.model.device],
        )
        wandb_run = init_wandb(
            config=run_config,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
        print(f"W&B run: {wandb_run.url if wandb_run else 'N/A'}")

    # Save config
    with open(dirs["root"] / "config.json", "w") as f:
        json.dump(
            {
                "mode": args.mode,
                "vocab_size": args.vocab_size,
                "sentences_per_word": args.sentences_per_word,
                "model": asdict(config.model),
                "pca": asdict(config.pca),
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    # ========================================
    # Step 1: Corpus
    # ========================================
    corpus_path = dirs["data"] / "corpus.txt"

    if "corpus" not in state["completed_steps"]:
        print("\n" + "=" * 60)
        print("Step 1: Preparing corpus")
        print("=" * 60)

        if args.corpus:
            # Copy existing corpus
            print(f"Using existing corpus: {args.corpus}")
            shutil.copy(args.corpus, corpus_path)
        else:
            # Download CC-100
            print("Downloading CC-100 corpus...")
            download_cc100(
                language="en",
                output_path=corpus_path,
                max_sentences=args.corpus_sentences,
                streaming=True,
                show_progress=True,
            )

        state["completed_steps"].append("corpus")
        save_state(dirs["root"], state)
    else:
        print("\n[Skipping] Corpus already prepared")

    # ========================================
    # Step 2: Vocabulary
    # ========================================
    vocab_path = dirs["data"] / "vocab.pkl"

    if "vocabulary" not in state["completed_steps"]:
        print("\n" + "=" * 60)
        print("Step 2: Building vocabulary")
        print("=" * 60)

        vocab = build_vocabulary(
            corpus_path=corpus_path,
            tokenizer_name=config.data.word_tokenizer,
            lowercase=True,
        )
        print(f"Raw vocabulary: {len(vocab):,} words")

        vocab = vocab.filter(
            min_freq=config.data.min_word_freq,
            min_len=config.data.min_word_len,
            max_size=args.vocab_size,
        )
        print(f"Filtered vocabulary: {len(vocab):,} words")
        vocab.save(vocab_path)

        state["completed_steps"].append("vocabulary")
        state["vocab_size"] = len(vocab)
        save_state(dirs["root"], state)
    else:
        print(f"\n[Skipping] Vocabulary already built ({state.get('vocab_size', '?')} words)")
        from qwen3_static_embeddings.data import Vocabulary

        vocab = Vocabulary.load(vocab_path)

    # ========================================
    # Step 3: Word2Sent
    # ========================================
    word2sent_path = dirs["data"] / "word2sent.pkl"

    if "word2sent" not in state["completed_steps"]:
        print("\n" + "=" * 60)
        print("Step 3: Building word2sent mapping")
        print("=" * 60)

        word2sent = build_word2sent(
            corpus_path=corpus_path,
            vocab=vocab,
            n_sentences=args.sentences_per_word,
            tokenizer_name=config.data.word_tokenizer,
            lowercase=True,
        )
        word2sent.save(word2sent_path)

        state["completed_steps"].append("word2sent")
        save_state(dirs["root"], state)
    else:
        print("\n[Skipping] Word2sent already built")
        from qwen3_static_embeddings.data import Word2Sent

        word2sent = Word2Sent.load(word2sent_path)

    # ========================================
    # Step 4: Extract embeddings
    # ========================================
    raw_emb_path = dirs["embeddings"] / "embeddings_raw.txt"

    if "extraction" not in state["completed_steps"]:
        print("\n" + "=" * 60)
        print("Step 4: Extracting embeddings")
        print("=" * 60)

        extractor = EmbeddingExtractor(config=config.model)
        embeddings = extractor.extract_vocabulary(
            word2sent=word2sent,
            output_path=raw_emb_path,
            n_contexts=args.sentences_per_word,
        )

        print(f"Extracted {len(embeddings):,} word embeddings")

        # Log embedding stats to wandb
        if args.wandb:
            log_embedding_stats(embeddings, prefix="raw_embeddings")

        state["completed_steps"].append("extraction")
        state["n_embeddings"] = len(embeddings)
        save_state(dirs["root"], state)
    else:
        print(f"\n[Skipping] Embeddings already extracted ({state.get('n_embeddings', '?')} words)")
        from qwen3_static_embeddings.transform.pca import load_embeddings

        embeddings, _ = load_embeddings(raw_emb_path)

    # ========================================
    # Step 5: PCA post-processing
    # ========================================
    pca_emb_path = dirs["embeddings"] / "embeddings_pca.txt"

    if "pca" not in state["completed_steps"]:
        print("\n" + "=" * 60)
        print("Step 5: Applying PCA post-processing")
        print("=" * 60)

        # Get sentence embeddings for PCA
        all_sentences = []
        for word in list(word2sent.words)[: min(1000, len(word2sent.words))]:
            all_sentences.extend(word2sent.get_sentences(word, n=5))

        raw_dim = list(embeddings.values())[0].shape[0]
        temp_encoder = StaticEncoder(word2vec=embeddings, dim=raw_dim, normalize=False)
        sentence_embeddings = temp_encoder.encode_batch(all_sentences[:5000])

        print(f"Using {len(sentence_embeddings)} sentences for PCA fitting")

        # Ensure dimensions are valid
        n_remove = min(config.pca.n_components_remove, len(sentence_embeddings) // 2)
        output_dim = min(config.pca.output_dim, len(sentence_embeddings) - 1, raw_dim - n_remove)

        final_embeddings = transform_embeddings(
            word_embeddings=embeddings,
            sentence_embeddings=sentence_embeddings,
            n_components_remove=n_remove,
            output_dim=output_dim,
        )

        save_embeddings(final_embeddings, pca_emb_path)

        # Log final embedding stats to wandb
        if args.wandb:
            log_embedding_stats(final_embeddings, prefix="final_embeddings")

        state["completed_steps"].append("pca")
        state["output_dim"] = output_dim
        save_state(dirs["root"], state)
    else:
        print(f"\n[Skipping] PCA already applied (output dim: {state.get('output_dim', '?')})")
        from qwen3_static_embeddings.transform.pca import load_embeddings

        final_embeddings, _ = load_embeddings(pca_emb_path)

    # ========================================
    # Step 6: Export
    # ========================================
    if "export" not in state["completed_steps"]:
        print("\n" + "=" * 60)
        print("Step 6: Exporting embeddings")
        print("=" * 60)

        export_embeddings(
            final_embeddings,
            output_dir=dirs["exports"],
            name="qwen3_static",
            formats=["word2vec_text", "word2vec_binary", "glove_text", "numpy"],
        )

        state["completed_steps"].append("export")
        save_state(dirs["root"], state)
    else:
        print("\n[Skipping] Already exported")

    # ========================================
    # Step 7: Evaluate
    # ========================================
    print("\n" + "=" * 60)
    print("Step 7: Evaluation")
    print("=" * 60)

    # Load evaluation datasets
    results = {}
    datasets = {
        "SimLex-999": load_simlex999(),
        "WordSim-353": load_wordsim353(),
        "Synthetic": create_synthetic_word_pairs(),
    }

    print("\nWord Similarity Results:")
    for name, word_pairs in datasets.items():
        if word_pairs:
            result = evaluate_word_similarity(final_embeddings, word_pairs, name)
            results[name] = {
                "spearman": result.spearman_rho,
                "pearson": result.pearson_r,
                "coverage": result.coverage,
                "n_pairs": result.n_pairs,
                "n_found": result.n_found,
            }
            print(f"  {name}: Spearman={result.spearman_rho:.3f}, coverage={result.coverage:.1%}")
        else:
            results[name] = {"error": "Dataset not available"}
            print(f"  {name}: Dataset not available")

    # Log evaluation results to wandb
    if args.wandb:
        log_evaluation_results(results)

    # Save results
    with open(dirs["root"] / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ========================================
    # Done
    # ========================================
    print("\n" + "=" * 60)
    print("Pipeline completed!")
    print("=" * 60)
    print(f"\nOutputs saved to: {dirs['root']}")
    print(f"  - Embeddings: {dirs['exports']}")
    print(f"  - Evaluation: {dirs['root'] / 'evaluation_results.json'}")

    # Log artifacts and finish wandb
    if args.wandb:
        log_artifact(
            path=dirs["exports"],
            name=f"embeddings-{args.mode}",
            artifact_type="embeddings",
            metadata={
                "vocab_size": len(final_embeddings),
                "dim": list(final_embeddings.values())[0].shape[0],
                "mode": args.mode,
            },
        )
        finish_wandb()
        print("W&B run completed")


if __name__ == "__main__":
    main()
