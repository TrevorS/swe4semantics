#!/usr/bin/env python3
"""End-to-end test of the embedding pipeline with Qwen3-0.6B.

This script tests the full pipeline:
1. Build vocabulary from test corpus
2. Build word2sent mapping
3. Extract embeddings using Qwen3-0.6B
4. Apply PCA post-processing
5. Test the resulting encoder

Usage:
    uv run python scripts/test_e2e.py
"""

import tempfile
from pathlib import Path

# Check for GPU
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

from qwen3_static_embeddings.config import Config
from qwen3_static_embeddings.data.vocabulary import build_vocabulary
from qwen3_static_embeddings.data.word2sent import build_word2sent
from qwen3_static_embeddings.encode.encoder import StaticEncoder
from qwen3_static_embeddings.extract.extractor import EmbeddingExtractor
from qwen3_static_embeddings.transform.pca import save_embeddings, transform_embeddings


def main():
    # Use testing config (0.6B model, small vocab)
    config = Config.for_testing()

    # Paths
    corpus_path = Path("data/test_corpus.txt")

    if not corpus_path.exists():
        print(f"Error: Test corpus not found at {corpus_path}")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        vocab_path = tmpdir / "vocab.pkl"
        word2sent_path = tmpdir / "word2sent.pkl"
        raw_emb_path = tmpdir / "embeddings_raw.txt"
        final_emb_path = tmpdir / "embeddings_256d.txt"

        # ============================================================
        # Step 1: Build vocabulary
        # ============================================================
        print("\n" + "=" * 60)
        print("Step 1: Building vocabulary")
        print("=" * 60)

        vocab = build_vocabulary(
            corpus_path=corpus_path,
            tokenizer_name=config.data.word_tokenizer,
            lowercase=True,
        )
        print(f"Raw vocabulary size: {len(vocab)}")

        # Filter to small vocab for testing
        vocab = vocab.filter(
            min_freq=1,  # Low threshold for small corpus
            min_len=3,
            max_size=100,  # Small vocab for quick test
        )
        print(f"Filtered vocabulary size: {len(vocab)}")
        vocab.save(vocab_path)

        # Show top words
        print("Top 10 words:", vocab.words[:10])

        # ============================================================
        # Step 2: Build word2sent mapping
        # ============================================================
        print("\n" + "=" * 60)
        print("Step 2: Building word2sent mapping")
        print("=" * 60)

        word2sent = build_word2sent(
            corpus_path=corpus_path,
            vocab=vocab,
            n_sentences=10,  # Few sentences per word for testing
            tokenizer_name=config.data.word_tokenizer,
            lowercase=True,
        )
        word2sent.save(word2sent_path)

        # Show example
        example_word = vocab.words[0]
        example_sents = word2sent.get_sentences(example_word, n=2)
        print(f"\nExample sentences for '{example_word}':")
        for s in example_sents:
            print(f"  - {s[:60]}...")

        # ============================================================
        # Step 3: Extract embeddings with Qwen3-0.6B
        # ============================================================
        print("\n" + "=" * 60)
        print("Step 3: Extracting embeddings with Qwen3-0.6B")
        print("=" * 60)

        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        if device == "cpu":
            print("Warning: Running on CPU will be slow. GPU recommended.")
            config.model.device = "cpu"
            config.model.dtype = "float32"

        # Initialize extractor
        print(f"\nLoading model: {config.model.name}")
        extractor = EmbeddingExtractor(config=config.model)

        # Extract embeddings
        print("\nExtracting embeddings...")
        embeddings = extractor.extract_vocabulary(
            word2sent=word2sent,
            output_path=raw_emb_path,
            n_contexts=10,
        )

        print(f"Extracted {len(embeddings)} word embeddings")
        if embeddings:
            first_word = list(embeddings.keys())[0]
            print(f"Embedding dimension: {embeddings[first_word].shape}")

        # ============================================================
        # Step 4: Apply PCA post-processing
        # ============================================================
        print("\n" + "=" * 60)
        print("Step 4: Applying PCA post-processing")
        print("=" * 60)

        if len(embeddings) < 10:
            print("Warning: Too few embeddings for PCA, skipping...")
            final_embeddings = embeddings
        else:
            # Sample sentences for PCA
            all_sentences = []
            for word in word2sent.words:
                all_sentences.extend(word2sent.get_sentences(word))

            # Create temporary encoder for sentence embeddings
            raw_dim = list(embeddings.values())[0].shape[0]
            temp_encoder = StaticEncoder(
                word2vec=embeddings,
                dim=raw_dim,
                normalize=False,
            )

            # Encode sentences
            sample_sentences = all_sentences[: min(100, len(all_sentences))]
            sentence_embeddings = temp_encoder.encode_batch(sample_sentences)
            print(f"Encoded {len(sample_sentences)} sentences for PCA fitting")

            # Determine output dim (can't exceed n_samples or n_embeddings)
            max_dim = min(len(embeddings), len(sample_sentences), raw_dim)
            output_dim = min(config.pca.output_dim, max_dim - 1)
            n_remove = min(config.pca.n_components_remove, max_dim // 2)

            print(f"Removing {n_remove} principal components")
            print(f"Reducing to {output_dim} dimensions")

            # Transform
            final_embeddings = transform_embeddings(
                word_embeddings=embeddings,
                sentence_embeddings=sentence_embeddings,
                n_components_remove=n_remove,
                output_dim=output_dim,
            )

            save_embeddings(final_embeddings, final_emb_path)

        # ============================================================
        # Step 5: Test the encoder
        # ============================================================
        print("\n" + "=" * 60)
        print("Step 5: Testing the encoder")
        print("=" * 60)

        final_dim = list(final_embeddings.values())[0].shape[0]
        encoder = StaticEncoder(
            word2vec=final_embeddings,
            dim=final_dim,
            normalize=True,
        )

        print(f"Encoder vocabulary size: {len(encoder)}")
        print(f"Embedding dimension: {final_dim}")

        # Test sentences
        test_pairs = [
            ("the quick brown fox", "a fast brown dog"),
            ("machine learning models", "deep learning networks"),
            ("the bank approved my loan", "the river bank was muddy"),
            ("python programming language", "data science tools"),
        ]

        print("\nSimilarity tests:")
        for s1, s2 in test_pairs:
            sim = encoder.similarity(s1, s2)
            print(f"  '{s1}' vs '{s2}': {sim:.3f}")

        # Self-similarity test
        test_sent = "neural networks learn representations"
        self_sim = encoder.similarity(test_sent, test_sent)
        print(f"\nSelf-similarity test: '{test_sent}' = {self_sim:.3f}")

        print("\n" + "=" * 60)
        print("End-to-end test completed successfully!")
        print("=" * 60)


if __name__ == "__main__":
    main()
