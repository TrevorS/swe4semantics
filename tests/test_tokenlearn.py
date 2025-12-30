"""Tests for Tokenlearn training module."""

import tempfile
from pathlib import Path

import numpy as np

from qwen3_static_embeddings.train.tokenlearn import (
    FeatureDataset,
    TokenlearnConfig,
    apply_sif_weighting,
    load_features,
    post_process_embeddings,
    remove_principal_component,
)


class TestTokenlearnConfig:
    """Tests for TokenlearnConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TokenlearnConfig()
        assert config.teacher_model == "Qwen/Qwen3-Embedding-0.6B"
        assert config.batch_size == 256
        assert config.learning_rate == 1e-3
        assert config.epochs == 10
        assert config.pca_dims == 256
        assert config.sif_coefficient == 1e-4

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TokenlearnConfig(
            teacher_model="test-model",
            batch_size=128,
            epochs=5,
        )
        assert config.teacher_model == "test-model"
        assert config.batch_size == 128
        assert config.epochs == 5


class TestFeatureDataset:
    """Tests for FeatureDataset."""

    def test_dataset_length(self):
        """Test dataset returns correct length."""
        texts = ["hello", "world", "test"]
        embeddings = np.random.randn(3, 256).astype(np.float32)
        dataset = FeatureDataset(texts, embeddings)
        assert len(dataset) == 3

    def test_dataset_getitem(self):
        """Test dataset returns correct items."""
        texts = ["hello", "world"]
        embeddings = np.random.randn(2, 256).astype(np.float32)
        dataset = FeatureDataset(texts, embeddings)

        text, emb = dataset[0]
        assert text == "hello"
        assert emb.shape == (256,)
        np.testing.assert_array_equal(emb, embeddings[0])

    def test_dataset_empty(self):
        """Test empty dataset."""
        texts: list[str] = []
        embeddings = np.array([]).reshape(0, 256).astype(np.float32)
        dataset = FeatureDataset(texts, embeddings)
        assert len(dataset) == 0


class TestSIFWeighting:
    """Tests for SIF weighting."""

    def test_sif_weighting_basic(self):
        """Test SIF weighting applies correctly."""
        embeddings = np.ones((100, 64), dtype=np.float32)
        # Token 0 is very frequent, token 1 is rare
        token_frequencies = {0: 10000, 1: 1, 2: 100}

        weighted = apply_sif_weighting(embeddings, token_frequencies, sif_coefficient=1e-3)

        # Frequent token should have lower weight than rare token
        assert weighted[0, 0] < weighted[1, 0]
        # Token not in frequencies should keep weight of 1
        assert weighted[50, 0] == 1.0

    def test_sif_weighting_preserves_shape(self):
        """Test SIF weighting preserves embedding shape."""
        embeddings = np.random.randn(1000, 256).astype(np.float32)
        token_frequencies = {i: i + 1 for i in range(500)}

        weighted = apply_sif_weighting(embeddings, token_frequencies)
        assert weighted.shape == embeddings.shape

    def test_sif_weighting_missing_tokens(self):
        """Test SIF weighting with tokens not in frequency dict."""
        embeddings = np.ones((100, 64), dtype=np.float32)
        # Only define frequencies for first 10 tokens
        token_frequencies = {i: 100 for i in range(10)}

        weighted = apply_sif_weighting(embeddings, token_frequencies)

        # Tokens not in frequencies should keep weight of 1
        assert weighted[50, 0] == 1.0
        # Tokens in frequencies should have weight < 1
        assert weighted[0, 0] < 1.0


class TestRemovePrincipalComponent:
    """Tests for principal component removal."""

    def test_remove_one_pc(self):
        """Test removing one principal component."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64).astype(np.float32)

        result = remove_principal_component(embeddings, n_components=1)

        assert result.shape == embeddings.shape
        # Result should be different from original
        assert not np.allclose(result, embeddings)

    def test_remove_multiple_pcs(self):
        """Test removing multiple principal components."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64).astype(np.float32)

        result = remove_principal_component(embeddings, n_components=3)

        assert result.shape == embeddings.shape

    def test_remove_pc_reduces_variance(self):
        """Test that removing PCs reduces variance in those directions."""
        np.random.seed(42)
        # Create embeddings with strong first PC
        base = np.random.randn(100, 64).astype(np.float32)
        pc1 = np.random.randn(64).astype(np.float32)
        pc1 /= np.linalg.norm(pc1)
        embeddings = base + 10 * np.outer(np.random.randn(100), pc1)

        result = remove_principal_component(embeddings, n_components=1)

        # Variance along first PC should be reduced
        original_var = np.var(embeddings @ pc1)
        result_var = np.var(result @ pc1)
        assert result_var < original_var


class TestPostProcessEmbeddings:
    """Tests for post-processing pipeline."""

    def test_post_process_full_pipeline(self):
        """Test full post-processing pipeline."""
        np.random.seed(42)
        embeddings = np.random.randn(1000, 256).astype(np.float32)
        token_frequencies = {i: np.random.randint(1, 1000) for i in range(500)}

        result = post_process_embeddings(
            embeddings=embeddings,
            token_frequencies=token_frequencies,
            apply_sif=True,
            apply_pca=False,
            remove_pc=True,
            n_pc_remove=1,
        )

        assert result.shape == embeddings.shape
        # Should be L2 normalized
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(1000), decimal=5)

    def test_post_process_no_sif(self):
        """Test post-processing without SIF."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64).astype(np.float32)

        result = post_process_embeddings(
            embeddings=embeddings,
            token_frequencies=None,
            apply_sif=False,
            apply_pca=False,
            remove_pc=True,
        )

        assert result.shape == embeddings.shape

    def test_post_process_with_pca(self):
        """Test post-processing with PCA reduction."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 256).astype(np.float32)

        result = post_process_embeddings(
            embeddings=embeddings,
            token_frequencies=None,
            apply_sif=False,
            apply_pca=True,
            pca_dims=64,
            remove_pc=True,
        )

        assert result.shape == (100, 64)

    def test_post_process_normalized_output(self):
        """Test that output is L2 normalized."""
        np.random.seed(42)
        embeddings = np.random.randn(100, 64).astype(np.float32) * 10  # Large values

        result = post_process_embeddings(
            embeddings=embeddings,
            token_frequencies=None,
            apply_sif=False,
            apply_pca=False,
            remove_pc=False,
        )

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(100), decimal=5)


class TestLoadFeatures:
    """Tests for loading features from disk."""

    def test_load_features(self):
        """Test loading saved features."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create test feature files
            texts = ["hello world", "test sentence"]
            embeddings = np.random.randn(2, 256).astype(np.float32)

            with open(tmppath / "texts_0001.json", "w") as f:
                json.dump(texts, f)
            np.save(tmppath / "embeddings_0001.npy", embeddings)

            # Load features
            loaded_texts, loaded_embeddings = load_features(tmppath)

            assert loaded_texts == texts
            assert loaded_embeddings.shape == (2, 256)
            np.testing.assert_array_almost_equal(loaded_embeddings, embeddings)

    def test_load_features_multiple_batches(self):
        """Test loading features from multiple batch files."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)

            # Create multiple batch files
            for batch_idx in range(1, 4):
                texts = [f"text_{batch_idx}_{i}" for i in range(10)]
                embeddings = np.random.randn(10, 64).astype(np.float32)

                with open(tmppath / f"texts_{batch_idx:04d}.json", "w") as f:
                    json.dump(texts, f)
                np.save(tmppath / f"embeddings_{batch_idx:04d}.npy", embeddings)

            # Load all features
            loaded_texts, loaded_embeddings = load_features(tmppath)

            assert len(loaded_texts) == 30
            assert loaded_embeddings.shape == (30, 64)
