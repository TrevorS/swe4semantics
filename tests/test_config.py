"""Tests for configuration module."""

from qwen3_static_embeddings.config import Config, DataConfig, ModelConfig, PCAConfig


def test_model_config_defaults():
    """Test ModelConfig has sensible defaults."""
    config = ModelConfig()

    assert config.name == "Qwen/Qwen3-Embedding-0.6B"
    assert config.output_dim == 1024
    assert config.final_dim == 256
    assert config.device == "cuda"


def test_model_config_properties():
    """Test model name properties."""
    config = ModelConfig()

    assert "0.6B" in config.model_0_6b
    assert "8B" in config.model_8b


def test_data_config_defaults():
    """Test DataConfig has sensible defaults."""
    config = DataConfig()

    assert config.vocab_size == 150_000
    assert config.contexts_per_word == 100
    assert config.min_word_freq == 10


def test_pca_config_defaults():
    """Test PCAConfig has sensible defaults."""
    config = PCAConfig()

    assert config.n_components_remove == 7
    assert config.output_dim == 256


def test_config_for_testing():
    """Test testing configuration."""
    config = Config.for_testing()

    assert "0.6B" in config.model.name
    assert config.data.vocab_size == 1_000
    assert config.data.contexts_per_word == 10


def test_config_for_production():
    """Test production configuration."""
    config = Config.for_production()

    assert "8B" in config.model.name
    assert config.model.batch_size < 32  # Smaller for 8B
