"""Tests for training module."""

import numpy as np
import pytest
import torch

from qwen3_static_embeddings.data.word2sent import Word2Sent
from qwen3_static_embeddings.train import (
    StaticEmbeddingModel,
    TrainConfig,
    distillation_loss,
    sample_training_sentences,
)


@pytest.fixture
def simple_embeddings():
    """Create simple test embeddings."""
    return {
        "hello": np.array([1.0, 0.0, 0.0]),
        "world": np.array([0.0, 1.0, 0.0]),
        "test": np.array([0.0, 0.0, 1.0]),
        "the": np.array([0.5, 0.5, 0.0]),
        "quick": np.array([0.3, 0.3, 0.4]),
    }


def test_train_config_defaults():
    """Test TrainConfig default values."""
    config = TrainConfig()

    assert config.epochs == 15
    assert config.batch_size == 128
    assert config.temperature == 0.05
    assert config.early_stop_patience == 5


def test_static_embedding_model_init(simple_embeddings):
    """Test StaticEmbeddingModel initialization."""
    model = StaticEmbeddingModel(simple_embeddings)

    assert model.dim == 3
    assert len(model.vocab2id) == 5
    assert model.emb.num_embeddings == 6  # 5 words + padding


def test_static_embedding_model_encode(simple_embeddings):
    """Test StaticEmbeddingModel encoding."""
    model = StaticEmbeddingModel(simple_embeddings)

    embeddings = model.encode(["hello world", "the quick test"])

    assert embeddings.shape == (2, 3)


def test_static_embedding_model_get_embeddings(simple_embeddings):
    """Test extracting embeddings from model."""
    model = StaticEmbeddingModel(simple_embeddings)

    extracted = model.get_embeddings()

    assert len(extracted) == 5
    assert "hello" in extracted
    assert extracted["hello"].shape == (3,)


def test_distillation_loss():
    """Test distillation loss computation."""
    batch_size = 4

    # Create mock similarity matrices
    student_sim = torch.randn(batch_size, batch_size)
    teacher_sim = torch.randn(batch_size, batch_size)

    # Mask diagonal
    mask = torch.eye(batch_size).bool()
    student_sim = student_sim.masked_fill(mask, float("-inf"))
    teacher_sim = teacher_sim.masked_fill(mask, float("-inf"))

    loss = distillation_loss(student_sim, teacher_sim, temperature=0.05)

    assert loss.ndim == 0  # Scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_sample_training_sentences():
    """Test sentence sampling from word2sent."""
    word2sent = Word2Sent(
        mapping={
            "hello": ["hello world", "hello there", "say hello", "hello again", "hello friend"],
            "world": ["hello world", "world news", "world peace", "around the world", "new world"],
            "test": ["test case", "unit test", "test data", "test run", "final test"],
        }
    )

    train, val = sample_training_sentences(
        word2sent,
        n_per_word=3,
        train_ratio=0.8,
        seed=42,
    )

    assert len(train) > 0
    assert len(val) > 0
    assert len(train) > len(val)  # 80/20 split
    # No overlap
    assert len(set(train) & set(val)) == 0
