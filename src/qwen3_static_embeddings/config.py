"""Configuration classes for the embedding pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for the Qwen3 embedding model."""

    # Model selection
    name: str = "Qwen/Qwen3-Embedding-0.6B"
    trust_remote_code: bool = True

    # Embedding dimensions
    output_dim: int = 1024  # Qwen3 supports 32-1024
    final_dim: int = 256  # After PCA reduction

    # Inference settings
    max_length: int = 512  # Max tokens per sentence
    batch_size: int = 32
    device: str = "cuda"
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"

    @property
    def model_8b(self) -> str:
        """Return the 8B model name for production."""
        return "Qwen/Qwen3-Embedding-8B"

    @property
    def model_0_6b(self) -> str:
        """Return the 0.6B model name for testing."""
        return "Qwen/Qwen3-Embedding-0.6B"


@dataclass
class DataConfig:
    """Configuration for corpus and vocabulary processing."""

    # Vocabulary
    vocab_size: int = 150_000
    min_word_freq: int = 10
    min_word_len: int = 3

    # Context collection
    contexts_per_word: int = 100
    max_sentence_len: int = 500  # Skip very long sentences

    # Word tokenization (BERT-style pre-tokenizer)
    word_tokenizer: str = "bert-base-uncased"


@dataclass
class PCAConfig:
    """Configuration for PCA post-processing."""

    # Sentence-level PCA (remove dominant directions)
    n_components_remove: int = 7
    n_sentences_fit: int = 10_000

    # Dimensionality reduction
    output_dim: int = 256


@dataclass
class TrainConfig:
    """Configuration for knowledge distillation training."""

    # Training
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-4

    # Distillation
    temperature: float = 0.05
    sentences_per_word: int = 3

    # Validation
    val_size: int = 10_000
    early_stop_patience: int = 5


@dataclass
class Config:
    """Main configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # Paths
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    cache_dir: Path = field(default_factory=lambda: Path(".cache"))

    def __post_init__(self):
        """Ensure paths are Path objects."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)

    @classmethod
    def for_testing(cls) -> "Config":
        """Return a config suitable for quick testing with 0.6B model."""
        config = cls()
        config.model.name = config.model.model_0_6b
        config.data.vocab_size = 1_000
        config.data.contexts_per_word = 10
        config.pca.n_sentences_fit = 100
        config.train.epochs = 1
        return config

    @classmethod
    def for_production(cls) -> "Config":
        """Return a config for full production run with 8B model."""
        config = cls()
        config.model.name = config.model.model_8b
        config.model.batch_size = 16  # Smaller batch for 8B model
        return config
