"""
Qwen3 Static Embeddings - High-quality static word embeddings via Qwen3 distillation.

This package implements the SWE4Semantics methodology using Qwen3-Embedding models
to create static word embeddings suitable for fast CPU inference.

Pipeline:
    1. data: Build word→sentences mapping from corpus
    2. extract: Extract token-level embeddings from Qwen3
    3. aggregate: Mean pooling across contexts
    4. transform: PCA post-processing and dimensionality reduction
    5. train: Optional knowledge distillation fine-tuning

Example:
    >>> from qwen3_static_embeddings import StaticEncoder
    >>> encoder = StaticEncoder.from_pretrained("path/to/embeddings.txt")
    >>> embedding = encoder.encode("The quick brown fox")
"""

__version__ = "0.1.0"

from qwen3_static_embeddings.config import Config, ModelConfig
from qwen3_static_embeddings.encode.encoder import StaticEncoder
from qwen3_static_embeddings.export import (
    export_embeddings,
    load_glove_text,
    load_numpy,
    load_word2vec_binary,
    load_word2vec_text,
    save_glove_text,
    save_numpy,
    save_word2vec_binary,
    save_word2vec_text,
)
from qwen3_static_embeddings.logging import (
    RunConfig,
    finish_wandb,
    init_wandb,
    log_artifact,
    log_embedding_stats,
    log_epoch_metrics,
    log_evaluation_results,
    log_extraction_progress,
    log_training_step,
)

__all__ = [
    "__version__",
    "Config",
    "ModelConfig",
    "StaticEncoder",
    "export_embeddings",
    "save_word2vec_text",
    "save_word2vec_binary",
    "save_glove_text",
    "save_numpy",
    "load_word2vec_text",
    "load_word2vec_binary",
    "load_glove_text",
    "load_numpy",
    # Logging
    "RunConfig",
    "init_wandb",
    "finish_wandb",
    "log_extraction_progress",
    "log_training_step",
    "log_epoch_metrics",
    "log_evaluation_results",
    "log_embedding_stats",
    "log_artifact",
]
