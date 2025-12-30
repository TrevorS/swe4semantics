"""Knowledge distillation training for static embeddings."""

from qwen3_static_embeddings.train.dataset import (
    sample_contrastive_pairs,
    sample_training_sentences,
)
from qwen3_static_embeddings.train.distillation import (
    StaticEmbeddingModel,
    TeacherEncoder,
    TrainConfig,
    distillation_loss,
    load_checkpoint,
    save_checkpoint,
    train_distillation,
)

__all__ = [
    "TrainConfig",
    "StaticEmbeddingModel",
    "TeacherEncoder",
    "distillation_loss",
    "train_distillation",
    "save_checkpoint",
    "load_checkpoint",
    "sample_training_sentences",
    "sample_contrastive_pairs",
]
