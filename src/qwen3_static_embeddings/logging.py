"""Logging and metrics tracking with Weights & Biases.

This module provides utilities for tracking experiments, logging metrics,
and visualizing training progress using wandb.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass
class RunConfig:
    """Configuration for a training/pipeline run."""

    # Model
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    model_type: str = "0.6B"

    # Data
    corpus: str = "cc100"
    language: str = "en"
    vocab_size: int = 150_000
    sentences_per_word: int = 100

    # PCA
    pca_components_remove: int = 7
    output_dim: int = 256

    # Training
    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-4
    temperature: float = 0.05

    # Run info
    run_name: str | None = None
    tags: list[str] | None = None


def init_wandb(
    config: RunConfig,
    project: str = "qwen3-static-embeddings",
    entity: str | None = None,
    mode: Literal["online", "offline", "disabled", "shared"] = "online",
) -> Any:
    """
    Initialize a wandb run for experiment tracking.

    Args:
        config: Run configuration
        project: wandb project name
        entity: wandb entity (team/user)
        mode: "online", "offline", or "disabled"

    Returns:
        wandb run object
    """
    try:
        import wandb
    except ImportError as err:
        raise ImportError("Please install wandb: pip install wandb") from err

    run = wandb.init(
        project=project,
        entity=entity,
        config=asdict(config),
        name=config.run_name,
        tags=config.tags,
        mode=mode,
    )

    return run


def log_extraction_progress(
    step: int,
    total: int,
    word: str,
    n_contexts: int,
    embedding_dim: int,
    words_per_sec: float | None = None,
) -> None:
    """Log embedding extraction progress."""
    try:
        import wandb

        if wandb.run is not None:
            metrics = {
                "extraction/step": step,
                "extraction/progress": step / total,
                "extraction/n_contexts": n_contexts,
                "extraction/embedding_dim": embedding_dim,
            }
            if words_per_sec is not None:
                metrics["extraction/words_per_sec"] = words_per_sec
            wandb.log(metrics)
    except ImportError:
        pass


def log_training_step(
    epoch: int,
    step: int,
    train_loss: float,
    learning_rate: float | None = None,
) -> None:
    """Log a training step."""
    try:
        import wandb

        if wandb.run is not None:
            metrics = {
                "train/epoch": epoch,
                "train/step": step,
                "train/loss": train_loss,
            }
            if learning_rate is not None:
                metrics["train/learning_rate"] = learning_rate
            wandb.log(metrics)
    except ImportError:
        pass


def log_epoch_metrics(
    epoch: int,
    train_loss: float,
    val_loss: float,
    is_best: bool = False,
) -> None:
    """Log epoch-level metrics."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "val/is_best": is_best,
                }
            )
    except ImportError:
        pass


def log_evaluation_results(results: dict[str, dict]) -> None:
    """
    Log evaluation results (word similarity, STS, etc.).

    Args:
        results: Dictionary mapping dataset names to metrics
    """
    try:
        import wandb

        if wandb.run is not None:
            for dataset, metrics in results.items():
                if "error" in metrics:
                    continue

                prefix = f"eval/{dataset}"
                log_metrics = {}

                if "spearman" in metrics:
                    log_metrics[f"{prefix}/spearman"] = metrics["spearman"]
                if "pearson" in metrics:
                    log_metrics[f"{prefix}/pearson"] = metrics["pearson"]
                if "coverage" in metrics:
                    log_metrics[f"{prefix}/coverage"] = metrics["coverage"]
                if "n_pairs" in metrics:
                    log_metrics[f"{prefix}/n_pairs"] = metrics["n_pairs"]

                wandb.log(log_metrics)
    except ImportError:
        pass


def log_embedding_stats(
    word2vec: dict[str, np.ndarray],
    prefix: str = "embeddings",
) -> None:
    """
    Log embedding statistics.

    Args:
        word2vec: Word to embedding dictionary
        prefix: Metric prefix
    """
    try:
        import wandb

        if wandb.run is not None:
            embeddings = np.array(list(word2vec.values()))
            norms = np.linalg.norm(embeddings, axis=1)

            wandb.log(
                {
                    f"{prefix}/vocab_size": len(word2vec),
                    f"{prefix}/dim": embeddings.shape[1],
                    f"{prefix}/mean_norm": float(np.mean(norms)),
                    f"{prefix}/std_norm": float(np.std(norms)),
                    f"{prefix}/min_norm": float(np.min(norms)),
                    f"{prefix}/max_norm": float(np.max(norms)),
                }
            )
    except ImportError:
        pass


def log_artifact(
    path: Path | str,
    name: str,
    artifact_type: str = "embeddings",
    metadata: dict | None = None,
) -> None:
    """
    Log a file or directory as a wandb artifact.

    Args:
        path: Path to file or directory
        name: Artifact name
        artifact_type: Type of artifact (embeddings, model, data)
        metadata: Additional metadata
    """
    try:
        import wandb

        if wandb.run is not None:
            artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
            path = Path(path)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            wandb.log_artifact(artifact)
    except ImportError:
        pass


def finish_wandb() -> None:
    """Finish the current wandb run."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.finish()
    except ImportError:
        pass
