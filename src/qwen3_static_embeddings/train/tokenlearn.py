"""Tokenlearn: Train static embeddings using sentence transformer distillation.

This implements the Tokenlearn training approach from MinishLab:
1. Featurize: Generate mean embeddings from teacher model on large corpus
2. Train: Minimize cosine distance between static and teacher embeddings
3. Post-process: Apply PCA and SIF weighting

Reference: https://github.com/MinishLab/tokenlearn
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class TokenlearnConfig:
    """Configuration for Tokenlearn training."""

    # Teacher model
    teacher_model: str = "Qwen/Qwen3-Embedding-0.6B"

    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    epochs: int = 10
    warmup_steps: int = 1000

    # Dimensions
    pca_dims: int = 256

    # SIF weighting
    sif_coefficient: float = 1e-4

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # NVIDIA optimizations
    use_bf16: bool = True  # Use bfloat16 for better precision than fp16
    compile_model: bool = False  # torch.compile for faster training
    num_workers: int = 4  # DataLoader workers

    # Checkpointing
    checkpoint_every: int = 1  # Save checkpoint every N epochs
    resume_from: str | None = None  # Path to checkpoint to resume from

    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "tokenlearn"
    wandb_run_name: str | None = None

    # Train-time evaluation
    eval_every: int = 1  # Evaluate every N epochs (0 to disable)
    eval_benchmarks: tuple[str, ...] = ("simlex", "wordsim")  # Benchmarks to run


def setup_nvidia_optimizations():
    """Setup optimizations for NVIDIA GPUs.

    Supports latest architectures including:
    - Blackwell (GB200, GB300, RTX 50xx) - Compute Capability 10.0+
    - Hopper (H100, H200) - Compute Capability 9.0
    - Ada Lovelace (RTX 40xx, L40) - Compute Capability 8.9
    - Ampere (A100, RTX 30xx) - Compute Capability 8.0-8.6
    """
    if not torch.cuda.is_available():
        return

    # Enable TF32 for faster matmuls on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Enable cudnn benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True

    # Print GPU info
    device_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    memory_gb = props.total_memory / 1e9
    compute_cap = f"{props.major}.{props.minor}"

    # Identify architecture
    arch_name = "Unknown"
    if props.major >= 10:
        arch_name = "Blackwell"
    elif props.major == 9:
        arch_name = "Hopper"
    elif props.major == 8 and props.minor >= 9:
        arch_name = "Ada Lovelace"
    elif props.major == 8:
        arch_name = "Ampere"

    print(f"GPU: {device_name} ({memory_gb:.1f} GB)")
    print(f"  Architecture: {arch_name} (Compute Capability {compute_cap})")

    # Blackwell/Hopper specific optimizations
    if props.major >= 9:
        print("  Enabled: TF32, BFloat16, Flash Attention 2, FP8 (if supported)")
    else:
        print("  Enabled: TF32, BFloat16, Flash Attention 2")

    # Check for multi-GPU
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1:
        print(f"Found {n_gpus} GPUs - consider using DataParallel or DistributedDataParallel")


# Cache for eval datasets (loaded once per process)
_eval_cache: dict[str, list[tuple[str, str, float]]] = {}


def _load_eval_dataset(name: str) -> list[tuple[str, str, float]]:
    """Load evaluation dataset with caching."""
    if name in _eval_cache:
        return _eval_cache[name]

    from qwen3_static_embeddings.evaluate import (
        load_simlex999,
        load_stsb,
        load_wordsim353,
    )

    if name == "simlex":
        pairs = load_simlex999()
    elif name == "wordsim":
        pairs = load_wordsim353()
    elif name == "stsb":
        pairs = load_stsb("validation")  # Use validation for train-time eval
    else:
        pairs = []

    _eval_cache[name] = pairs
    return pairs


def evaluate_during_training(
    model: "StaticModelForTraining",
    benchmarks: tuple[str, ...] = ("simlex", "wordsim"),
) -> dict[str, float]:
    """
    Quick evaluation during training on word similarity benchmarks.

    Args:
        model: StaticModelForTraining instance
        benchmarks: Tuple of benchmark names ("simlex", "wordsim", "stsb")

    Returns:
        Dict mapping benchmark names to Spearman rho scores
    """
    from scipy import stats

    model.eval()
    results = {}

    for bench_name in benchmarks:
        pairs = _load_eval_dataset(bench_name)
        if not pairs:
            continue

        predictions = []
        gold_scores = []

        with torch.no_grad():
            for text1, text2, gold in pairs:
                try:
                    # Encode both texts
                    emb1 = model.encode([text1])[0]
                    emb2 = model.encode([text2])[0]

                    # Cosine similarity (embeddings are already normalized)
                    sim = float(torch.dot(emb1, emb2).cpu())
                    predictions.append(sim)
                    gold_scores.append(gold)
                except Exception:
                    continue

        if len(predictions) >= 2:
            spearman_rho, _ = stats.spearmanr(predictions, gold_scores)
            results[bench_name] = spearman_rho

    model.train()
    return results


class FeatureDataset(Dataset[tuple[str, np.ndarray]]):
    """Dataset of (text, embedding) pairs for Tokenlearn training."""

    def __init__(self, texts: list[str], embeddings: np.ndarray):
        self.texts = texts
        self.embeddings = embeddings

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray]:  # type: ignore[override]
        return self.texts[idx], self.embeddings[idx]


def featurize_corpus(
    corpus_path: Path,
    output_dir: Path,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    max_sentences: int = 1_000_000,
    batch_size: int = 32,
    max_length: int = 512,
    save_every: int = 10_000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    """
    Generate mean embeddings from teacher model for corpus sentences.

    Args:
        corpus_path: Path to corpus file (one sentence per line)
        output_dir: Directory to save features
        model_name: Teacher model name
        max_sentences: Maximum sentences to process
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        save_every: Save checkpoint every N sentences
        device: Device to use
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading teacher model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load corpus
    print(f"Loading corpus from {corpus_path}")
    with open(corpus_path) as f:
        sentences = [line.strip() for line in f if line.strip()]

    sentences = sentences[:max_sentences]
    print(f"Processing {len(sentences)} sentences")

    all_texts = []
    all_embeddings = []
    batch_idx = 0

    for i in tqdm(range(0, len(sentences), batch_size), desc="Featurizing"):
        batch_texts = sentences[i : i + batch_size]

        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        # Get token embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use last hidden state
            hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)

            # Get attention mask for proper averaging
            attention_mask = inputs["attention_mask"]

            # Mean pool (excluding padding)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask

            # Normalize
            mean_embeddings = F.normalize(mean_embeddings, p=2, dim=-1)
            mean_embeddings = mean_embeddings.cpu().numpy().astype(np.float32)

        all_texts.extend(batch_texts)
        all_embeddings.append(mean_embeddings)

        # Save checkpoint
        if len(all_texts) >= save_every and len(all_texts) % save_every < batch_size:
            batch_idx += 1
            _save_batch(output_dir, batch_idx, all_texts, all_embeddings)
            all_texts = []
            all_embeddings = []

    # Save final batch
    if all_texts:
        batch_idx += 1
        _save_batch(output_dir, batch_idx, all_texts, all_embeddings)

    print(f"Featurization complete. Saved to {output_dir}")


def featurize_c4(
    output_dir: Path,
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    max_sentences: int = 1_000_000,
    batch_size: int = 32,
    max_length: int = 512,
    save_every: int = 10_000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    use_flash_attention: bool = True,
) -> None:
    """
    Generate mean embeddings from teacher model using C4 dataset.

    This mirrors the original Tokenlearn approach which uses C4.

    Args:
        output_dir: Directory to save features
        model_name: Teacher model name
        max_sentences: Maximum sentences to process
        batch_size: Batch size for encoding
        max_length: Maximum sequence length
        save_every: Save checkpoint every N sentences
        device: Device to use
        use_flash_attention: Use Flash Attention 2 if available
    """
    from datasets import load_dataset
    from transformers import AutoModel, AutoTokenizer

    # Setup NVIDIA optimizations
    if device == "cuda":
        setup_nvidia_optimizations()

    print(f"Loading teacher model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Use bfloat16 on modern GPUs (better than float16)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Try to use Flash Attention 2
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }
    if use_flash_attention and device == "cuda":
        try:
            model_kwargs["attn_implementation"] = "flash_attention_2"
            print("Using Flash Attention 2")
        except Exception:
            pass

    model = AutoModel.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load C4 dataset (streaming to avoid downloading full dataset)
    print("Loading C4 dataset from HuggingFace (streaming)...")
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)

    all_texts = []
    all_embeddings = []
    batch_idx = 0
    n_processed = 0

    batch_texts = []

    print(f"Processing up to {max_sentences} sentences from C4...")
    for example in tqdm(dataset, total=max_sentences, desc="Featurizing C4"):
        text = example["text"].strip()
        if not text or len(text) < 20:  # Skip very short texts
            continue

        # Truncate very long texts
        if len(text) > 2000:
            text = text[:2000]

        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            # Process batch
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                hidden_states = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]

                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                mean_embeddings = sum_embeddings / sum_mask
                mean_embeddings = F.normalize(mean_embeddings, p=2, dim=-1)
                mean_embeddings = mean_embeddings.cpu().numpy().astype(np.float32)

            all_texts.extend(batch_texts)
            all_embeddings.append(mean_embeddings)
            n_processed += len(batch_texts)
            batch_texts = []

            # Save checkpoint
            if len(all_texts) >= save_every:
                batch_idx += 1
                _save_batch(output_dir, batch_idx, all_texts, all_embeddings)
                all_texts = []
                all_embeddings = []

        if n_processed >= max_sentences:
            break

    # Process remaining batch
    if batch_texts:
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]

            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            mean_embeddings = sum_embeddings / sum_mask
            mean_embeddings = F.normalize(mean_embeddings, p=2, dim=-1)
            mean_embeddings = mean_embeddings.cpu().numpy().astype(np.float32)

        all_texts.extend(batch_texts)
        all_embeddings.append(mean_embeddings)

    # Save final batch
    if all_texts:
        batch_idx += 1
        _save_batch(output_dir, batch_idx, all_texts, all_embeddings)

    print(f"Featurization complete. Processed {n_processed} sentences. Saved to {output_dir}")


def _save_batch(
    output_dir: Path,
    batch_idx: int,
    texts: list[str],
    embeddings: list[np.ndarray],
) -> None:
    """Save a batch of features to disk."""
    # Concatenate embeddings
    embeddings_array = np.concatenate(embeddings, axis=0)

    # Save
    text_path = output_dir / f"texts_{batch_idx:04d}.json"
    emb_path = output_dir / f"embeddings_{batch_idx:04d}.npy"

    with open(text_path, "w") as f:
        json.dump(texts, f)
    np.save(emb_path, embeddings_array)

    print(f"Saved batch {batch_idx}: {len(texts)} samples")


def load_features(feature_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load all features from a directory."""
    feature_dir = Path(feature_dir)

    all_texts = []
    all_embeddings = []

    # Find all batch files
    text_files = sorted(feature_dir.glob("texts_*.json"))

    for text_file in text_files:
        batch_idx = text_file.stem.split("_")[1]
        emb_file = feature_dir / f"embeddings_{batch_idx}.npy"

        with open(text_file) as f:
            texts = json.load(f)
        embeddings = np.load(emb_file)

        all_texts.extend(texts)
        all_embeddings.append(embeddings)

    return all_texts, np.concatenate(all_embeddings, axis=0)


class StaticModelForTraining(nn.Module):
    """Wrapper for training static embeddings with gradient descent."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        output_dim: int,
        token_to_idx: dict[str, int],
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.projection = nn.Linear(embedding_dim, output_dim)
        self.token_to_idx = token_to_idx
        self.tokenizer = None  # Set after initialization

    @classmethod
    def from_model2vec(cls, model_path: str, output_dim: int = 256):
        """Initialize from a distilled Model2Vec model."""
        from model2vec import StaticModel

        static_model = StaticModel.from_pretrained(model_path)

        # Get vocabulary
        vocab = static_model.tokenizer.get_vocab()
        vocab_size = len(vocab)
        embedding_dim = static_model.dim

        # Create trainable model
        model = cls(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            token_to_idx=vocab,
        )

        # Initialize embeddings from static model
        with torch.no_grad():
            embeddings = static_model.embedding
            model.embeddings.weight.copy_(torch.tensor(embeddings))

        model.tokenizer = static_model.tokenizer

        return model

    def encode(self, texts: list[str]) -> torch.Tensor:
        """Encode texts to embeddings (mean of token embeddings)."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not set. Use from_model2vec() to create the model.")

        batch_embeddings = []

        for text in texts:
            # Tokenize
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids = tokens.ids

            if len(token_ids) == 0:
                # Empty text, return zeros
                batch_embeddings.append(
                    torch.zeros(self.projection.out_features, device=self.embeddings.weight.device)
                )
                continue

            # Get embeddings
            token_ids_tensor = torch.tensor(token_ids, device=self.embeddings.weight.device)
            token_embs = self.embeddings(token_ids_tensor)

            # Mean pool
            mean_emb = token_embs.mean(dim=0)

            # Project
            output = self.projection(mean_emb)
            output = F.normalize(output, p=2, dim=-1)

            batch_embeddings.append(output)

        return torch.stack(batch_embeddings)

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.encode(texts)


def train_tokenlearn(
    model: StaticModelForTraining,
    texts: list[str],
    target_embeddings: np.ndarray,
    config: TokenlearnConfig,
    output_path: Path,
) -> StaticModelForTraining:
    """
    Train static model using Tokenlearn approach.

    Args:
        model: Trainable static model
        texts: Training texts
        target_embeddings: Target embeddings from teacher
        config: Training configuration
        output_path: Path to save trained model

    Returns:
        Trained model
    """
    # Setup NVIDIA optimizations
    if config.device == "cuda":
        setup_nvidia_optimizations()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    if config.use_wandb:
        import wandb

        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config={
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "epochs": config.epochs,
                "pca_dims": config.pca_dims,
                "use_bf16": config.use_bf16,
                "compile_model": config.compile_model,
            },
        )

    # Apply PCA to targets
    n_components = min(config.pca_dims, len(texts), target_embeddings.shape[1])
    print(f"Applying PCA to reduce targets to {n_components} dimensions")
    pca = PCA(n_components=n_components)
    target_embeddings = pca.fit_transform(target_embeddings)
    target_embeddings = target_embeddings / (
        np.linalg.norm(target_embeddings, axis=1, keepdims=True) + 1e-8
    )

    # Reinitialize projection layer to match target dimension
    embedding_dim = model.embeddings.weight.shape[1]
    model.projection = nn.Linear(embedding_dim, n_components)

    # Create dataset
    dataset = FeatureDataset(texts, target_embeddings)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device == "cuda"),
    )

    # Setup training
    model = model.to(config.device)

    # Optional: torch.compile for faster training (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, "compile"):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)  # type: ignore[assignment]

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader) * config.epochs
    )

    # Setup AMP for mixed precision training
    use_amp = config.device == "cuda" and config.use_bf16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if config.use_bf16 else torch.float16

    if use_amp:
        print(f"Using automatic mixed precision (AMP) with {amp_dtype}")

    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    if config.resume_from:
        checkpoint_path = Path(config.resume_from) / "checkpoint.pt"
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint["global_step"]
            best_loss = checkpoint.get("best_loss", float("inf"))
            print(f"  Resuming from epoch {start_epoch}, step {global_step}")

    # Training loop
    model.train()

    for epoch in range(start_epoch, config.epochs):
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for batch_texts, batch_targets in pbar:
            batch_targets = batch_targets.to(config.device)

            # Forward with AMP
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                outputs = model(list(batch_texts))

                # Cosine loss (1 - cosine_similarity)
                cos_sim = F.cosine_similarity(outputs, batch_targets, dim=-1)
                loss = (1 - cos_sim).mean()

                # Add L2 regularization on embeddings
                l2_reg = 1e-5 * model.embeddings.weight.pow(2).mean()
                loss = loss + l2_reg

            # Backward with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Log to wandb
            if config.use_wandb and global_step % 100 == 0:
                import wandb

                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    }
                )

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

        # Log epoch metrics to wandb
        if config.use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch/avg_loss": avg_loss,
                    "epoch/epoch": epoch + 1,
                }
            )

        # Train-time evaluation
        eval_results = {}
        if config.eval_every > 0 and (epoch + 1) % config.eval_every == 0:
            print("  Running evaluation...")
            eval_results = evaluate_during_training(model, config.eval_benchmarks)
            for bench_name, score in eval_results.items():
                print(f"    {bench_name}: ρ = {score:.4f}")

            # Log eval metrics to wandb
            if config.use_wandb:
                import wandb

                wandb.log(
                    {f"eval/{bench_name}": score for bench_name, score in eval_results.items()}
                )

        # Save checkpoint every N epochs
        if (epoch + 1) % config.checkpoint_every == 0 or (epoch + 1) == config.epochs:
            checkpoint_data = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_loss": min(best_loss, avg_loss),
                "eval_results": eval_results,
                "config": {
                    "batch_size": config.batch_size,
                    "learning_rate": config.learning_rate,
                    "epochs": config.epochs,
                    "pca_dims": config.pca_dims,
                },
            }
            torch.save(checkpoint_data, output_path / "checkpoint.pt")
            print(f"  Saved checkpoint at epoch {epoch + 1}")

            # Save best model (based on loss)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(checkpoint_data, output_path / "checkpoint_best.pt")
                print(f"  New best model (loss={avg_loss:.4f})")

    # Save final model
    torch.save(
        {
            "embeddings": model.embeddings.weight.detach().cpu(),
            "projection": model.projection.state_dict(),
            "token_to_idx": model.token_to_idx,
            "pca_components": pca.components_,
            "pca_mean": pca.mean_,
        },
        output_path / "tokenlearn_model.pt",
    )

    print(f"Saved trained model to {output_path}")

    # Finish wandb run
    if config.use_wandb:
        import wandb

        wandb.finish()

    return model


def collect_token_frequencies(
    texts: list[str],
    tokenizer,
) -> dict[int, int]:
    """
    Collect token frequencies from corpus texts.

    Args:
        texts: List of texts
        tokenizer: Tokenizer to use

    Returns:
        Mapping from token index to frequency count
    """
    from collections import Counter

    token_counts: Counter = Counter()

    for text in tqdm(texts, desc="Collecting token frequencies"):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_counts.update(tokens.ids)

    return dict(token_counts)


def apply_sif_weighting(
    embeddings: np.ndarray,
    token_frequencies: dict[int, int],
    sif_coefficient: float = 1e-3,
) -> np.ndarray:
    """
    Apply Smooth Inverse Frequency (SIF) weighting to embeddings.

    This reweights embeddings based on token frequency in the corpus:
    weight = a / (a + prob) where prob = count / total

    Reference: https://openreview.net/pdf?id=SyK00v5xx (SIF paper)

    Args:
        embeddings: Token embeddings (vocab_size, dim)
        token_frequencies: Mapping from token index to frequency count
        sif_coefficient: SIF coefficient 'a' (default 1e-3, as in POTION)

    Returns:
        Weighted embeddings
    """
    total_freq = sum(token_frequencies.values())
    vocab_size = embeddings.shape[0]

    # Compute weights for each token
    weights = np.ones(vocab_size, dtype=np.float32)

    for token_idx, count in token_frequencies.items():
        if token_idx < vocab_size:
            prob = count / total_freq
            weights[token_idx] = sif_coefficient / (sif_coefficient + prob)

    # Apply weights to embeddings
    weighted = embeddings * weights[:, np.newaxis]

    return weighted


def remove_principal_component(
    embeddings: np.ndarray,
    n_components: int = 1,
) -> np.ndarray:
    """
    Remove first N principal components from embeddings (All-but-the-Top).

    This removes the common component that doesn't carry semantic information.

    Args:
        embeddings: Token embeddings (vocab_size, dim)
        n_components: Number of principal components to remove

    Returns:
        Embeddings with principal components removed
    """
    # Center embeddings
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean

    # Compute principal components
    pca = PCA(n_components=n_components)
    pca.fit(centered)

    # Remove principal components
    for i in range(n_components):
        pc = pca.components_[i]
        projection = np.outer(centered @ pc, pc)
        centered = centered - projection

    return centered + mean


def post_process_embeddings(
    embeddings: np.ndarray,
    token_frequencies: dict[int, int] | None = None,
    apply_sif: bool = True,
    apply_pca: bool = True,
    pca_dims: int | None = None,
    remove_pc: bool = True,
    n_pc_remove: int = 1,
    sif_coefficient: float = 1e-3,
) -> np.ndarray:
    """
    Apply post-training re-regularization to embeddings.

    This implements the POTION post-processing steps:
    1. Apply SIF weighting based on token frequencies
    2. Optionally apply PCA dimensionality reduction
    3. Remove first principal component (All-but-the-Top)
    4. L2 normalize

    Args:
        embeddings: Token embeddings (vocab_size, dim)
        token_frequencies: Mapping from token index to frequency count
        apply_sif: Whether to apply SIF weighting
        apply_pca: Whether to apply PCA
        pca_dims: Target dimensions for PCA (None = no reduction)
        remove_pc: Whether to remove first principal component
        n_pc_remove: Number of principal components to remove
        sif_coefficient: SIF coefficient

    Returns:
        Post-processed embeddings
    """
    result = embeddings.copy()

    # Step 1: SIF weighting
    if apply_sif and token_frequencies is not None:
        print("Applying SIF weighting...")
        result = apply_sif_weighting(result, token_frequencies, sif_coefficient)

    # Step 2: PCA dimensionality reduction
    if apply_pca and pca_dims is not None and pca_dims < result.shape[1]:
        print(f"Applying PCA: {result.shape[1]} -> {pca_dims} dimensions")
        pca = PCA(n_components=pca_dims)
        result = pca.fit_transform(result)

    # Step 3: Remove principal components (All-but-the-Top)
    if remove_pc:
        print(f"Removing {n_pc_remove} principal component(s)...")
        result = remove_principal_component(result, n_pc_remove)

    # Step 4: L2 normalize
    print("L2 normalizing...")
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / (norms + 1e-8)

    return result
