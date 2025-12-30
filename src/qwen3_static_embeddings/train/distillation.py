"""Knowledge distillation training for static embeddings.

This module implements the knowledge distillation approach from SWE4Semantics:
- Student: Trainable static word embeddings
- Teacher: Frozen Qwen3 sentence encoder
- Loss: Match sentence-level similarity distributions
"""

import copy
import json
import string
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Punctuation to ignore
PUNCTUATION = set(
    list(string.punctuation) + ["。", "、", "？", "！", "「", "」", "（", "）", "：", "・", "，"]
)


@dataclass
class TrainConfig:
    """Configuration for distillation training."""

    epochs: int = 15
    batch_size: int = 128
    learning_rate: float = 1e-4
    temperature: float = 0.05
    early_stop_patience: int = 5
    val_size: int = 10_000
    grad_clip: float = 5.0
    checkpoint_every: int = 1  # Save checkpoint every N epochs


class TeacherEncoder(nn.Module):
    """
    Teacher encoder using HuggingFace transformers.

    Wraps a pretrained model with mean pooling for sentence embeddings.
    This is the idiomatic HF pattern for sentence encoding.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        trust_remote_code: bool = True,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(device)
        self.model.eval()
        self.device = device

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode(self, sentences: list[str]) -> torch.Tensor:
        """
        Encode sentences to embeddings using mean pooling.

        Args:
            sentences: List of sentences to encode

        Returns:
            Tensor of shape (batch_size, hidden_dim)
        """
        # Tokenize with padding
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass
        outputs = self.model(**inputs)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Mean pooling over tokens (excluding padding)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)  # (batch, seq_len, 1)
        masked_hidden = hidden_states * attention_mask
        sum_hidden = masked_hidden.sum(dim=1)  # (batch, hidden)
        sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
        embeddings = sum_hidden / sum_mask  # (batch, hidden)

        return embeddings


class StaticEmbeddingModel(nn.Module):
    """
    Trainable static embedding model for knowledge distillation.

    This model:
    1. Stores word embeddings in a trainable nn.Embedding
    2. Encodes sentences by averaging word embeddings
    3. Supports subword fallback for OOV words
    """

    def __init__(
        self,
        word2vec: dict[str, np.ndarray],
        word_tokenizer_name: str = "bert-base-uncased",
        model_tokenizer_name: str | None = None,
    ):
        super().__init__()

        # Get embedding dimension
        self.dim = len(next(iter(word2vec.values())))

        # Build vocabulary mapping
        self.vocab2id = {w: idx + 1 for idx, w in enumerate(word2vec.keys())}
        self.id2vocab = {v: k for k, v in self.vocab2id.items()}

        # Trainable embedding layer (index 0 is padding)
        self.emb = nn.Embedding(len(word2vec) + 1, self.dim, padding_idx=0)

        # Initialize with pre-trained embeddings
        with torch.no_grad():
            for word, vec in word2vec.items():
                idx = self.vocab2id[word]
                self.emb.weight.data[idx] = torch.FloatTensor(vec)

        # Tokenizers
        word_tok = AutoTokenizer.from_pretrained(word_tokenizer_name)
        self.word_tokenize = word_tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str

        self.model_tokenizer = None
        if model_tokenizer_name:
            self.model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)

    def encode(self, sentences: list[str]) -> torch.Tensor:
        """
        Encode sentences to embeddings.

        Args:
            sentences: List of sentences to encode

        Returns:
            Tensor of shape (batch_size, dim)
        """
        embeddings = []

        for sent in sentences:
            # Tokenize to words
            word_spans = self.word_tokenize(sent)
            words = [w for w, _ in word_spans]

            sent_embs = []
            for word in words:
                if word in PUNCTUATION:
                    continue

                # Try exact match
                vec = self._get_word_embedding(word)
                if vec is None:
                    vec = self._get_word_embedding(word.lower())

                # Try subword fallback
                if vec is None and self.model_tokenizer is not None:
                    vec = self._subword_lookup(word)

                if vec is not None:
                    sent_embs.append(vec)

            if sent_embs:
                sent_emb = torch.stack(sent_embs).sum(dim=0)
            else:
                sent_emb = torch.zeros(self.dim, device=self.emb.weight.device)

            embeddings.append(sent_emb)

        return torch.stack(embeddings)

    def _get_word_embedding(self, word: str) -> torch.Tensor | None:
        """Get embedding for a word if in vocabulary."""
        if word in self.vocab2id:
            idx = self.vocab2id[word]
            return self.emb.weight[idx]
        return None

    def _subword_lookup(self, word: str) -> torch.Tensor | None:
        """Try to find embedding via subword matching."""
        if self.model_tokenizer is None:
            return None

        subwords = self.model_tokenizer.tokenize(word)

        if not subwords:
            return None

        while len(subwords) > 1:
            subwords = subwords[:-1]
            subword_text = "".join(subwords).replace("##", "").replace("▁", "")

            vec = self._get_word_embedding(subword_text)
            if vec is None:
                vec = self._get_word_embedding(subword_text.lower())

            if vec is not None:
                return vec

        return None

    def forward(self, sentences: list[str]) -> torch.Tensor:
        """Forward pass - alias for encode."""
        return self.encode(sentences)

    def get_embeddings(self) -> dict[str, np.ndarray]:
        """Extract current embeddings as numpy arrays."""
        embeddings = {}
        for word, idx in self.vocab2id.items():
            embeddings[word] = self.emb.weight[idx].detach().cpu().numpy()
        return embeddings


def distillation_loss(
    student_sim: torch.Tensor,
    teacher_sim: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Compute distillation loss between student and teacher similarities.

    Uses KL divergence between softmax distributions of similarity matrices.

    Args:
        student_sim: Student similarity matrix (batch, batch)
        teacher_sim: Teacher similarity matrix (batch, batch)
        temperature: Temperature for softmax

    Returns:
        Scalar loss value
    """
    # Apply temperature scaling
    student_logits = student_sim / temperature
    teacher_logits = teacher_sim / temperature

    # Softmax distributions
    student_probs = F.log_softmax(student_logits, dim=-1)
    teacher_probs = F.softmax(teacher_logits, dim=-1)

    # KL divergence (use nansum to handle masked -inf positions)
    kl_div = teacher_probs * student_probs
    loss = -(kl_div.nansum() / teacher_probs.nansum()).mean()
    return loss


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    student: StaticEmbeddingModel,
    optimizer: optim.Optimizer,
    best_val_loss: float,
    config: TrainConfig,
    is_best: bool = False,
) -> Path:
    """Save training checkpoint for resumability."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "student_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": asdict(config),
        "vocab2id": student.vocab2id,
        "dim": student.dim,
    }

    # Save latest checkpoint
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    # Save epoch checkpoint
    epoch_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, epoch_path)

    # Save best model separately
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"  Saved best model to {best_path}")

    return latest_path


def load_checkpoint(
    checkpoint_path: Path,
    word2vec: dict[str, np.ndarray],
    device: str = "cuda",
) -> tuple[StaticEmbeddingModel, optim.Optimizer, int, float, TrainConfig]:
    """Load training checkpoint to resume training."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Rebuild config
    config = TrainConfig(**checkpoint["config"])

    # Rebuild student model
    student = StaticEmbeddingModel(word2vec)
    student.load_state_dict(checkpoint["student_state_dict"])
    student.to(device)

    # Rebuild optimizer
    optimizer = optim.Adam(student.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]

    print(f"Resumed from epoch {epoch}, best_val_loss={best_val_loss:.4f}")

    return student, optimizer, epoch, best_val_loss, config


def train_distillation(
    word2vec: dict[str, np.ndarray],
    train_sentences: list[str],
    val_sentences: list[str],
    teacher_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    config: TrainConfig | None = None,
    device: str = "cuda",
    show_progress: bool = True,
    checkpoint_dir: Path | str | None = None,
    resume_from: Path | str | None = None,
) -> dict[str, np.ndarray]:
    """
    Train static embeddings via knowledge distillation.

    Args:
        word2vec: Initial word embeddings
        train_sentences: Training sentences
        val_sentences: Validation sentences
        teacher_model_name: HuggingFace model for teacher
        config: Training configuration
        device: Device to train on
        show_progress: Whether to show progress bars
        checkpoint_dir: Directory to save checkpoints (None = no checkpointing)
        resume_from: Path to checkpoint to resume from

    Returns:
        Fine-tuned word embeddings
    """
    if config is None:
        config = TrainConfig()

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Save config
        with open(checkpoint_dir / "config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)

    print(f"Training distillation with {len(train_sentences)} sentences")
    print(f"Teacher model: {teacher_model_name}")

    # Initialize or resume
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_from is not None:
        student, optimizer, start_epoch, best_val_loss, config = load_checkpoint(
            Path(resume_from), word2vec, device
        )
        start_epoch += 1  # Start from next epoch
    else:
        student = StaticEmbeddingModel(word2vec)
        student.to(device)
        optimizer = optim.Adam(student.parameters(), lr=config.learning_rate)

    # Load teacher model
    print("Loading teacher model...")
    teacher = TeacherEncoder(teacher_model_name, device=device)

    # Training loop
    best_model = None
    no_improvement = 0

    for epoch in range(start_epoch, config.epochs):
        print(f"\n=== Epoch {epoch + 1}/{config.epochs} ===")

        # Shuffle training data
        train_idx = np.random.permutation(len(train_sentences))
        train_sentences_shuffled = [train_sentences[i] for i in train_idx]

        # Training
        student.train()
        train_loss = 0.0
        n_batches = 0

        iterator = range(0, len(train_sentences_shuffled), config.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Training")

        for i in iterator:
            batch = train_sentences_shuffled[i : i + config.batch_size]
            if len(batch) < 2:
                continue

            optimizer.zero_grad()

            # Teacher embeddings (already frozen via TeacherEncoder)
            teacher_embs = teacher.encode(batch)
            teacher_embs = F.normalize(teacher_embs, dim=-1)
            teacher_sim = torch.matmul(teacher_embs, teacher_embs.T)

            # Mask diagonal
            mask = torch.eye(len(batch), device=device).bool()
            teacher_sim = teacher_sim.masked_fill(mask, float("-inf"))

            # Student embeddings
            student_embs = student(batch)
            student_embs = F.normalize(student_embs, dim=-1)
            student_sim = torch.matmul(student_embs, student_embs.T)
            student_sim = student_sim.masked_fill(mask, float("-inf"))

            # Loss
            loss = distillation_loss(student_sim, teacher_sim, config.temperature)
            loss.backward()

            nn.utils.clip_grad_norm_(student.parameters(), config.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)
        print(f"Train loss: {avg_train_loss:.4f}")

        # Validation
        student.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            val_iterator = range(0, len(val_sentences), config.batch_size)
            if show_progress:
                val_iterator = tqdm(val_iterator, desc="Validation")

            for i in val_iterator:
                batch = val_sentences[i : i + config.batch_size]
                if len(batch) < 2:
                    continue

                # Teacher
                teacher_embs = teacher.encode(batch)
                teacher_embs = F.normalize(teacher_embs, dim=-1)
                teacher_sim = torch.matmul(teacher_embs, teacher_embs.T)

                mask = torch.eye(len(batch), device=device).bool()
                teacher_sim = teacher_sim.masked_fill(mask, float("-inf"))

                # Student
                student_embs = student(batch)
                student_embs = F.normalize(student_embs, dim=-1)
                student_sim = torch.matmul(student_embs, student_embs.T)
                student_sim = student_sim.masked_fill(mask, float("-inf"))

                loss = distillation_loss(student_sim, teacher_sim, config.temperature)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        print(f"Val loss: {avg_val_loss:.4f}")

        # Check for improvement
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(student.cpu())
            student.to(device)
            no_improvement = 0
            print("New best model!")
        else:
            no_improvement += 1

        # Save checkpoint
        if checkpoint_dir is not None and (epoch + 1) % config.checkpoint_every == 0:
            save_checkpoint(
                checkpoint_dir,
                epoch,
                student,
                optimizer,
                best_val_loss,
                config,
                is_best=is_best,
            )

        # Early stopping
        if no_improvement >= config.early_stop_patience:
            print("Early stopping triggered")
            break

    # Return best embeddings
    if best_model is not None:
        return best_model.get_embeddings()
    else:
        return student.cpu().get_embeddings()
