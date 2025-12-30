"""Main embedding extraction pipeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from qwen3_static_embeddings.config import ModelConfig
from qwen3_static_embeddings.data.word2sent import Word2Sent
from qwen3_static_embeddings.extract.model import get_hidden_dim, load_model, load_tokenizer
from qwen3_static_embeddings.extract.tokenization import (
    find_word_positions_batch,
    tokenize_word,
)


@dataclass
class ExtractionResult:
    """Result of embedding extraction for a single word."""

    word: str
    embedding: np.ndarray  # Shape: (hidden_dim,)
    n_contexts: int  # Number of contexts used


class EmbeddingExtractor:
    """
    Extract static word embeddings from Qwen3 contextual model.

    This class implements the core extraction logic:
    1. For each word, get its example sentences
    2. Tokenize and find word positions
    3. Extract hidden states at word positions
    4. Average across contexts to get static embedding
    """

    def __init__(
        self,
        config: ModelConfig | None = None,
        model=None,
        tokenizer=None,
    ):
        """
        Initialize the extractor.

        Args:
            config: Model configuration (uses defaults if None)
            model: Pre-loaded model (loads from config if None)
            tokenizer: Pre-loaded tokenizer (loads from config if None)
        """
        self.config = config or ModelConfig()

        if tokenizer is None:
            print(f"Loading tokenizer: {self.config.name}")
            self.tokenizer = load_tokenizer(
                self.config.name,
                trust_remote_code=self.config.trust_remote_code,
            )
        else:
            self.tokenizer = tokenizer

        if model is None:
            print(f"Loading model: {self.config.name}")
            self.model = load_model(
                self.config.name,
                device=self.config.device,
                dtype=self.config.dtype,
                trust_remote_code=self.config.trust_remote_code,
            )
        else:
            self.model = model

        self.hidden_dim = get_hidden_dim(self.model)
        print(f"Model hidden dimension: {self.hidden_dim}")

    def extract_word_embedding(
        self,
        word: str,
        sentences: list[str],
        n_contexts: int | None = None,
        subword_mode: bool = False,
    ) -> ExtractionResult | None:
        """
        Extract static embedding for a single word.

        Args:
            word: Target word
            sentences: List of sentences containing the word
            n_contexts: Max contexts to use (None for all)
            subword_mode: Whether word is a subword

        Returns:
            ExtractionResult or None if extraction failed
        """
        if not sentences:
            return None

        # Limit number of contexts
        if n_contexts is not None:
            sentences = sentences[:n_contexts]

        # Tokenize the word
        add_space = not subword_mode
        word_token_ids = tokenize_word(word, self.tokenizer, add_space=add_space)

        if len(word_token_ids) == 0:
            return None

        # Tokenize all sentences
        batch = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        batch_input_ids = batch["input_ids"].tolist()

        # Find word positions in each sentence
        positions, valid_sent_ids = find_word_positions_batch(
            batch_input_ids=batch_input_ids,
            word=word,
            word_token_ids=word_token_ids,
            tokenizer=self.tokenizer,
            subword_mode=subword_mode,
        )

        if len(valid_sent_ids) == 0:
            return None

        # Extract embeddings for valid sentences
        embeddings = self._extract_batch_embeddings(
            sentences=[sentences[i] for i in valid_sent_ids],
            positions=positions,
        )

        # Average across contexts and tokens
        # embeddings shape: (n_sentences, n_tokens, hidden_dim)
        embeddings = embeddings.mean(axis=1)  # Average across tokens
        static_embedding = embeddings.mean(axis=0)  # Average across sentences

        return ExtractionResult(
            word=word,
            embedding=static_embedding,
            n_contexts=len(valid_sent_ids),
        )

    def _extract_batch_embeddings(
        self,
        sentences: list[str],
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Extract embeddings for a batch of sentences at specified positions.

        Args:
            sentences: List of sentences
            positions: Array of shape (n_sentences, n_tokens) with positions

        Returns:
            Array of shape (n_sentences, n_tokens, hidden_dim)
        """
        # Sort by length for efficient batching
        sent_lens = [len(s) for s in sentences]
        sorted_indices = np.argsort(sent_lens)[::-1]  # Descending

        sentences_sorted = [sentences[i] for i in sorted_indices]
        positions_sorted = positions[sorted_indices]

        all_embeddings = []
        max_tokens = self.config.batch_size * self.config.max_length

        # Process in batches
        batch_start = 0
        while batch_start < len(sentences_sorted):
            # Determine batch size based on token budget
            batch_end = batch_start
            current_tokens = 0

            while batch_end < len(sentences_sorted):
                est_len = len(sentences_sorted[batch_end].split()) * 2  # Rough estimate
                if current_tokens + est_len > max_tokens and batch_end > batch_start:
                    break
                current_tokens += est_len
                batch_end += 1

            # Extract this batch
            batch_sents = sentences_sorted[batch_start:batch_end]
            batch_pos = positions_sorted[batch_start:batch_end]

            batch_embs = self._forward_batch(batch_sents, batch_pos)
            all_embeddings.append(batch_embs)

            batch_start = batch_end

        # Concatenate and unsort
        all_embeddings = np.concatenate(all_embeddings, axis=0)

        # Restore original order
        unsort_indices = np.argsort(sorted_indices)
        all_embeddings = all_embeddings[unsort_indices]

        return all_embeddings

    def _forward_batch(
        self,
        sentences: list[str],
        positions: np.ndarray,
    ) -> np.ndarray:
        """
        Run forward pass and extract hidden states at positions.

        Args:
            sentences: Batch of sentences
            positions: Shape (batch_size, n_tokens)

        Returns:
            Array of shape (batch_size, n_tokens, hidden_dim)
        """
        # Tokenize
        batch = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(self.config.device) for k, v in batch.items()}

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**batch, output_hidden_states=True)

        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Extract at positions
        batch_size = len(sentences)

        # Create row indices for advanced indexing
        row_idx = np.arange(batch_size)[:, None]  # (batch, 1)

        # Extract embeddings at positions
        extracted = hidden_states[row_idx, positions]  # (batch, n_tokens, hidden)
        extracted = extracted.cpu().numpy()

        return extracted

    def extract_vocabulary(
        self,
        word2sent: Word2Sent,
        output_path: Path,
        n_contexts: int = 100,
        subword_mode: bool = False,
        show_progress: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Extract embeddings for all words in word2sent mapping.

        Args:
            word2sent: Word to sentences mapping
            output_path: Path to save embeddings (word2vec format)
            n_contexts: Number of contexts per word
            subword_mode: Whether to use subword matching
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping words to embeddings
        """
        words = word2sent.words
        embeddings = {}

        # Open output file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count_path = output_path.with_suffix(".count.txt")

        iterator = tqdm(words, desc="Extracting embeddings") if show_progress else words

        with open(output_path, "w") as f_vec, open(count_path, "w") as f_count:
            for word in iterator:
                sentences = word2sent.get_sentences(word, n=n_contexts)

                if not sentences:
                    continue

                result = self.extract_word_embedding(
                    word=word,
                    sentences=sentences,
                    n_contexts=n_contexts,
                    subword_mode=subword_mode,
                )

                if result is None:
                    f_count.write(f"{word} NOT_FOUND\n")
                    continue

                # Store embedding
                embeddings[word] = result.embedding

                # Write to files
                vec_str = " ".join(str(x) for x in result.embedding)
                f_vec.write(f"{word} {vec_str}\n")
                f_count.write(f"{word} {result.n_contexts}\n")

        print(f"Extracted {len(embeddings)} word embeddings")
        print(f"Saved to {output_path}")

        return embeddings
