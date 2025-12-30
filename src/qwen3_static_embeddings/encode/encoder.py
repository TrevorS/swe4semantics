"""Static embedding encoder for fast sentence encoding."""

import string
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from qwen3_static_embeddings.transform.pca import load_embeddings

# Punctuation set for filtering
PUNCTUATION = set(
    list(string.punctuation) + ["。", "、", "？", "！", "「", "」", "（", "）", "：", "・", "，"]
)


class StaticEncoder:
    """
    Encode sentences using static word embeddings.

    This provides fast CPU-based sentence encoding by:
    1. Tokenizing text into words
    2. Looking up word embeddings
    3. Averaging to get sentence embedding

    Example:
        >>> encoder = StaticEncoder.from_pretrained("embeddings.txt")
        >>> emb = encoder.encode("The quick brown fox")
        >>> sim = encoder.similarity("Hello world", "Hi there")
    """

    def __init__(
        self,
        word2vec: dict[str, np.ndarray],
        dim: int,
        word_tokenizer=None,
        model_tokenizer=None,
        normalize: bool = True,
    ):
        """
        Initialize the encoder.

        Args:
            word2vec: Dict mapping words to embeddings
            dim: Embedding dimension
            word_tokenizer: Callable for word tokenization
            model_tokenizer: Tokenizer for subword fallback
            normalize: Whether to L2 normalize sentence embeddings
        """
        self.word2vec = word2vec
        self.dim = dim
        self.normalize = normalize

        # Setup tokenizers
        if word_tokenizer is None:
            bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.word_tokenizer = bert_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
        else:
            self.word_tokenizer = word_tokenizer

        self.model_tokenizer = model_tokenizer

    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        model_tokenizer_name: str | None = None,
        normalize: bool = True,
    ) -> "StaticEncoder":
        """
        Load encoder from pretrained embeddings file.

        Args:
            path: Path to embeddings file (word2vec text format)
            model_tokenizer_name: HuggingFace model name for subword fallback
            normalize: Whether to normalize sentence embeddings

        Returns:
            Initialized StaticEncoder
        """
        word2vec, dim = load_embeddings(Path(path))

        model_tokenizer = None
        if model_tokenizer_name:
            model_tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name)

        return cls(
            word2vec=word2vec,
            dim=dim,
            model_tokenizer=model_tokenizer,
            normalize=normalize,
        )

    def encode(self, text: str) -> np.ndarray:
        """
        Encode a single text into an embedding.

        Args:
            text: Input text

        Returns:
            Embedding vector of shape (dim,)
        """
        return self._encode_single(text)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Embeddings of shape (n_texts, dim)
        """
        embeddings = [self._encode_single(t) for t in texts]
        return np.array(embeddings)

    def similarity(self, text1: str, text2: str) -> float:
        """
        Compute cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity score
        """
        emb1 = self.encode(text1)
        emb2 = self.encode(text2)
        return float(np.dot(emb1, emb2))

    def _encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string."""
        eps = 1e-8

        # Tokenize into words
        word_spans = self.word_tokenizer(text)
        words = [word for word, _ in word_spans]

        # Collect word embeddings
        word_embeddings = []

        for word in words:
            if word in PUNCTUATION:
                continue

            # Try exact match
            vec = self.word2vec.get(word)

            # Try lowercase
            if vec is None:
                vec = self.word2vec.get(word.lower())

            # Try subword fallback
            if vec is None and self.model_tokenizer is not None:
                vec = self._subword_lookup(word)

            if vec is not None:
                word_embeddings.append(vec)

        # Average embeddings
        if len(word_embeddings) > 0:
            sentence_emb = np.mean(word_embeddings, axis=0)
        else:
            sentence_emb = np.zeros(self.dim)

        # Normalize if requested
        if self.normalize:
            norm = np.linalg.norm(sentence_emb)
            sentence_emb = sentence_emb / (norm + eps)

        return sentence_emb

    def _subword_lookup(self, word: str) -> np.ndarray | None:
        """
        Try to find embedding via subword matching.

        If the full word isn't in vocabulary, try progressively
        shorter subword prefixes.
        """
        if self.model_tokenizer is None:
            return None

        subwords = self.model_tokenizer.tokenize(word)

        if not subwords:
            return None

        # Try progressively shorter prefixes
        while len(subwords) > 1:
            subwords = subwords[:-1]

            # Reconstruct text from subwords
            subword_text = "".join(subwords).replace("##", "").replace("▁", "")

            # Try lookup
            vec = self.word2vec.get(subword_text)
            if vec is None:
                vec = self.word2vec.get(subword_text.lower())

            if vec is not None:
                return vec

        return None

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.word2vec)

    def __contains__(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word in self.word2vec or word.lower() in self.word2vec
