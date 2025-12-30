"""Dataset utilities for training."""

import numpy as np

from qwen3_static_embeddings.data.word2sent import Word2Sent


def sample_training_sentences(
    word2sent: Word2Sent,
    n_per_word: int = 3,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """
    Sample training and validation sentences from word2sent mapping.

    Args:
        word2sent: Word to sentences mapping
        n_per_word: Number of sentences to sample per word
        train_ratio: Ratio of sentences for training (rest for validation)
        seed: Random seed

    Returns:
        Tuple of (train_sentences, val_sentences)
    """
    np.random.seed(seed)

    all_sentences = set()

    for word in word2sent.words:
        sentences = word2sent.get_sentences(word)
        if len(sentences) >= n_per_word:
            # Sample n_per_word sentences
            sampled = np.random.choice(sentences, size=n_per_word, replace=False)
            all_sentences.update(sampled)

    all_sentences = list(all_sentences)
    np.random.shuffle(all_sentences)

    # Split into train/val
    split_idx = int(len(all_sentences) * train_ratio)
    train_sentences = all_sentences[:split_idx]
    val_sentences = all_sentences[split_idx:]

    return train_sentences, val_sentences


def sample_contrastive_pairs(
    word2sent: Word2Sent,
    n_pairs: int = 10000,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """
    Sample sentence pairs for contrastive learning.

    Pairs are sampled such that they share at least one word.

    Args:
        word2sent: Word to sentences mapping
        n_pairs: Number of pairs to sample
        seed: Random seed

    Returns:
        List of (sentence1, sentence2) tuples
    """
    np.random.seed(seed)

    pairs = []
    words = word2sent.words

    while len(pairs) < n_pairs and words:
        # Sample a word
        word = np.random.choice(words)
        sentences = word2sent.get_sentences(word)

        if len(sentences) >= 2:
            # Sample two sentences containing this word
            sampled = np.random.choice(sentences, size=2, replace=False)
            pairs.append((sampled[0], sampled[1]))

    return pairs
