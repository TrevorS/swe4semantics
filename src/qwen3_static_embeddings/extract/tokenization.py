"""Tokenization utilities for finding word positions in sentences."""

import numpy as np


def find_word_positions(
    input_ids: list[int],
    word: str,
    word_token_ids: list[int],
    tokenizer,
    start_idx: int = 0,
    subword_mode: bool = False,
) -> list[int] | None:
    """
    Find the token positions where a word appears in a tokenized sentence.

    Args:
        input_ids: Token IDs of the full sentence
        word: The target word to find
        word_token_ids: Token IDs of the word alone
        tokenizer: The tokenizer (for subword mode)
        start_idx: Index to start searching from (skip prompt tokens)
        subword_mode: If True, match subwords within single tokens

    Returns:
        List of token indices where the word appears, or None if not found
    """
    if len(word_token_ids) == 0:
        return None

    # Search through the sentence
    for i in range(start_idx, len(input_ids) - len(word_token_ids) + 1):
        if subword_mode:
            # Subword mode: check if word appears within a single token
            token_text = tokenizer.convert_ids_to_tokens([input_ids[i]])[0]
            # Remove special markers
            token_text = token_text.replace("▁", "").replace("##", "").strip()
            if token_text == word:
                return [i]
        else:
            # Standard mode: match exact token ID sequence
            match = True
            for j in range(len(word_token_ids)):
                if input_ids[i + j] != word_token_ids[j]:
                    match = False
                    break

            if match:
                return [i + j for j in range(len(word_token_ids))]

    return None


def find_word_positions_batch(
    batch_input_ids: list[list[int]],
    word: str,
    word_token_ids: list[int],
    tokenizer,
    start_idx: int = 0,
    subword_mode: bool = False,
    max_sent_len: int = 500,
) -> tuple[np.ndarray, list[int]]:
    """
    Find word positions across a batch of sentences.

    Args:
        batch_input_ids: List of token ID lists for each sentence
        word: Target word to find
        word_token_ids: Token IDs of the word
        tokenizer: The tokenizer
        start_idx: Index to start searching from
        subword_mode: Whether to use subword matching
        max_sent_len: Skip sentences longer than this

    Returns:
        Tuple of:
            - positions: Array of shape (n_valid, n_tokens) with token indices
            - valid_sent_ids: List of sentence indices where word was found
    """
    positions = []
    valid_sent_ids = []

    for sent_idx, input_ids in enumerate(batch_input_ids):
        # Skip very long sentences
        if len(input_ids) > max_sent_len:
            continue

        word_pos = find_word_positions(
            input_ids=input_ids,
            word=word,
            word_token_ids=word_token_ids,
            tokenizer=tokenizer,
            start_idx=start_idx,
            subword_mode=subword_mode,
        )

        if word_pos is not None:
            positions.append(word_pos)
            valid_sent_ids.append(sent_idx)

    if not positions:
        return np.array([]), []

    return np.array(positions), valid_sent_ids


def tokenize_word(
    word: str,
    tokenizer,
    add_space: bool = True,
) -> list[int]:
    """
    Tokenize a word to get its token IDs.

    Args:
        word: The word to tokenize
        tokenizer: The tokenizer
        add_space: Whether to add leading space (for proper tokenization)

    Returns:
        List of token IDs
    """
    text = " " + word if add_space else word
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    # Handle sentencepiece-style tokenizers that add ▁ as separate token
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    if tokens and tokens[0] == "▁":
        token_ids = token_ids[1:]

    return token_ids
