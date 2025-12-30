"""Tests for tokenization module."""


from qwen3_static_embeddings.extract.tokenization import (
    find_word_positions,
    tokenize_word,
)


# Mock tokenizer for testing
class MockTokenizer:
    """Simple mock tokenizer for testing."""

    def __init__(self):
        self.vocab = {
            "hello": 1,
            "world": 2,
            "the": 3,
            "quick": 4,
            "brown": 5,
            "fox": 6,
            "▁": 0,
        }
        self.id2token = {v: k for k, v in self.vocab.items()}

    def __call__(self, text, add_special_tokens=False):
        tokens = text.strip().lower().split()
        ids = [self.vocab.get(t, 99) for t in tokens]
        return {"input_ids": ids}

    def convert_ids_to_tokens(self, ids):
        return [self.id2token.get(i, "[UNK]") for i in ids]


def test_find_word_positions_basic():
    """Test finding word positions in tokenized sentence."""
    tokenizer = MockTokenizer()

    # Sentence: "the quick brown fox"
    input_ids = [3, 4, 5, 6]  # the quick brown fox
    word_token_ids = [5]  # brown

    positions = find_word_positions(
        input_ids=input_ids,
        word="brown",
        word_token_ids=word_token_ids,
        tokenizer=tokenizer,
    )

    assert positions == [2]  # "brown" is at index 2


def test_find_word_positions_not_found():
    """Test when word is not in sentence."""
    tokenizer = MockTokenizer()

    input_ids = [3, 4, 5, 6]  # the quick brown fox
    word_token_ids = [1]  # hello

    positions = find_word_positions(
        input_ids=input_ids,
        word="hello",
        word_token_ids=word_token_ids,
        tokenizer=tokenizer,
    )

    assert positions is None


def test_find_word_positions_with_start_idx():
    """Test searching from a start index."""
    tokenizer = MockTokenizer()

    # Sentence with repeated word: "the the quick"
    input_ids = [3, 3, 4]  # the the quick
    word_token_ids = [3]  # the

    # Search from beginning - should find first occurrence
    positions = find_word_positions(
        input_ids=input_ids,
        word="the",
        word_token_ids=word_token_ids,
        tokenizer=tokenizer,
        start_idx=0,
    )
    assert positions == [0]

    # Search from index 1 - should find second occurrence
    positions = find_word_positions(
        input_ids=input_ids,
        word="the",
        word_token_ids=word_token_ids,
        tokenizer=tokenizer,
        start_idx=1,
    )
    assert positions == [1]


def test_tokenize_word():
    """Test word tokenization."""
    tokenizer = MockTokenizer()

    # Note: This is a simplified test - real tokenizers behave differently
    ids = tokenize_word("hello", tokenizer, add_space=False)
    assert len(ids) > 0
