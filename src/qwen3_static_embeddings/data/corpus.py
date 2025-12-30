"""Corpus loading and sentence splitting utilities."""

from collections.abc import Iterator
from pathlib import Path

from tqdm import tqdm

# Optional blingfire import - falls back to simple splitting if not available
try:
    from blingfire import text_to_sentences

    HAS_BLINGFIRE = True
except ImportError:
    HAS_BLINGFIRE = False


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Uses BlingFire if available (recommended), otherwise falls back to
    simple period-based splitting.

    Args:
        text: Input text to split

    Returns:
        List of sentences
    """
    if HAS_BLINGFIRE:
        return text_to_sentences(text).split("\n")
    else:
        # Simple fallback - split on sentence-ending punctuation
        import re

        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]


def iter_corpus_lines(path: Path) -> Iterator[str]:
    """
    Iterate over lines in a corpus file.

    Handles large files by streaming line by line.

    Args:
        path: Path to corpus file (text file, one document per line)

    Yields:
        Individual lines from the file
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def iter_corpus_sentences(
    path: Path,
    show_progress: bool = True,
) -> Iterator[str]:
    """
    Iterate over all sentences in a corpus file.

    Reads documents line by line and splits each into sentences.

    Args:
        path: Path to corpus file
        show_progress: Whether to show progress bar

    Yields:
        Individual sentences from the corpus
    """
    lines = iter_corpus_lines(path)

    if show_progress:
        lines = tqdm(lines, desc=f"Reading {path.name}")

    for line in lines:
        for sentence in split_sentences(line):
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short sentences
                yield sentence


def count_corpus_lines(path: Path) -> int:
    """Count number of lines in a corpus file."""
    count = 0
    with open(path, encoding="utf-8", errors="ignore") as f:
        for _ in f:
            count += 1
    return count
