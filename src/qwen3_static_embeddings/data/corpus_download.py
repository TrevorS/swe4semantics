"""Corpus download utilities for replicating SWE4Semantics experiments.

The original paper uses:
- CC-100 (https://data.statmt.org/cc-100/) for English
- CCMatrix (https://opus.nlpl.eu/CCMatrix/) for cross-lingual

This module provides utilities to download and prepare these corpora.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from tqdm import tqdm


@dataclass
class CorpusConfig:
    """Configuration for corpus download."""

    name: str
    language: str = "en"
    max_sentences: int | None = None  # None = all
    output_path: Path | None = None
    streaming: bool = True  # Stream for large datasets


# Available corpora matching the paper
CORPUS_CONFIGS = {
    # CC-100: Used for English SWE training in the paper
    "cc100": CorpusConfig(
        name="cc100",
        language="en",
    ),
    # CCMatrix: Used for cross-lingual training
    "ccmatrix-en-de": CorpusConfig(
        name="ccmatrix",
        language="en-de",
    ),
    "ccmatrix-en-zh": CorpusConfig(
        name="ccmatrix",
        language="en-zh",
    ),
    "ccmatrix-en-ja": CorpusConfig(
        name="ccmatrix",
        language="en-ja",
    ),
}


def download_cc100(
    language: str = "en",
    output_path: Path | None = None,
    max_sentences: int | None = None,
    streaming: bool = True,
    show_progress: bool = True,
) -> Path | Iterator[str]:
    """
    Download CC-100 corpus using HuggingFace datasets.

    CC-100 is the corpus used in the SWE4Semantics paper for English.
    It contains ~82GB of cleaned Common Crawl text for English.

    Uses HuggingFace's text loader with the raw CC-100 data files from
    https://data.statmt.org/cc-100/

    Args:
        language: Language code (e.g., "en", "de", "zh", "ja")
        output_path: Path to save corpus (if None, returns iterator)
        max_sentences: Maximum sentences to download (None = all)
        streaming: Whether to stream (recommended for large datasets)
        show_progress: Whether to show progress bar

    Returns:
        Path to saved file, or iterator of sentences if output_path is None
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("Please install datasets: pip install datasets") from err

    # CC-100 raw data URLs from statmt.org
    cc100_urls = {
        "en": "https://data.statmt.org/cc-100/en.txt.xz",
        "de": "https://data.statmt.org/cc-100/de.txt.xz",
        "zh": "https://data.statmt.org/cc-100/zh-Hans.txt.xz",
        "ja": "https://data.statmt.org/cc-100/ja.txt.xz",
        "fr": "https://data.statmt.org/cc-100/fr.txt.xz",
        "es": "https://data.statmt.org/cc-100/es.txt.xz",
    }

    if language not in cc100_urls:
        available = ", ".join(cc100_urls.keys())
        raise ValueError(
            f"Language '{language}' not configured. Available: {available}. "
            f"For other languages, see https://data.statmt.org/cc-100/"
        )

    data_url = cc100_urls[language]
    print(f"Loading CC-100 ({language}) via HuggingFace datasets...")
    print(f"Source: {data_url}")

    # Use HuggingFace's text loader with the raw data URL
    # This properly streams the compressed file
    dataset = load_dataset(
        "text",
        data_files=data_url,
        split="train",
        streaming=streaming,
    )

    def sentence_iterator():
        """Iterate over sentences in the corpus."""
        from qwen3_static_embeddings.data.corpus import split_sentences

        count = 0
        iterator = dataset

        if show_progress and max_sentences:
            iterator = tqdm(iterator, total=max_sentences, desc=f"CC-100 ({language})")
        elif show_progress:
            iterator = tqdm(iterator, desc=f"CC-100 ({language})")

        for item in iterator:
            text = item.get("text", "")
            if not text:
                continue

            # Split into sentences using BlingFire (as in original paper)
            for sentence in split_sentences(text):
                sentence = sentence.strip()
                if len(sentence) > 10:  # Skip very short
                    yield sentence
                    count += 1

                    if max_sentences and count >= max_sentences:
                        return

    # If no output path, return iterator
    if output_path is None:
        return sentence_iterator()

    # Otherwise save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for sentence in sentence_iterator():
            f.write(sentence + "\n")

    print(f"Saved {output_path}")
    return output_path


def download_ccmatrix(
    lang_pair: str = "en-de",
    output_path: Path | None = None,
    max_pairs: int | None = None,
    show_progress: bool = True,
) -> tuple[Path, Path] | Iterator[tuple[str, str]]:
    """
    Download CCMatrix parallel corpus from OPUS.

    CCMatrix is used in the SWE4Semantics paper for cross-lingual training.

    Args:
        lang_pair: Language pair (e.g., "en-de", "en-zh", "en-ja")
        output_path: Base path for output files (creates {lang1}.txt, {lang2}.txt)
        max_pairs: Maximum sentence pairs to download
        show_progress: Whether to show progress bar

    Returns:
        Tuple of paths, or iterator of (sent1, sent2) pairs
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        raise ImportError("Please install datasets: pip install datasets") from err

    lang1, lang2 = lang_pair.split("-")
    print(f"Loading CCMatrix ({lang_pair}) from HuggingFace...")

    # Try loading from opus-mt or similar
    # Note: CCMatrix may require manual download from OPUS
    try:
        dataset = load_dataset(
            "opus/CCMatrix",
            lang_pair,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as err:
        # Fallback: try alternative source
        print("CCMatrix not available on HuggingFace, trying OPUS...")
        print(
            "Please download manually from: https://opus.nlpl.eu/CCMatrix/corpus/version/CCMatrix"
        )
        raise ValueError(
            f"CCMatrix ({lang_pair}) requires manual download from OPUS. "
            f"Visit: https://opus.nlpl.eu/CCMatrix/corpus/version/CCMatrix"
        ) from err

    def pair_iterator():
        count = 0
        iterator = dataset
        if show_progress:
            iterator = tqdm(iterator, desc=f"CCMatrix ({lang_pair})")

        for item in iterator:
            sent1 = item.get("translation", {}).get(lang1, "")
            sent2 = item.get("translation", {}).get(lang2, "")

            if sent1 and sent2:
                yield (sent1.strip(), sent2.strip())
                count += 1

                if max_pairs and count >= max_pairs:
                    return

    if output_path is None:
        return pair_iterator()

    # Save to separate files
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    path1 = output_path.parent / f"{lang1}.txt"
    path2 = output_path.parent / f"{lang2}.txt"

    print(f"Saving to {path1} and {path2}...")
    with open(path1, "w", encoding="utf-8") as f1, open(path2, "w", encoding="utf-8") as f2:
        for sent1, sent2 in pair_iterator():
            f1.write(sent1 + "\n")
            f2.write(sent2 + "\n")

    return path1, path2


def get_corpus_sample(
    corpus: str = "cc100",
    language: str = "en",
    n_sentences: int = 10000,
) -> list[str]:
    """
    Get a sample of sentences from a corpus.

    Useful for quick testing without downloading full corpus.

    Args:
        corpus: Corpus name ("cc100" or "ccmatrix")
        language: Language code
        n_sentences: Number of sentences to sample

    Returns:
        List of sampled sentences
    """
    if corpus == "cc100":
        result = download_cc100(
            language=language,
            max_sentences=n_sentences,
            streaming=True,
            show_progress=True,
        )
        # When output_path is None, download_cc100 returns an iterator
        iterator = cast("Iterator[str]", result)
        return list(iterator)
    else:
        raise ValueError(f"Unknown corpus: {corpus}")


def estimate_corpus_size(corpus: str = "cc100", language: str = "en") -> dict:
    """
    Estimate corpus size without downloading.

    Returns:
        Dict with size estimates
    """
    # Approximate sizes based on documentation
    sizes = {
        "cc100": {
            "en": {"sentences": "~300M", "size_gb": "~82GB"},
            "de": {"sentences": "~100M", "size_gb": "~27GB"},
            "zh": {"sentences": "~50M", "size_gb": "~15GB"},
            "ja": {"sentences": "~40M", "size_gb": "~12GB"},
        }
    }

    return sizes.get(corpus, {}).get(language, {"sentences": "unknown", "size_gb": "unknown"})
