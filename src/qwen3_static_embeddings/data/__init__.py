"""Data processing: corpus loading, vocabulary building, and word2sent mapping."""

from qwen3_static_embeddings.data.corpus_download import (
    CORPUS_CONFIGS,
    download_cc100,
    download_ccmatrix,
    estimate_corpus_size,
    get_corpus_sample,
)
from qwen3_static_embeddings.data.vocabulary import Vocabulary, build_vocabulary
from qwen3_static_embeddings.data.word2sent import Word2Sent, build_word2sent

__all__ = [
    "Vocabulary",
    "build_vocabulary",
    "Word2Sent",
    "build_word2sent",
    "download_cc100",
    "download_ccmatrix",
    "get_corpus_sample",
    "estimate_corpus_size",
    "CORPUS_CONFIGS",
]
