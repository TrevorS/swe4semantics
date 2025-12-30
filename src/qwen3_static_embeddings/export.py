"""Export embeddings to various formats for compatibility with downstream tools.

Supported formats:
- Word2Vec text format (.txt)
- Word2Vec binary format (.bin)
- GloVe text format (.txt)
- NumPy format (.npz)
"""

from pathlib import Path

import numpy as np


def save_word2vec_text(
    word2vec: dict[str, np.ndarray],
    output_path: Path | str,
) -> Path:
    """
    Save embeddings in Word2Vec text format.

    Format: First line is "vocab_size dim", followed by "word vec1 vec2 ..." lines.

    Args:
        word2vec: Dictionary mapping words to embedding vectors
        output_path: Path to save the embeddings

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_size = len(word2vec)
    dim = len(next(iter(word2vec.values())))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{vocab_size} {dim}\n")
        for word, vec in word2vec.items():
            vec_str = " ".join(f"{v:.6f}" for v in vec)
            f.write(f"{word} {vec_str}\n")

    print(f"Saved {vocab_size} embeddings to {output_path} (Word2Vec text)")
    return output_path


def save_word2vec_binary(
    word2vec: dict[str, np.ndarray],
    output_path: Path | str,
) -> Path:
    """
    Save embeddings in Word2Vec binary format (.bin).

    Compatible with gensim.models.KeyedVectors.load_word2vec_format(binary=True).

    Args:
        word2vec: Dictionary mapping words to embedding vectors
        output_path: Path to save the embeddings

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_size = len(word2vec)
    dim = len(next(iter(word2vec.values())))

    with open(output_path, "wb") as f:
        # Header line (text): "vocab_size dim\n"
        header = f"{vocab_size} {dim}\n"
        f.write(header.encode("utf-8"))

        # Each word: word + space + binary floats + newline
        for word, vec in word2vec.items():
            word_bytes = word.encode("utf-8")
            f.write(word_bytes)
            f.write(b" ")
            # Pack as float32 (4 bytes each)
            vec_float32 = vec.astype(np.float32)
            f.write(vec_float32.tobytes())
            f.write(b"\n")

    print(f"Saved {vocab_size} embeddings to {output_path} (Word2Vec binary)")
    return output_path


def save_glove_text(
    word2vec: dict[str, np.ndarray],
    output_path: Path | str,
) -> Path:
    """
    Save embeddings in GloVe text format.

    Format: "word vec1 vec2 ..." lines (no header).
    GloVe format is identical to Word2Vec text without the header line.

    Args:
        word2vec: Dictionary mapping words to embedding vectors
        output_path: Path to save the embeddings

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    vocab_size = len(word2vec)

    with open(output_path, "w", encoding="utf-8") as f:
        for word, vec in word2vec.items():
            vec_str = " ".join(f"{v:.6f}" for v in vec)
            f.write(f"{word} {vec_str}\n")

    print(f"Saved {vocab_size} embeddings to {output_path} (GloVe text)")
    return output_path


def save_numpy(
    word2vec: dict[str, np.ndarray],
    output_path: Path | str,
) -> Path:
    """
    Save embeddings in NumPy compressed format (.npz).

    Creates two arrays:
    - words: array of word strings
    - vectors: 2D array of shape (vocab_size, dim)

    Load with:
        data = np.load("embeddings.npz", allow_pickle=True)
        words = data["words"]
        vectors = data["vectors"]

    Args:
        word2vec: Dictionary mapping words to embedding vectors
        output_path: Path to save the embeddings

    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    words = np.array(list(word2vec.keys()))
    vectors = np.array(list(word2vec.values()))

    np.savez_compressed(output_path, words=words, vectors=vectors)

    print(f"Saved {len(words)} embeddings to {output_path} (NumPy npz)")
    return output_path


def load_word2vec_text(input_path: Path | str) -> dict[str, np.ndarray]:
    """
    Load embeddings from Word2Vec text format.

    Args:
        input_path: Path to the embeddings file

    Returns:
        Dictionary mapping words to embedding vectors
    """
    word2vec = {}

    with open(input_path, encoding="utf-8") as f:
        # First line is header
        header = f.readline().strip()
        vocab_size, dim = map(int, header.split())

        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            word2vec[word] = vec

    print(f"Loaded {len(word2vec)} embeddings from {input_path}")
    return word2vec


def load_word2vec_binary(input_path: Path | str) -> dict[str, np.ndarray]:
    """
    Load embeddings from Word2Vec binary format.

    Args:
        input_path: Path to the embeddings file

    Returns:
        Dictionary mapping words to embedding vectors
    """
    word2vec = {}

    with open(input_path, "rb") as f:
        # Read header (text line)
        header = b""
        while True:
            c = f.read(1)
            if c == b"\n":
                break
            header += c
        vocab_size, dim = map(int, header.decode("utf-8").split())

        # Read each word
        for _ in range(vocab_size):
            # Read word until space
            word_bytes = b""
            while True:
                c = f.read(1)
                if c == b" ":
                    break
                word_bytes += c
            word = word_bytes.decode("utf-8")

            # Read vector (dim * 4 bytes for float32)
            vec_bytes = f.read(dim * 4)
            vec = np.frombuffer(vec_bytes, dtype=np.float32).copy()
            word2vec[word] = vec

            # Read newline
            f.read(1)

    print(f"Loaded {len(word2vec)} embeddings from {input_path}")
    return word2vec


def load_glove_text(input_path: Path | str) -> dict[str, np.ndarray]:
    """
    Load embeddings from GloVe text format.

    Args:
        input_path: Path to the embeddings file

    Returns:
        Dictionary mapping words to embedding vectors
    """
    word2vec = {}

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            word2vec[word] = vec

    print(f"Loaded {len(word2vec)} embeddings from {input_path}")
    return word2vec


def load_numpy(input_path: Path | str) -> dict[str, np.ndarray]:
    """
    Load embeddings from NumPy compressed format.

    Args:
        input_path: Path to the embeddings file

    Returns:
        Dictionary mapping words to embedding vectors
    """
    data = np.load(input_path, allow_pickle=True)
    words = data["words"]
    vectors = data["vectors"]

    word2vec = {word: vec for word, vec in zip(words, vectors, strict=True)}
    print(f"Loaded {len(word2vec)} embeddings from {input_path}")
    return word2vec


def export_embeddings(
    word2vec: dict[str, np.ndarray],
    output_dir: Path | str,
    name: str = "embeddings",
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """
    Export embeddings to multiple formats at once.

    Args:
        word2vec: Dictionary mapping words to embedding vectors
        output_dir: Directory to save the embeddings
        name: Base name for the output files
        formats: List of formats to export. Options: "word2vec_text", "word2vec_binary",
                 "glove_text", "numpy". Default: all formats.

    Returns:
        Dictionary mapping format names to output paths
    """
    if formats is None:
        formats = ["word2vec_text", "word2vec_binary", "glove_text", "numpy"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for fmt in formats:
        if fmt == "word2vec_text":
            path = save_word2vec_text(word2vec, output_dir / f"{name}.w2v.txt")
        elif fmt == "word2vec_binary":
            path = save_word2vec_binary(word2vec, output_dir / f"{name}.w2v.bin")
        elif fmt == "glove_text":
            path = save_glove_text(word2vec, output_dir / f"{name}.glove.txt")
        elif fmt == "numpy":
            path = save_numpy(word2vec, output_dir / f"{name}.npz")
        else:
            raise ValueError(f"Unknown format: {fmt}")

        results[fmt] = path

    return results
