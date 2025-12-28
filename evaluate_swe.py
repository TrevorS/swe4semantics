"""
Evaluate trained SWE embeddings on word/sentence similarity tasks.
"""
import numpy as np
from transformers import AutoTokenizer
import string

PUNCTLIST = set(list(string.punctuation))


def load_embeddings(path):
    """Load embeddings from word2vec text format."""
    word2vec = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            word2vec[word] = vec
    return word2vec


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def encode_sentence(sent, tokenizer, word2vec):
    """Encode sentence by averaging word embeddings."""
    words = [x[0].lower() for x in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)]
    embs = [word2vec[w] for w in words if w in word2vec and w not in PUNCTLIST]
    if embs:
        return np.mean(embs, axis=0)
    return None


def evaluate_word_similarity(word2vec):
    """Evaluate on simple word similarity pairs."""
    test_pairs = [
        # Similar pairs (should have high similarity)
        ("king", "queen", True, "royalty"),
        ("man", "woman", True, "gender"),
        ("good", "great", True, "positive adj"),
        ("big", "large", True, "size"),
        ("small", "little", True, "size"),
        ("new", "old", False, "antonyms"),
        ("first", "second", True, "ordinals"),
        ("time", "year", True, "temporal"),
        ("people", "government", False, "different concepts"),
        ("world", "country", True, "geo"),
    ]

    print("\n=== Word Similarity Evaluation ===")
    print("-" * 50)

    for w1, w2, expected_similar, category in test_pairs:
        if w1 in word2vec and w2 in word2vec:
            sim = cosine_similarity(word2vec[w1], word2vec[w2])
            match = (sim > 0.3) == expected_similar
            status = "✓" if match else "✗"
            exp = "high" if expected_similar else "low"
            print(f"{status} {w1:10} - {w2:10} ({category:15}): {sim:6.3f} (expected {exp})")
        else:
            missing = w1 if w1 not in word2vec else w2
            print(f"? {w1:10} - {w2:10}: '{missing}' not in vocab")


def evaluate_sentence_similarity(word2vec, tokenizer):
    """Evaluate on sentence similarity pairs."""
    test_pairs = [
        # Similar sentence pairs
        ("The movie was great and I enjoyed it", "I really liked the film", True),
        ("The weather is sunny today", "Today is a beautiful sunny day", True),
        ("He went to the store", "She visited the shop", True),
        # Dissimilar pairs
        ("The cat is sleeping", "The economy is growing", False),
        ("I love programming", "The food was delicious", False),
        ("Mathematics is difficult", "The garden looks beautiful", False),
    ]

    print("\n=== Sentence Similarity Evaluation ===")
    print("-" * 60)

    for s1, s2, expected_similar in test_pairs:
        emb1 = encode_sentence(s1, tokenizer, word2vec)
        emb2 = encode_sentence(s2, tokenizer, word2vec)

        if emb1 is not None and emb2 is not None:
            sim = cosine_similarity(emb1, emb2)
            match = (sim > 0.4) == expected_similar
            status = "✓" if match else "✗"
            exp = "similar" if expected_similar else "dissimilar"
            print(f"{status} sim={sim:5.3f} ({exp})")
            print(f"    S1: {s1[:50]}...")
            print(f"    S2: {s2[:50]}...")
        else:
            print(f"? Could not encode sentences (missing words)")


def find_most_similar(word, word2vec, topk=5):
    """Find most similar words."""
    if word not in word2vec:
        return []

    target = word2vec[word]
    similarities = []
    for w, vec in word2vec.items():
        if w != word:
            sim = cosine_similarity(target, vec)
            similarities.append((w, sim))

    return sorted(similarities, key=lambda x: -x[1])[:topk]


def main():
    print("Loading embeddings...")
    word2vec = load_embeddings("data/output/swe_1k.txt")
    print(f"Loaded {len(word2vec)} word embeddings")
    print(f"Embedding dimension: {len(list(word2vec.values())[0])}")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    evaluate_word_similarity(word2vec)
    evaluate_sentence_similarity(word2vec, tokenizer)

    # Find similar words for some examples
    print("\n=== Most Similar Words ===")
    print("-" * 50)

    test_words = ["king", "computer", "good", "time", "people"]
    for word in test_words:
        similar = find_most_similar(word, word2vec, topk=5)
        if similar:
            sim_str = ", ".join([f"{w}({s:.2f})" for w, s in similar])
            print(f"{word:12}: {sim_str}")
        else:
            print(f"{word:12}: not in vocabulary")

    print("\n=== Pipeline Summary ===")
    print("-" * 50)
    print(f"Vocabulary size: {len(word2vec)} words")
    print(f"Embedding dim:   {len(list(word2vec.values())[0])}")
    print("Pipeline:        Extract → PCA+ABTT → Distillation")
    print("Base model:      distilbert-base-uncased")
    print("Teacher model:   all-MiniLM-L6-v2")


if __name__ == "__main__":
    main()
