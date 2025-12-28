"""
Compare SWE embeddings from different model configurations.
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


def evaluate_word_pairs(word2vec, name):
    """Evaluate on word similarity pairs."""
    test_pairs = [
        ("man", "woman", True),
        ("good", "great", True),
        ("big", "large", True),
        ("new", "old", False),
        ("first", "second", True),
        ("time", "year", True),
        ("people", "government", False),
        ("world", "country", True),
        ("high", "low", False),
        ("fast", "quick", True),
    ]

    correct = 0
    total = 0
    similarities = []

    for w1, w2, expected_similar in test_pairs:
        if w1 in word2vec and w2 in word2vec:
            sim = cosine_similarity(word2vec[w1], word2vec[w2])
            is_correct = (sim > 0.3) == expected_similar
            correct += is_correct
            total += 1
            similarities.append((w1, w2, sim, expected_similar, is_correct))

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, similarities


def evaluate_analogy(word2vec, name):
    """Evaluate on word analogies (king - man + woman = queen)."""
    analogies = [
        ("man", "woman", "king", "queen"),
        ("big", "bigger", "small", "smaller"),
        ("good", "better", "bad", "worse"),
    ]

    correct = 0
    total = 0

    for w1, w2, w3, expected in analogies:
        if all(w in word2vec for w in [w1, w2, w3, expected]):
            # a - b + c = d  =>  d should be closest to a - b + c
            target = word2vec[w1] - word2vec[w2] + word2vec[w3]

            # Find closest word
            best_sim = -1
            best_word = None
            for w, vec in word2vec.items():
                if w not in [w1, w2, w3]:
                    sim = cosine_similarity(target, vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_word = w

            is_correct = best_word == expected
            correct += is_correct
            total += 1

    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy


def find_similar_words(word, word2vec, topk=5):
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
    models = [
        ("data/output/swe_1k.txt", "DistilBERT + MiniLM", "fast"),
        ("data/output/swe_bge_1k.txt", "BGE-small + BGE-base", "balanced"),
        ("data/output/swe_qwen_1k.txt", "Qwen3-Embedding-0.6B (1k)", "sota"),
    ]

    print("=" * 70)
    print("SWE Embedding Model Comparison")
    print("=" * 70)

    results = []

    for path, name, quality in models:
        try:
            word2vec = load_embeddings(path)
            print(f"\n{'='*70}")
            print(f"Model: {name} ({quality})")
            print(f"Vocabulary: {len(word2vec)} words, {len(list(word2vec.values())[0])}d")
            print("=" * 70)

            # Word similarity
            accuracy, similarities = evaluate_word_pairs(word2vec, name)
            print(f"\nWord Pair Similarity: {accuracy:.1f}%")
            for w1, w2, sim, expected, correct in similarities:
                status = "✓" if correct else "✗"
                exp = "high" if expected else "low"
                print(f"  {status} {w1:10} - {w2:10}: {sim:6.3f} (expected {exp})")

            # Analogies
            analogy_acc = evaluate_analogy(word2vec, name)
            print(f"\nAnalogy Accuracy: {analogy_acc:.1f}%")

            # Sample similar words
            print("\nMost Similar Words:")
            for word in ["good", "time", "people"]:
                similar = find_similar_words(word, word2vec, topk=3)
                if similar:
                    sim_str = ", ".join([f"{w}({s:.2f})" for w, s in similar])
                    print(f"  {word:12}: {sim_str}")

            results.append({
                "name": name,
                "quality": quality,
                "word_sim_acc": accuracy,
                "analogy_acc": analogy_acc
            })

        except FileNotFoundError:
            print(f"\nSkipping {name}: file not found")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Model':<30} {'Quality':<12} {'Word Sim':<12} {'Analogy':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<30} {r['quality']:<12} {r['word_sim_acc']:>6.1f}%      {r['analogy_acc']:>6.1f}%")


if __name__ == "__main__":
    main()
