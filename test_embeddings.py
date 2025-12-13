"""
Test the trained SWE embeddings by computing sentence similarities.
"""
import numpy as np
from transformers import AutoTokenizer
from util import load_w2v
import string

PUNCTLIST = set(list(string.punctuation))


def encode_sentence(sent, tokenizer, word2vec, edim):
    """Encode a sentence by averaging word embeddings."""
    words = [x[0] for x in
            tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)]

    embs = []
    for w in words:
        if w not in PUNCTLIST:
            if w in word2vec:
                embs.append(word2vec[w])
            elif w.lower() in word2vec:
                embs.append(word2vec[w.lower()])

    if len(embs) == 0:
        return np.zeros(edim)

    embs = np.vstack(embs)
    sent_emb = embs.mean(axis=0)
    # L2 normalize
    sent_emb = sent_emb / np.linalg.norm(sent_emb)
    return sent_emb


def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)


def main():
    print("Loading trained embeddings...")
    word2vec, edim = load_w2v("data/trained/trained_embeddings.txt")
    print(f"Loaded {len(word2vec)} words, dimension: {edim}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Test sentence pairs (semantic similarity tests)
    test_pairs = [
        # Similar sentences
        ("The computer processes millions of calculations.",
         "A powerful computer is needed for gaming."),
        ("Machine learning transforms many industries.",
         "Deep learning requires large datasets."),
        ("The algorithm runs in linear time.",
         "The algorithm complexity was carefully analyzed."),

        # Dissimilar sentences
        ("The computer crashed while I was writing.",
         "The language model generates fluent text."),
        ("She trained a new model yesterday.",
         "The network connection was unstable."),
    ]

    print("\n" + "="*70)
    print("SENTENCE SIMILARITY TEST")
    print("="*70)

    for i, (sent1, sent2) in enumerate(test_pairs):
        emb1 = encode_sentence(sent1, tokenizer, word2vec, edim)
        emb2 = encode_sentence(sent2, tokenizer, word2vec, edim)

        sim = cosine_similarity(emb1, emb2)

        print(f"\nPair {i+1}:")
        print(f"  Sent1: {sent1[:60]}...")
        print(f"  Sent2: {sent2[:60]}...")
        print(f"  Similarity: {sim:.4f}")

    print("\n" + "="*70)

    # Compare with raw (pre-training) embeddings
    print("\nComparing with PCA embeddings (before distillation):")
    word2vec_pca, _ = load_w2v("data/pca_output/vec.txt")

    for i, (sent1, sent2) in enumerate(test_pairs[:2]):
        emb1_trained = encode_sentence(sent1, tokenizer, word2vec, edim)
        emb2_trained = encode_sentence(sent2, tokenizer, word2vec, edim)

        emb1_pca = encode_sentence(sent1, tokenizer, word2vec_pca, edim)
        emb2_pca = encode_sentence(sent2, tokenizer, word2vec_pca, edim)

        sim_trained = cosine_similarity(emb1_trained, emb2_trained)
        sim_pca = cosine_similarity(emb1_pca, emb2_pca)

        print(f"\nPair {i+1}:")
        print(f"  Before distillation: {sim_pca:.4f}")
        print(f"  After distillation:  {sim_trained:.4f}")

    print("\n" + "="*70)
    print("Pipeline test completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
