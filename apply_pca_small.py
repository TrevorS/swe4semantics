"""
Simplified PCA script for small-scale testing.
Works with small vocabularies by using all available sentences.
"""
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.decomposition import PCA
from util import load_w2v
import string

PUNCTLIST = list(string.punctuation)


def encode_sentences(tokenizer, word2vec, sents, model_tokenizer):
    """Encode sentences by averaging word vectors."""
    embeddings = []

    for sent in tqdm(sents, desc="Encoding sentences"):
        words = [x[0] for x in
                tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)]

        sent_emb = []
        for w in words:
            if w not in PUNCTLIST:
                if w in word2vec:
                    sent_emb.append(word2vec[w])
                elif w.lower() in word2vec:
                    sent_emb.append(word2vec[w.lower()])

        if len(sent_emb):
            sent_emb = np.vstack(sent_emb)
            embeddings.append(sent_emb.mean(axis=0))

    if len(embeddings) == 0:
        raise ValueError("No valid sentences encoded")

    return np.vstack(embeddings)


def main():
    parser = argparse.ArgumentParser(description='Apply PCA (small-scale version)')
    parser.add_argument('-model', required=True, help='HuggingFace model')
    parser.add_argument('-vec_path', required=True, help='Path to word vectors')
    parser.add_argument('-output_folder', required=True, help='Output folder')
    parser.add_argument('-word2sent', required=True, help='word2sent pickle file')
    parser.add_argument('-embd', default=64, type=int, help='Output embedding dimension')
    parser.add_argument('-d_remove', default=1, type=int, help='Components to remove (ABTT)')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Load word vectors
    print(f"Loading word vectors from {args.vec_path}")
    word2vec, edim = load_w2v(args.vec_path)
    print(f"Vocabulary size: {len(word2vec)}, Embedding dim: {edim}")

    # Load all sentences
    print(f"Loading sentences from {args.word2sent}")
    with open(args.word2sent, 'rb') as f:
        word2sent_dict = pickle.load(f)

    # Collect ALL sentences for PCA (not just one per word)
    lines = []
    for w in word2sent_dict:
        lines.extend(word2sent_dict[w])

    lines = list(set(lines))  # Remove duplicates
    print(f"Collected {len(lines)} unique sentences for PCA")

    # Initialize tokenizers
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    model_tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Encode sentences
    sent_embs = encode_sentences(tokenizer, word2vec, lines, model_tokenizer)
    print(f"Encoded sentence embeddings shape: {sent_embs.shape}")

    # Determine max components
    n_samples, n_features = sent_embs.shape
    max_components = min(n_samples, n_features)
    target_dim = min(args.embd + args.d_remove, max_components)

    print(f"\nFitting PCA with {target_dim} components (max possible: {max_components})")

    pca = PCA(n_components=target_dim, whiten=False)
    pca.fit(sent_embs)

    print(f"PCA components shape: {pca.components_.shape}")

    # Save PCA components
    np.save(f"{args.output_folder}/all_pca_mean.npy", pca.mean_)
    np.save(f"{args.output_folder}/all_pca_components.npy", pca.components_)

    # Report variance explained
    print("\n" + "="*60)
    print("Variance Explained by Principal Components:")
    print("="*60)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    for i in [1, 3, 5, 10, 20]:
        if i <= len(cumsum):
            print(f"Top {i} components: {cumsum[i-1]:.4f}")
    print("="*60)

    # Transform word vectors
    print("\nTransforming word vectors...")
    vocab = list(word2vec.keys())
    emb_list = np.array([word2vec[w] for w in vocab])

    # Apply ABTT: remove first d_remove components
    actual_dim = min(args.embd, target_dim - args.d_remove)
    if actual_dim <= 0:
        actual_dim = target_dim  # Use all if we can't remove any
        d_remove = 0
    else:
        d_remove = args.d_remove

    print(f"Removing first {d_remove} components, keeping {actual_dim} dimensions")
    new_emb = (emb_list - pca.mean_).dot(pca.components_[d_remove:d_remove+actual_dim].T)
    print(f"Transformed embeddings shape: {new_emb.shape}")

    # Save transformed vectors
    output_file = f"{args.output_folder}/vec.txt"
    with open(output_file, "w") as f:
        for i, w in enumerate(vocab):
            vec = new_emb[i]
            f.write(w + " " + " ".join([str(x) for x in vec]) + "\n")

    print(f"\n✓ Saved {len(vocab)} transformed word vectors to {output_file}")
    print("Done!")


if __name__ == "__main__":
    main()
