#!/usr/bin/env python3
"""
Multi-Sense SWE Pipeline

Instead of averaging all contextual embeddings for a word, we:
1. Cluster the contextual embeddings using k-means
2. Keep top-k sense centroids per word
3. At inference, use primary sense or simple disambiguation

This recovers polysemy while keeping inference fast (still just lookups).
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Lazy imports for heavy libraries
torch = None
AutoTokenizer = None
AutoModel = None


def lazy_import_torch():
    global torch, AutoTokenizer, AutoModel
    if torch is None:
        import torch as _torch
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import AutoModel as _AutoModel
        torch = _torch
        AutoTokenizer = _AutoTokenizer
        AutoModel = _AutoModel


MODEL_CONFIGS = {
    "fast": {
        "base": "distilbert-base-uncased",
        "teacher": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "qwen": {
        "base": "Qwen/Qwen3-Embedding-0.6B",
        "teacher": "Qwen/Qwen3-Embedding-0.6B",
    }
}

# Required words for fair evaluation comparison
REQUIRED_WORDS = [
    "man", "woman", "good", "great", "big", "large", "new", "old",
    "first", "second", "time", "year", "people", "government",
    "world", "country", "high", "low", "king", "queen", "boy", "girl",
    "better", "best", "small", "little", "day", "night", "city", "town"
]


def create_word2sent(max_words=1000):
    """Create word -> sentences mapping from wikitext."""
    from datasets import load_dataset

    print("Loading wikitext dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    word2sent = defaultdict(list)
    word_counts = defaultdict(int)
    sentences_processed = 0
    target_sentences = max_words * 100  # Process more data

    print("Processing sentences...")
    for item in tqdm(dataset, desc="Collecting"):
        text = item['text'].strip()
        if len(text) < 20:
            continue

        sentences = text.split('.')
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 10 or len(sent) > 200:
                continue

            sentences_processed += 1
            words = sent.lower().split()
            for word in words:
                word = ''.join(c for c in word if c.isalnum())
                if len(word) < 2:
                    continue
                if word_counts[word] < 50:  # More contexts for clustering
                    word2sent[word].append(sent)
                    word_counts[word] += 1

        # Check if we have enough words with sufficient contexts
        if sentences_processed >= target_sentences:
            words_with_contexts = sum(1 for w in word2sent if len(word2sent[w]) >= 10)
            if words_with_contexts >= max_words:
                break

    # Keep words with enough contexts for clustering
    filtered = {w: sents for w, sents in word2sent.items()
                if len(sents) >= 10}  # Need enough for clustering

    # First, ensure required words are included (if they have contexts)
    final = {}
    for word in REQUIRED_WORDS:
        if word in filtered:
            final[word] = filtered[word][:50]

    required_added = len(final)

    # Fill remaining slots with top frequency words
    sorted_words = sorted(filtered.keys(),
                          key=lambda w: len(filtered[w]), reverse=True)
    remaining_slots = max_words - len(final)

    for w in sorted_words:
        if w not in final and remaining_slots > 0:
            final[w] = filtered[w][:50]
            remaining_slots -= 1

    print(f"Created word2sent with {len(final)} words ({required_added} required words)")
    return final


def extract_contextual_embeddings(word2sent, model_name, n_contexts=30, use_token_level=True):
    """Extract contextual embeddings for each word in its contexts.

    If use_token_level=True: Extract the hidden state of the target word token
    If use_token_level=False: Use sentence embedding (less precise)
    """
    lazy_import_torch()

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    word_embeddings = defaultdict(list)  # word -> list of contextual embeddings

    for word, sentences in tqdm(word2sent.items(), desc="Extracting"):
        contexts = sentences[:n_contexts]
        if len(contexts) < 5:
            continue

        for sent in contexts:
            try:
                # Tokenize
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Get hidden states
                with torch.no_grad():
                    outputs = model(**inputs, output_hidden_states=True)

                # Get last layer hidden states
                if hasattr(outputs, 'last_hidden_state'):
                    hidden = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]
                else:
                    hidden = outputs.hidden_states[-1][0]

                if use_token_level:
                    # Find the target word's token position(s)
                    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                    word_lower = word.lower()

                    # Find token positions that match the word
                    word_positions = []
                    for i, tok in enumerate(tokens):
                        # Handle subword tokens (e.g., "##ing" in BERT, "▁word" in sentencepiece)
                        tok_clean = tok.replace('##', '').replace('▁', '').replace('Ġ', '').lower()
                        if tok_clean == word_lower or word_lower.startswith(tok_clean):
                            word_positions.append(i)

                    if word_positions:
                        # Average hidden states at word positions
                        word_hidden = hidden[word_positions].mean(dim=0)
                    else:
                        # Fallback: use mean pooling (exclude special tokens)
                        word_hidden = hidden[1:-1].mean(dim=0)
                else:
                    # Mean pooling over all tokens (sentence embedding)
                    word_hidden = hidden[1:-1].mean(dim=0)

                word_embeddings[word].append(word_hidden.cpu().numpy())

            except Exception as e:
                continue

    # Filter words with enough embeddings
    result = {w: embs for w, embs in word_embeddings.items() if len(embs) >= 5}
    print(f"Extracted embeddings for {len(result)} words")
    return result


def cluster_word_senses(word_embeddings, n_senses=3, min_sense_size=2):
    """Cluster contextual embeddings to find word senses."""
    print(f"Clustering word senses (k={n_senses})...")

    word_senses = {}  # word -> list of (sense_embedding, sense_size)

    for word, embs in tqdm(word_embeddings.items(), desc="Clustering"):
        embs_array = np.array(embs)

        if len(embs) < n_senses:
            # Not enough examples, just average
            word_senses[word] = [(embs_array.mean(axis=0), len(embs))]
            continue

        # K-means clustering
        actual_k = min(n_senses, len(embs))
        kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embs_array)

        # Get sense centroids and sizes
        senses = []
        for i in range(actual_k):
            mask = labels == i
            sense_size = mask.sum()
            if sense_size >= min_sense_size:
                centroid = embs_array[mask].mean(axis=0)
                senses.append((centroid, sense_size))

        # Sort by size (most common sense first)
        senses.sort(key=lambda x: -x[1])

        if not senses:
            senses = [(embs_array.mean(axis=0), len(embs))]

        word_senses[word] = senses

    return word_senses


def apply_pca_to_senses(word_senses, target_dim=256, n_components_remove=5):
    """Apply PCA and ABTT to all sense embeddings."""
    print("Applying PCA and ABTT denoising to senses...")

    # Collect all sense embeddings
    all_embs = []
    for senses in word_senses.values():
        for emb, _ in senses:
            all_embs.append(emb)

    all_embs = np.array(all_embs)
    n_samples, orig_dim = all_embs.shape

    # Ensure we have enough samples for PCA
    max_components = min(n_samples, orig_dim) - 1
    if max_components < n_components_remove + 10:
        print(f"Warning: Not enough samples ({n_samples}) for full PCA, returning as-is")
        # Just normalize and return
        all_embs = all_embs - all_embs.mean(axis=0)
        idx = 0
        result = {}
        for word, senses in word_senses.items():
            new_senses = []
            for emb, size in senses:
                new_senses.append((all_embs[idx], size))
                idx += 1
            result[word] = new_senses
        return result

    # PCA to target dimension
    n_components = min(target_dim + n_components_remove, max_components)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(all_embs)

    # Remove top components (ABTT) if we have enough
    if transformed.shape[1] > n_components_remove:
        transformed = transformed[:, n_components_remove:]
    else:
        print(f"Warning: Not enough components for ABTT, keeping all")

    # Re-center
    transformed = transformed - transformed.mean(axis=0)

    print(f"Applied PCA: {orig_dim}d -> {transformed.shape[1]}d")

    # Map back to word senses
    idx = 0
    result = {}
    for word, senses in word_senses.items():
        new_senses = []
        for emb, size in senses:
            new_senses.append((transformed[idx], size))
            idx += 1
        result[word] = new_senses

    return result


def distill_senses(word_senses, teacher_model_name, sentences,
                   max_steps=200, lr=0.01):
    """Knowledge distillation for sense embeddings."""
    lazy_import_torch()

    print(f"Loading teacher model: {teacher_model_name}...")

    from sentence_transformers import SentenceTransformer
    teacher = SentenceTransformer(teacher_model_name, trust_remote_code=True)

    # Build vocabulary
    vocab = list(word_senses.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}

    # Get dimensions
    sample_sense = list(word_senses.values())[0][0][0]
    student_dim = len(sample_sense)
    teacher_dim = teacher.get_sentence_embedding_dimension()

    print(f"Student dim: {student_dim}, Teacher dim: {teacher_dim}")

    # Initialize student embeddings (primary sense only for distillation)
    student_embs = np.array([senses[0][0] for senses in word_senses.values()])

    # Create projection layer
    np.random.seed(42)
    projection = np.random.randn(student_dim, teacher_dim) * 0.1

    print(f"Training on {len(sentences)} sentences...")
    best_loss = float('inf')

    # Store original embeddings to maintain diversity
    original_embs = student_embs.copy()

    # Use smaller learning rate to avoid collapse
    lr = lr * 0.1

    for step in tqdm(range(max_steps), desc="Distilling"):
        # Sample batch
        batch_sents = np.random.choice(sentences, size=min(32, len(sentences)),
                                       replace=False)

        # Get teacher embeddings
        teacher_embs = teacher.encode(list(batch_sents), show_progress_bar=False)

        # Compute student sentence embeddings (average of word embeddings)
        student_sent_embs = []
        valid_indices = []

        for i, sent in enumerate(batch_sents):
            words = sent.lower().split()
            word_embs = []
            for w in words:
                w = ''.join(c for c in w if c.isalnum())
                if w in word2idx:
                    word_embs.append(student_embs[word2idx[w]])

            if word_embs:
                sent_emb = np.mean(word_embs, axis=0)
                student_sent_embs.append(sent_emb)
                valid_indices.append(i)

        if len(student_sent_embs) < 2:
            continue

        student_sent_embs = np.array(student_sent_embs)
        teacher_embs = teacher_embs[valid_indices]

        # Project student to teacher dimension
        student_proj = student_sent_embs @ projection

        # Normalize
        student_norm = student_proj / (np.linalg.norm(student_proj, axis=1, keepdims=True) + 1e-8)
        teacher_norm = teacher_embs / (np.linalg.norm(teacher_embs, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity loss
        loss = 1 - np.mean(np.sum(student_norm * teacher_norm, axis=1))

        if loss < best_loss:
            best_loss = loss

        # Gradient update for student embeddings with regularization
        for i, sent in enumerate(batch_sents):
            if i not in valid_indices:
                continue
            words = sent.lower().split()
            for w in words:
                w = ''.join(c for c in w if c.isalnum())
                if w in word2idx:
                    idx = word2idx[w]
                    # Simple gradient approximation
                    grad = (student_proj[valid_indices.index(i)] - teacher_embs[valid_indices.index(i)]) @ projection.T
                    grad = grad / len(words)

                    # Regularization to maintain diversity (pull back to original)
                    reg = 0.1 * (student_embs[idx] - original_embs[idx])

                    student_embs[idx] -= lr * (grad + reg)

    print(f"Training complete. Best loss: {best_loss:.4f}")

    # Normalize all embeddings
    norms = np.linalg.norm(student_embs, axis=1, keepdims=True) + 1e-8
    student_embs = student_embs / norms

    # Update word_senses with distilled primary sense
    result = {}
    for i, (word, senses) in enumerate(word_senses.items()):
        new_senses = [(student_embs[i], senses[0][1])]  # Updated primary sense
        # Normalize secondary senses as well
        for emb, freq in senses[1:]:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            new_senses.append((emb_norm, freq))
        result[word] = new_senses

    return result


def save_multisense_embeddings(word_senses, output_path, format="json"):
    """Save multi-sense embeddings."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    if format == "json":
        # JSON format with all senses
        data = {}
        for word, senses in word_senses.items():
            data[word] = [
                {"embedding": emb.tolist(), "frequency": int(freq)}
                for emb, freq in senses
            ]

        with open(output_path, 'w') as f:
            json.dump(data, f)
    else:
        # Word2vec format (primary sense only)
        with open(output_path, 'w') as f:
            for word, senses in word_senses.items():
                emb = senses[0][0]  # Primary sense
                vec_str = ' '.join(f'{x:.6f}' for x in emb)
                f.write(f"{word} {vec_str}\n")

    print(f"Saved {len(word_senses)} words to {output_path}")


def evaluate_multisense(word_senses):
    """Evaluate multi-sense embeddings."""
    print("\nEvaluating multi-sense embeddings...")

    def get_primary(word):
        if word in word_senses:
            return word_senses[word][0][0]
        return None

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

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
    ]

    correct = 0
    total = 0

    print("\nWord Similarity Tests:")
    for w1, w2, expected in test_pairs:
        e1, e2 = get_primary(w1), get_primary(w2)
        if e1 is None or e2 is None:
            print(f"  ? {w1:10} - {w2:10}: missing word(s)")
            continue

        sim = cosine_sim(e1, e2)
        is_correct = (sim > 0.3) == expected
        correct += is_correct
        total += 1

        status = "✓" if is_correct else "✗"
        exp = "high" if expected else "low"
        print(f"  {status} {w1:10} - {w2:10}: {sim:6.3f} (expected {exp})")

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # Show polysemy examples
    print("\nPolysemy Examples (multiple senses):")
    for word in ["bank", "run", "play", "light", "spring"]:
        if word in word_senses:
            senses = word_senses[word]
            print(f"  {word}: {len(senses)} sense(s)")
            for i, (emb, freq) in enumerate(senses[:3]):
                print(f"    Sense {i+1}: freq={freq}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Sense SWE Pipeline")
    parser.add_argument('-quality', type=str, default='qwen',
                        choices=['fast', 'qwen'],
                        help='Model quality level')
    parser.add_argument('-max_words', type=int, default=500)
    parser.add_argument('-n_senses', type=int, default=3,
                        help='Maximum senses per word')
    parser.add_argument('-n_contexts', type=int, default=30,
                        help='Contexts per word for extraction')
    parser.add_argument('-max_steps', type=int, default=200)
    parser.add_argument('-output', type=str,
                        default='data/output/swe_multisense.json')
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.quality]

    print("=" * 60)
    print("MULTI-SENSE SWE PIPELINE")
    print("=" * 60)
    print(f"Model: {config['base']}")
    print(f"Max senses per word: {args.n_senses}")
    print("=" * 60)

    # Step 1: Create word2sent
    print("\n[Step 1/5] Creating word2sent data...")
    word2sent = create_word2sent(max_words=args.max_words)

    # Step 2: Extract contextual embeddings
    print("\n[Step 2/5] Extracting contextual embeddings...")
    word_embeddings = extract_contextual_embeddings(
        word2sent, config['base'], n_contexts=args.n_contexts
    )

    # Step 3: Cluster into senses
    print("\n[Step 3/5] Clustering word senses...")
    word_senses = cluster_word_senses(word_embeddings, n_senses=args.n_senses)

    # Step 4: Apply PCA
    print("\n[Step 4/5] Applying PCA and ABTT...")
    word_senses = apply_pca_to_senses(word_senses)

    # Step 5: Knowledge distillation
    print("\n[Step 5/6] Knowledge distillation...")
    sentences = []
    for sents in word2sent.values():
        sentences.extend(sents)
    sentences = list(set(sentences))[:50000]

    word_senses = distill_senses(
        word_senses, config['teacher'], sentences,
        max_steps=args.max_steps
    )

    # Step 6: Evaluate and save
    print("\n[Step 6/6] Evaluating and saving...")
    evaluate_multisense(word_senses)

    # Save both formats
    save_multisense_embeddings(word_senses, args.output, format="json")

    # Also save word2vec format for comparison
    w2v_output = args.output.replace('.json', '.txt')
    save_multisense_embeddings(word_senses, w2v_output, format="w2v")

    print("\n" + "=" * 60)
    print(f"Pipeline complete!")
    print(f"Multi-sense: {args.output}")
    print(f"Primary sense: {w2v_output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
