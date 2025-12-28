"""
High-Quality SWE training pipeline using SOTA embedding models.
Uses GTE/BGE models for extraction and distillation.
"""
import torch
import os
import numpy as np
import pickle
import argparse
import string
import re
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from scipy import stats

PUNCTLIST = set(list(string.punctuation))

# Model configurations for different quality levels
MODEL_CONFIGS = {
    "fast": {
        "base": "distilbert-base-uncased",
        "teacher": "all-MiniLM-L6-v2",
        "desc": "Fast CPU training (~10 min)"
    },
    "balanced": {
        "base": "BAAI/bge-small-en-v1.5",
        "teacher": "BAAI/bge-base-en-v1.5",
        "desc": "Good quality, reasonable speed (~20 min)"
    },
    "high": {
        "base": "BAAI/bge-base-en-v1.5",
        "teacher": "BAAI/bge-large-en-v1.5",
        "desc": "High quality BGE models (~40 min)"
    },
    "sota": {
        "base": "Alibaba-NLP/gte-base-en-v1.5",
        "teacher": "Alibaba-NLP/gte-large-en-v1.5",
        "desc": "SOTA GTE models from paper (~60 min)"
    },
    "qwen": {
        "base": "Qwen/Qwen3-Embedding-0.6B",
        "teacher": "Qwen/Qwen3-Embedding-0.6B",
        "desc": "Qwen3 0.6B - top open-source embeddings (~30 min)"
    }
}


def create_word2sent(max_words=1000, sents_per_word=50):
    """Create word2sent from wikitext dataset."""
    print("Loading wikitext dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    word_counts = defaultdict(int)
    word_sentences = defaultdict(list)

    print("Processing sentences...")
    for i, item in enumerate(tqdm(dataset, desc="Collecting")):
        text = item['text']
        if not text or len(text.strip()) < 20:
            continue

        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = ' '.join(sent.split())
            words = sent.split()
            if len(words) < 5 or len(words) > 40:
                continue

            for word in words:
                word_lower = word.lower()
                if word_lower.isalpha() and len(word_lower) > 2:
                    word_counts[word_lower] += 1
                    if len(word_sentences[word_lower]) < sents_per_word:
                        word_sentences[word_lower].append(sent)

        if i >= 200000:
            break

    # Get top words with enough sentences
    top_words = sorted(word_counts.items(), key=lambda x: -x[1])
    word2sent = {}
    for word, count in top_words:
        if len(word_sentences[word]) >= 10:
            word2sent[word] = word_sentences[word][:sents_per_word]
            if len(word2sent) >= max_words:
                break

    print(f"Created word2sent with {len(word2sent)} words")
    return word2sent


def extract_embeddings(word2sent, model_name, nsent=20):
    """Extract contextual embeddings for each word."""
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    word2vec = {}

    with torch.no_grad():
        for word, sentences in tqdm(word2sent.items(), desc="Extracting"):
            sents = sentences[:nsent]

            # Handle different tokenizer types
            try:
                word_ids = tokenizer(" " + word, add_special_tokens=False)["input_ids"]
            except:
                word_ids = tokenizer.encode(" " + word, add_special_tokens=False)

            if len(word_ids) == 0:
                continue

            embeddings = []
            for sent in sents:
                try:
                    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
                    outputs = model(**inputs)
                    hidden = outputs.last_hidden_state[0]  # [seq_len, dim]

                    # Find word position
                    input_ids = inputs["input_ids"][0].tolist()
                    for j in range(len(input_ids) - len(word_ids) + 1):
                        if input_ids[j:j+len(word_ids)] == word_ids:
                            emb = hidden[j:j+len(word_ids)].mean(dim=0)
                            embeddings.append(emb.numpy())
                            break
                except Exception as e:
                    continue

            if embeddings:
                word2vec[word] = np.mean(embeddings, axis=0)

    print(f"Extracted embeddings for {len(word2vec)} words")
    return word2vec, tokenizer


def apply_pca_abtt(word2vec, word2sent, tokenizer, d_remove=5, output_dim=256):
    """Apply PCA and All-But-The-Top (ABTT) denoising."""
    print("Generating sentence embeddings for PCA...")

    # Collect all sentences
    all_sents = []
    for word, sents in word2sent.items():
        all_sents.extend(sents[:10])
    all_sents = all_sents[:10000]  # Cap for memory

    # Encode sentences
    edim = len(list(word2vec.values())[0])
    sent_embs = []

    for sent in tqdm(all_sents, desc="Encoding sentences"):
        try:
            words = [x[0] for x in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)]
        except:
            words = sent.lower().split()
        embs = [word2vec[w.lower()] for w in words if w.lower() in word2vec and w not in PUNCTLIST]
        if embs:
            sent_embs.append(np.mean(embs, axis=0))

    sent_embs = np.array(sent_embs)
    print(f"Collected {len(sent_embs)} sentence embeddings")

    # PCA for dimensionality reduction
    n_components = min(output_dim, sent_embs.shape[0] - 1, sent_embs.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(sent_embs)

    # Transform word embeddings
    word_matrix = np.array(list(word2vec.values()))
    word_names = list(word2vec.keys())

    word_pca = pca.transform(word_matrix)

    # ABTT: Remove top d components (they capture frequency, not semantics)
    if d_remove > 0 and d_remove < word_pca.shape[1]:
        u, s, vt = np.linalg.svd(word_pca, full_matrices=False)
        word_pca = word_pca - (word_pca @ vt[:d_remove].T) @ vt[:d_remove]

    # Normalize
    norms = np.linalg.norm(word_pca, axis=1, keepdims=True)
    word_pca = word_pca / (norms + 1e-8)

    new_word2vec = {name: vec for name, vec in zip(word_names, word_pca)}
    print(f"Applied PCA: {edim}d -> {word_pca.shape[1]}d, removed top {d_remove} components")

    return new_word2vec


def train_distillation(word2vec, tokenizer, teacher_model_name, epochs=2, batch_size=32, max_steps=500):
    """Train embeddings via knowledge distillation from teacher model."""
    print(f"Loading teacher model: {teacher_model_name}...")
    teacher = SentenceTransformer(teacher_model_name, trust_remote_code=True)
    teacher_dim = teacher.get_sentence_embedding_dimension()

    # Load training sentences
    print("Loading training sentences...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    train_sents = []
    for item in dataset:
        text = item['text']
        if text and len(text.strip()) > 20:
            for sent in re.split(r'[.!?]+', text):
                sent = ' '.join(sent.split())
                if 10 < len(sent) < 200:
                    train_sents.append(sent)
                    if len(train_sents) >= 50000:
                        break
        if len(train_sents) >= 50000:
            break

    print(f"Using {len(train_sents)} training sentences")

    # Convert to numpy for training
    vocab = list(word2vec.keys())
    vocab_to_idx = {w: i for i, w in enumerate(vocab)}
    edim = len(list(word2vec.values())[0])
    embeddings = np.array([word2vec[w] for w in vocab])

    # Create projection layer: student_dim -> teacher_dim
    np.random.seed(42)
    projection = np.random.randn(edim, teacher_dim) * 0.1
    print(f"Student dim: {edim}, Teacher dim: {teacher_dim}")

    # Training loop
    lr = 0.01
    best_loss = float('inf')
    best_embeddings = embeddings.copy()

    step = 0
    for epoch in range(epochs):
        np.random.shuffle(train_sents)
        losses = []

        pbar = tqdm(range(0, len(train_sents), batch_size), desc=f"Epoch {epoch+1}")
        for i in pbar:
            if step >= max_steps:
                break

            batch = train_sents[i:i+batch_size]

            # Get teacher embeddings
            with torch.no_grad():
                teacher_embs = teacher.encode(batch, convert_to_numpy=True)

            # Get student embeddings (average of word vectors)
            student_embs = []
            valid_indices = []

            for j, sent in enumerate(batch):
                try:
                    words = [x[0].lower() for x in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)]
                except:
                    words = sent.lower().split()
                word_indices = [vocab_to_idx[w] for w in words if w in vocab_to_idx and w not in PUNCTLIST]

                if word_indices:
                    emb = embeddings[word_indices].mean(axis=0)
                    student_embs.append(emb)
                    valid_indices.append(j)

            if not student_embs:
                continue

            student_embs = np.array(student_embs)
            teacher_embs = teacher_embs[valid_indices]

            # Project student to teacher dimension
            student_projected = student_embs @ projection

            # Normalize
            student_norm = student_projected / (np.linalg.norm(student_projected, axis=1, keepdims=True) + 1e-8)
            teacher_norm = teacher_embs / (np.linalg.norm(teacher_embs, axis=1, keepdims=True) + 1e-8)

            # MSE loss and gradient
            loss = np.mean((student_norm - teacher_norm) ** 2)
            losses.append(loss)

            # Simple gradient update for embeddings and projection
            diff = student_norm - teacher_norm
            for j, sent in enumerate(batch[:len(valid_indices)]):
                try:
                    words = [x[0].lower() for x in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)]
                except:
                    words = sent.lower().split()
                word_indices = [vocab_to_idx[w] for w in words if w in vocab_to_idx and w not in PUNCTLIST]

                if word_indices:
                    # Backprop through projection
                    grad_proj = diff[j].reshape(1, -1)
                    grad_emb = (grad_proj @ projection.T).flatten() / len(word_indices)
                    for idx in word_indices:
                        embeddings[idx] -= lr * grad_emb

            # Update projection layer
            projection -= lr * 0.1 * (student_embs.T @ diff) / len(student_embs)

            step += 1
            if step % 50 == 0:
                avg_loss = np.mean(losses[-50:])
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_embeddings = embeddings.copy()

        if step >= max_steps:
            break

    print(f"Training complete. Best loss: {best_loss:.4f}")

    # Normalize final embeddings
    norms = np.linalg.norm(best_embeddings, axis=1, keepdims=True)
    best_embeddings = best_embeddings / (norms + 1e-8)

    return {w: best_embeddings[i] for i, w in enumerate(vocab)}


def evaluate_embeddings(word2vec, tokenizer):
    """Quick evaluation using word similarity."""
    print("\nEvaluating embeddings...")

    test_pairs = [
        ("king", "queen", True),
        ("man", "woman", True),
        ("good", "excellent", True),
        ("big", "large", True),
        ("computer", "laptop", True),
        ("computer", "banana", False),
        ("king", "computer", False),
    ]

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    print("\nWord Similarity Tests:")
    correct = 0
    total = 0
    for w1, w2, should_be_similar in test_pairs:
        if w1 in word2vec and w2 in word2vec:
            sim = cosine_sim(word2vec[w1], word2vec[w2])
            is_correct = (sim > 0.3) == should_be_similar
            status = "✓" if is_correct else "✗"
            print(f"  {status} {w1} - {w2}: {sim:.3f} (expected {'high' if should_be_similar else 'low'})")
            correct += is_correct
            total += 1
        else:
            print(f"  ? {w1} - {w2}: missing word(s)")

    if total > 0:
        print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.1f}%)")


def save_embeddings(word2vec, output_path):
    """Save embeddings in word2vec text format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for word, vec in word2vec.items():
            vec_str = ' '.join(f'{v:.6f}' for v in vec)
            f.write(f"{word} {vec_str}\n")
    print(f"Saved {len(word2vec)} embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="High-Quality SWE Training Pipeline")
    parser.add_argument('-quality', type=str, default='balanced',
                        choices=['fast', 'balanced', 'high', 'sota', 'qwen'],
                        help='Quality level: fast, balanced, high, or sota')
    parser.add_argument('-max_words', type=int, default=1000, help='Max vocabulary size')
    parser.add_argument('-nsent', type=int, default=20, help='Sentences per word for extraction')
    parser.add_argument('-model', type=str, default=None, help='Override base model')
    parser.add_argument('-teacher', type=str, default=None, help='Override teacher model')
    parser.add_argument('-d_remove', type=int, default=5, help='PCA components to remove')
    parser.add_argument('-embd', type=int, default=256, help='Output embedding dimension')
    parser.add_argument('-epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('-max_steps', type=int, default=500, help='Max training steps')
    parser.add_argument('-output', type=str, default=None, help='Output path')
    args = parser.parse_args()

    # Get model configuration
    config = MODEL_CONFIGS[args.quality]
    base_model = args.model or config["base"]
    teacher_model = args.teacher or config["teacher"]

    if args.output is None:
        args.output = f"data/output/swe_{args.quality}_{args.max_words}.txt"

    print("=" * 60)
    print("SWE4SEMANTICS: High-Quality Training Pipeline")
    print("=" * 60)
    print(f"Quality level: {args.quality} - {config['desc']}")
    print(f"Base model:    {base_model}")
    print(f"Teacher model: {teacher_model}")
    print("=" * 60)

    # Step 1: Create word2sent data
    print("\n[Step 1/5] Creating word2sent data...")
    word2sent = create_word2sent(max_words=args.max_words, sents_per_word=50)

    # Step 2: Extract embeddings
    print("\n[Step 2/5] Extracting contextual embeddings...")
    word2vec, tokenizer = extract_embeddings(word2sent, base_model, nsent=args.nsent)

    # Step 3: Apply PCA + ABTT
    print("\n[Step 3/5] Applying PCA and ABTT denoising...")
    word2vec = apply_pca_abtt(word2vec, word2sent, tokenizer, d_remove=args.d_remove, output_dim=args.embd)

    # Step 4: Knowledge distillation
    print("\n[Step 4/5] Training via knowledge distillation...")
    word2vec = train_distillation(word2vec, tokenizer, teacher_model, epochs=args.epochs, max_steps=args.max_steps)

    # Step 5: Evaluate and save
    print("\n[Step 5/5] Evaluating and saving...")
    evaluate_embeddings(word2vec, tokenizer)
    save_embeddings(word2vec, args.output)

    print("\n" + "=" * 60)
    print(f"Pipeline complete! Output: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
