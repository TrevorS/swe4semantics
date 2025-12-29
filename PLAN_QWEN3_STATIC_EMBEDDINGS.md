# Plan: High-Quality Static Word Embeddings via Qwen3-8B Distillation

## Executive Summary

We propose building state-of-the-art static word embeddings by distilling contextual representations from Qwen3-Embedding-8B into a 150k-word vocabulary. This combines the quality of large transformer models with the speed and simplicity of static embeddings.

**Expected outcome**: Static embeddings achieving ~62-65 MTEB score, usable on CPU with instant inference.

---

## 1. Why Static Embeddings?

### 1.1 The Speed-Quality Tradeoff

| Model Type | Inference | Memory | Quality |
|------------|-----------|--------|---------|
| Qwen3-8B | ~100ms/sentence (GPU) | 16 GB | High |
| Static (ours) | <1ms/sentence (CPU) | ~150 MB | Medium-High |

**Use cases favoring static embeddings:**
- Edge devices / mobile
- High-throughput systems (millions of queries/sec)
- Resource-constrained environments
- Interpretability (one vector per word)

### 1.2 Prior Art

| Paper | Approach | Limitation |
|-------|----------|------------|
| Word2Vec (Mikolov 2013) | Predict context words | No transformer knowledge |
| GloVe (Pennington 2014) | Co-occurrence matrix | No transformer knowledge |
| BERT Wears GloVes (Bommasani 2019) | Average BERT contexts | BERT is now outdated |
| Model2Vec/POTION (2024) | Single-pass distillation | Loses contextual nuance |
| **SWE4Semantics (2025)** | **Multi-context + PCA + KD** | Uses GTE, not latest models |

**Our contribution**: Apply SWE4Semantics methodology with Qwen3-Embedding-8B as the teacher model.

---

## 2. Why Qwen3-Embedding-8B?

### 2.1 Model Selection Rationale

| Model | Parameters | MTEB Score | Reason |
|-------|------------|------------|--------|
| GTE-base (SWE4Semantics) | 110M | ~56 | Original paper choice |
| BGE-large | 335M | ~64 | Good but older |
| E5-mistral-7b | 7B | ~66 | Strong but instruction-tuned |
| **Qwen3-Embedding-8B** | 8B | ~70+ | Latest SOTA, multilingual |

**Key advantages of Qwen3-Embedding-8B:**
1. **State-of-the-art quality**: Top performance on MTEB leaderboard
2. **Large capacity**: 8B parameters capture rich semantics
3. **Multilingual**: Can extend to cross-lingual embeddings later
4. **Trust_remote_code**: Easy to use with HuggingFace

### 2.2 References

- Qwen3 Technical Report: https://arxiv.org/abs/2505.09388
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard

---

## 3. Methodology

### 3.1 Overview (Based on SWE4Semantics)

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Build word2sent mapping                                │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ CC-100      │ -> │ Word tokenize │ -> │ Top 150k words    │  │
│  │ Corpus      │    │ (BERT pre-tok)│    │ + 100 sents each  │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: Extract contextual embeddings                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ For each    │ -> │ Qwen3-8B     │ -> │ Extract word's    │  │
│  │ word+context│    │ forward pass │    │ hidden state      │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 3: Aggregate to static embedding                         │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ 100 context │ -> │ Average      │ -> │ 1 static vector   │  │
│  │ embeddings  │    │ pooling      │    │ per word          │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 4: Post-processing                                       │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ Raw 4096d   │ -> │ Sentence PCA │ -> │ 256d final        │  │
│  │ embeddings  │    │ (remove bias)│    │ embeddings        │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 5: Knowledge Distillation (Optional)                     │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │ Static embs │ -> │ Match Qwen3  │ -> │ Fine-tuned        │  │
│  │ (student)   │    │ sent scores  │    │ embeddings        │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Step-by-Step Details

#### Step 1: Vocabulary Construction

**Source corpus**: CC-100 English (Common Crawl, cleaned)
- ~50GB of text
- Sentence-split using BlingFire

**Tokenization**: BERT's pre-tokenizer (word-level, not subword)
```python
from transformers import AutoTokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
word_tokenizer = bert_tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str
```

**Why words, not subwords?**
- SWE4Semantics paper shows word-level outperforms subword for sentence tasks
- POTION uses subwords but sacrifices quality
- Classic embeddings (Word2Vec, GloVe) are word-level

**Vocabulary selection**:
- Count word frequencies across corpus
- Take top 150,000 words
- Filter: ≥10 occurrences, ≥3 characters, alphabetic

**Reference**: SWE4Semantics Section 3.1

#### Step 2: Context Collection

For each word in vocabulary:
- Find 100 sentences containing the word
- Sentences, not paragraphs (better context focus)
- Shuffle to get diverse contexts

**Why 100 contexts?**
- SWE4Semantics tested N=50, 100, 200
- N=100 is sweet spot (diminishing returns after)
- BERT Wears GloVes used N=1,000,000 (overkill)

**Reference**:
- SWE4Semantics Section 4.2, Table 3
- BERT Wears GloVes Section 4.1

#### Step 3: Contextual Embedding Extraction

**Model**: Qwen/Qwen3-Embedding-8B

**Extraction method**: Token-level (not sentence-level)
```python
# Tokenize sentence
inputs = tokenizer(sentence, return_tensors="pt")

# Forward pass
outputs = model(**inputs, output_hidden_states=True)
hidden = outputs.last_hidden_state  # [1, seq_len, 4096]

# Find target word's token positions
word_positions = find_word_tokens(tokens, target_word)

# Extract and average word's hidden states
word_embedding = hidden[0, word_positions].mean(dim=0)  # [4096]
```

**Why token-level, not sentence-level?**
- Sentence embedding conflates all words
- Token-level isolates the target word's representation
- Our experiments showed this matters (77.8% vs lower accuracy)

**Which layer?**
- BERT Wears GloVes found early layers (0-3) best for word similarity
- But for sentence tasks, later layers may be better
- We use last layer (matches SWE4Semantics)

**Reference**:
- BERT Wears GloVes Section 4.2, Figure 2
- SWE4Semantics uses last layer

#### Step 4: Aggregation

**Method**: Mean pooling across contexts
```python
# For each word
word_contexts = [emb1, emb2, ..., emb100]  # 100 x 4096
static_embedding = np.mean(word_contexts, axis=0)  # 4096
```

**Why mean pooling?**
- Simple and effective
- Robust to outlier contexts
- SWE4Semantics validated this choice

**Alternative (Multi-sense)**:
```python
# K-means clustering for polysemous words
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(word_contexts)
sense_embeddings = [contexts[labels==i].mean(0) for i in range(3)]
```

We can optionally produce multi-sense embeddings for polysemous words.

#### Step 5: Sentence-level PCA

**Purpose**: Remove dominant directions that encode sentence-level bias

**Method**:
1. Sample 10,000 sentences from corpus
2. Encode with static embeddings (average word vectors)
3. Compute PCA on sentence embeddings
4. Remove top-k principal components from word embeddings

```python
# Remove top 7 components (SWE4Semantics default)
pca = PCA(n_components=7)
pca.fit(sentence_embeddings)

# Project out dominant directions
for word in vocabulary:
    for i in range(7):
        word_emb -= (word_emb @ pca.components_[i]) * pca.components_[i]
```

**Why remove top components?**
- Top PCs often encode frequency, position, or other non-semantic info
- Similar to "all-but-the-top" technique in sentence embeddings
- SWE4Semantics shows this improves sentence similarity tasks

**Reference**:
- SWE4Semantics Section 3.2
- "All-but-the-Top" (Mu & Viswanath 2018)

#### Step 6: Dimensionality Reduction

**Reduce from 4096d to 256d**:
```python
pca = PCA(n_components=256)
reduced_embeddings = pca.fit_transform(raw_embeddings)
```

**Why 256d?**
- Good balance of quality vs size
- SWE4Semantics uses 256d
- 150k words × 256d × 4 bytes = ~150 MB

#### Step 7: Knowledge Distillation (Optional)

**Purpose**: Fine-tune static embeddings to match Qwen3 sentence scores

**Method**:
1. Sample sentence pairs
2. Compute Qwen3 similarity (teacher)
3. Compute static embedding similarity (student)
4. Minimize MSE loss

```python
for sent1, sent2 in pairs:
    # Teacher score
    teacher_score = cosine(qwen3.encode(sent1), qwen3.encode(sent2))

    # Student score
    student_score = cosine(static_encode(sent1), static_encode(sent2))

    # Loss
    loss = (teacher_score - student_score) ** 2
```

**Reference**: SWE4Semantics Section 3.3

---

## 4. Implementation Plan

### 4.1 Compute Requirements

| Step | GPU Hours | Storage |
|------|-----------|---------|
| Corpus preparation | 0 (CPU) | ~100 GB |
| Embedding extraction | ~200 hrs (A100) | ~50 GB |
| PCA + reduction | 0 (CPU) | ~5 GB |
| Knowledge distillation | ~20 hrs (A100) | - |
| **Total** | **~220 GPU hours** | **~155 GB** |

**With 8x A100 cluster**: ~1 week total

### 4.2 Optimization Strategies

1. **Batch processing**: Process multiple words per GPU batch
2. **Gradient checkpointing**: Reduce memory for 8B model
3. **FP16/BF16**: Half precision for faster inference
4. **Caching**: Save intermediate embeddings to disk

### 4.3 Phased Approach

| Phase | Vocabulary | Purpose |
|-------|------------|---------|
| Phase 1 | 10k words | Validate pipeline, tune hyperparameters |
| Phase 2 | 50k words | Benchmark against baselines |
| Phase 3 | 150k words | Full production model |

---

## 5. Evaluation Plan

### 5.1 Benchmarks

| Benchmark | Task | Metric |
|-----------|------|--------|
| SimLex-999 | Word similarity | Spearman ρ |
| WordSim-353 | Word relatedness | Spearman ρ |
| SCWS | Contextual word similarity | Spearman ρ |
| STS Benchmark | Sentence similarity | Spearman ρ |
| MTEB | Multi-task | Average score |

### 5.2 Baselines

| Model | Type | Expected Score |
|-------|------|----------------|
| GloVe 300d | Static | ~45 MTEB |
| Word2Vec 300d | Static | ~44 MTEB |
| POTION-32M | Static (distilled) | ~51 MTEB |
| SWE-GTE-256d | Static (distilled) | ~58 MTEB |
| **Ours (Qwen3-8B)** | Static (distilled) | **~62-65 MTEB** |

### 5.3 Ablations

1. **Context count**: N = 50, 100, 200, 500
2. **Layer selection**: Last layer vs middle layers
3. **PCA components removed**: k = 0, 3, 7, 15
4. **Output dimension**: 128d, 256d, 512d
5. **With/without knowledge distillation**

---

## 6. Extensions (Future Work)

### 6.1 Multi-sense Embeddings

For polysemous words, produce multiple sense vectors:
```
bank_1: [financial sense vector]
bank_2: [river sense vector]
bank_3: [turning sense vector]
```

**Method**: K-means clustering on contextual embeddings

**Reference**:
- "Word Sense Induction with Knowledge Distillation from BERT" (2023)
- Our multi-sense experiments (77.8% word similarity)

### 6.2 Cross-lingual Embeddings

Extend to multilingual using parallel corpora:
- English-German
- English-Chinese
- English-Japanese

**Reference**: SWE4Semantics Section 5 (cross-lingual)

### 6.3 Automatic Sense Count

Instead of fixed k=3, learn optimal sense count per word:
- Based on cluster quality metrics (silhouette score)
- Or based on WordNet sense count

---

## 7. References

1. **SWE4Semantics**: Wada et al. "Static Word Embeddings for Sentence Semantic Representation" EMNLP 2025. https://arxiv.org/abs/2506.04624

2. **BERT Wears GloVes**: Bommasani et al. "Distilling Static Embeddings from Pretrained Contextual Representations" 2019. https://openreview.net/forum?id=SJg3T2EFvr

3. **Model2Vec/POTION**: MinishLab. "Fast State-of-the-Art Static Embeddings" 2024. https://github.com/MinishLab/model2vec

4. **WSI from BERT**: "Word Sense Induction with Knowledge Distillation from BERT" 2023. https://arxiv.org/abs/2304.10642

5. **Word2Vec**: Mikolov et al. "Efficient Estimation of Word Representations in Vector Space" 2013.

6. **GloVe**: Pennington et al. "GloVe: Global Vectors for Word Representation" 2014.

7. **All-but-the-Top**: Mu & Viswanath. "All-but-the-Top: Simple and Effective Postprocessing for Word Representations" 2018.

8. **Qwen3**: Qwen Team. "Qwen3 Technical Report" 2025. https://arxiv.org/abs/2505.09388

---

## 8. Timeline

| Week | Task |
|------|------|
| 1 | Corpus preparation, word2sent pipeline |
| 2-3 | Phase 1: 10k word extraction + validation |
| 4-5 | Phase 2: 50k word extraction + benchmarking |
| 6-8 | Phase 3: Full 150k extraction |
| 9 | PCA, dimensionality reduction |
| 10 | Knowledge distillation |
| 11 | Evaluation + ablations |
| 12 | Documentation + release |

---

## 9. Deliverables

1. **swe_qwen3_256d_en.txt** - 150k English word embeddings (256d)
2. **swe_qwen3_multisense_en.json** - Multi-sense version (3 senses/word)
3. **Training code** - Full pipeline for reproduction
4. **Evaluation scripts** - Benchmarking on standard tasks
5. **Documentation** - Usage examples, API

---

## Appendix A: Quick Start (After Training)

```python
from util import load_w2v, encode_text
from transformers import AutoTokenizer

# Load embeddings
word2vec, dim = load_w2v("swe_qwen3_256d_en.txt")

# Setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
word_tokenizer = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str

# Encode sentences
emb1 = encode_text("The bank approved my loan.", word_tokenizer, word2vec, dim)
emb2 = encode_text("I deposited money at the bank.", word_tokenizer, word2vec, dim)

# Compute similarity
similarity = emb1 @ emb2.T
print(f"Similarity: {similarity:.3f}")
```
