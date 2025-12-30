# swe4semantics

This repository provides code and static word embeddings (SWEs) proposed in our paper "[Static Word Embeddings for Sentence Semantic Representation](https://aclanthology.org/2025.emnlp-main.316)" (EMNLP 25 Main).

## Quick Start (New Pipeline)

This repository now includes a modernized **uv-powered pipeline** that automates the full training process using Qwen3-Embedding models.

### Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Run the Pipeline

```bash
# Quick test (0.6B model, 1k vocab, ~15 min on CPU)
uv run python scripts/run_pipeline.py --mode test --output-dir outputs/test

# Production run (8B model, 150k vocab, requires GPU)
uv run python scripts/run_pipeline.py --mode production --output-dir outputs/prod

# With Weights & Biases experiment tracking
uv run python scripts/run_pipeline.py --mode test --output-dir outputs/test --wandb
```

### Using the Makefile

```bash
make install      # Install dependencies
make run-test     # Run test pipeline
make run-prod     # Run production pipeline
make check        # Run linting, formatting, and tests
```

### Using the Static Encoder

```python
from qwen3_static_embeddings import StaticEncoder, load_word2vec_text

# Load embeddings
embeddings = load_word2vec_text("outputs/test/exports/qwen3_static.w2v.txt")

# Create encoder
encoder = StaticEncoder(word2vec=embeddings, dim=256, normalize=True)

# Encode text
embedding = encoder.encode("machine learning is amazing")

# Batch encoding
texts = ["hello world", "deep learning", "natural language processing"]
embeddings = encoder.encode_batch(texts)

# Compute similarity
similarity = encoder.similarity("cat", "dog")
```

### Pipeline Steps

1. **Corpus Download**: Streams CC-100 English corpus via HuggingFace
2. **Vocabulary Building**: Builds word frequency vocabulary with BlingFire
3. **Word2Sent Mapping**: Maps vocabulary words to context sentences
4. **Embedding Extraction**: Extracts embeddings from Qwen3-Embedding
5. **PCA Post-processing**: Applies "All-but-the-Top" dimensionality reduction
6. **Export**: Saves in Word2Vec, GloVe, and NumPy formats
7. **Evaluation**: Runs word similarity benchmarks

### Export Formats

| Format | File | Compatible With |
|--------|------|-----------------|
| Word2Vec Text | `*.w2v.txt` | gensim, fastText |
| Word2Vec Binary | `*.w2v.bin` | gensim (faster) |
| GloVe Text | `*.glove.txt` | GloVe tools |
| NumPy | `*.npz` | NumPy, scikit-learn |

### Model Comparison

Compare your trained embeddings against baselines:

```bash
# Compare against Model2Vec baselines (e.g., potion-base-8M)
uv run python scripts/compare_models.py \
    --static outputs/test/exports/qwen3_static.w2v.txt \
    --baseline minishlab/potion-base-8M \
    --word-only

# Compare against original transformer model
uv run python scripts/compare_models.py \
    --static outputs/test/exports/qwen3_static.w2v.txt \
    --model Qwen/Qwen3-Embedding-0.6B
```

### Tokenlearn Training (POTION-style)

For higher quality embeddings, use the full [POTION](https://minishlab.github.io/tokenlearn_blogpost/) pipeline which includes:

1. **Distillation**: Create base Model2Vec model
2. **Featurization**: Generate teacher embeddings on C4 corpus
3. **Training**: Minimize cosine distance between static and teacher embeddings
4. **Post-processing**: Apply SIF weighting and remove principal components

```bash
# Step 1: Create a base Model2Vec model
uv run python -c "
from model2vec.distill import distill
model = distill('Qwen/Qwen3-Embedding-0.6B', pca_dims=256,
                pooling='last', trust_remote_code=True)
model.save_pretrained('outputs/model2vec_base')
"

# Step 2: Generate teacher embeddings using C4 dataset
uv run python scripts/run_tokenlearn.py featurize \
    --use-c4 \
    --output outputs/tokenlearn/features \
    --model Qwen/Qwen3-Embedding-0.6B \
    --max-sentences 1000000

# Step 3: Train with Tokenlearn
uv run python scripts/run_tokenlearn.py train \
    --features outputs/tokenlearn/features \
    --base-model outputs/model2vec_base \
    --output outputs/tokenlearn/trained \
    --epochs 10

# Step 4: Post-process (SIF weighting + remove principal component)
uv run python scripts/run_tokenlearn.py post-process \
    --model outputs/tokenlearn/trained \
    --features outputs/tokenlearn/features \
    --base-model outputs/model2vec_base \
    --output outputs/tokenlearn/processed

# Step 5: Export to Word2Vec format
uv run python scripts/run_tokenlearn.py export \
    --model outputs/tokenlearn/processed \
    --base-model outputs/model2vec_base \
    --output outputs/tokenlearn/final
```

**Post-processing details:**
- **SIF Weighting**: Reweights token embeddings based on corpus frequency: `w = a / (a + prob)` where `a=1e-3`
- **Remove Principal Component**: Removes the first principal component (All-but-the-Top) to improve isotropy
- **L2 Normalization**: Final normalization for cosine similarity

The Tokenlearn approach trains the static model to minimize cosine distance between its outputs and the teacher's mean embeddings, resulting in embeddings that better capture sentence-level semantics.

### NVIDIA GPU Optimization (DGX Spark)

The Tokenlearn pipeline includes automatic optimizations for NVIDIA GPUs:

- **TF32**: Enabled for faster matmul on Ampere+ GPUs (A100, H100)
- **BFloat16**: Uses BF16 instead of FP16 for better numerical stability
- **Flash Attention 2**: Automatic support when available
- **cuDNN Benchmark**: Auto-tuning for optimal convolution algorithms
- **AMP with GradScaler**: Mixed precision training with gradient scaling

For production training on DGX Spark:

```bash
# Full production run (1M sentences, ~2-4 hours on A100)
uv run python scripts/run_tokenlearn.py featurize \
    --use-c4 \
    --output outputs/tokenlearn/features \
    --model Qwen/Qwen3-Embedding-0.6B \
    --max-sentences 1000000 \
    --batch-size 64 \
    --device cuda

# Train with larger batch size (benefits from GPU memory)
uv run python scripts/run_tokenlearn.py train \
    --features outputs/tokenlearn/features \
    --base-model outputs/model2vec_base \
    --output outputs/tokenlearn/trained \
    --epochs 10 \
    --batch-size 512 \
    --device cuda

# Post-process (CPU is fine, fast operation)
uv run python scripts/run_tokenlearn.py post-process \
    --model outputs/tokenlearn/trained \
    --features outputs/tokenlearn/features \
    --base-model outputs/model2vec_base \
    --output outputs/tokenlearn/processed
```

---


# How to Use SWEs for Encoding Sentences
English and cross-lingual (English-{German/Japanese/Chinese}) SWEs are stored in the "embeddings" folder.  **Code and SWE models, except for the English-Japanese one ("swe_mgte256_enja.txt"), are released under the Apache license 2.0. The English-Japanese one follows the license [JParaCrawl](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/) (placed at "embeddings/LICENSE_swe_mgte256_enja.txt").**

Refer to "example.py" for how to use English SWEs, and "example_xling.py" for cross-lignual ones. As denoted in the paper title, these embeddings are more effective for encoding sentences than long text like paragraphs/documents.

# Train English SWEs
First, prepare the **"word2sent.pkl"** file that pickles the Python dictionary where keys are a list of words in (pre-defined) vocabulary and values are a list of N unlabelled sentences (**not passages or documents**) that contain the key word (e.g. "good": ["I have good news.", "He is a good student.", "That sounds good.", ...] ). In our paper, we employ [CC-100](https://data.statmt.org/cc-100/) and split text in each line into sentences using [BlingFire](https://github.com/microsoft/BlingFire), and then sample N=100 sentences for each word in the 150k vocab.

**Note that some pieces of code are hard-coded for the BERT-style tokenisation that specifies the subword boundary with "##". Modify relevant parts if necessary.**

## 1. Extract English SWEs from GTE-base
```
model=Alibaba-NLP/gte-base-en-v1.5
word2sent=path_to_word2sent.pkl
output_folder=output_folder_path
nsent=100
CUDA_VISIBLE_DEVICES=0 python extract_embs.py  -prompt "" -output_folder ${output_folder} -model ${model} -word2sent ${word2sent}  -nsent ${nsent}
```

## 2. Apply Sentence-level PCA
```
model="Alibaba-NLP/gte-base-en-v1.5"
word2sent=path_to_word2sent.pkl
vec_path=output_folder_path/vec.txt
output_folder=output_pca_folder_path
python apply_pca.py -d_remove 7 -embd 256 -word2sent ${word2sent} -vec_path ${vec_path} -model ${model} -output_folder ${output_folder} 
```

## 3. Fine-tune SWEs with Knowledge Distillation
```
model="Alibaba-NLP/gte-base-en-v1.5"
word2sent=path_to_word2sent.pkl
vec_path=output_pca_folder_path/vec.txt
output_folder=final_output_folder_path
CUDA_VISIBLE_DEVICES=0 python train.py -prompt "" -word2sent ${word2sent} -epoch 15 -bs 128  -model ${model} -vec_path ${vec_path} -output_folder ${output_folder}
```

# Train Cross-lingual SWEs
As in monolingual SWEs, prepare the "word2sent.pkl" file that pickles a python dictionary where keys are a list of words in a pre-defined vocabulary and values are a list of N unlabelled sentences (**not passages or documents**) that contain the key word. In our paper, we use  [CCMatrix](https://opus.nlpl.eu/CCMatrix/corpus/version/CCMatrix) and sample N=100 sentences for each word.

**Note that some pieces of code are hard-coded for language pairs used in our paper (en-de, en-zh, en-ja); modify relevant parts if necessary.**

## 1. Extract English and German SWEs Separately from mGTE-base
```
model=Alibaba-NLP/gte-multilingual-base
word2sent_en=path_to_english_word2sent
folder=output_english_folder_path
CUDA_VISIBLE_DEVICES=0 python extract_embs.py -prompt "" -folder ${folder} -model ${model} -word2sent ${word2sent_en}  -nsent 100 

word2sent_de=path_to_german_word2sent
folder=output_german_folder_path
CUDA_VISIBLE_DEVICES=0 python extract_embs.py  -prompt "" -folder ${folder} -model ${model} -word2sent ${word2sent_de} -nsent 100 
```

(If the input language is Japanese/Chinese, enable the "-subword" option)

## 2. Merge SWEs and Apply Sentence-level PCA

**Generate English-German SWEs**
```
langs="en de"
vec_path="output_english_folder_path/vec.txt output_german_folder_path/vec.txt"
model="Alibaba-NLP/gte-multilingual-base"
word2sent="${word2sent_en} ${word2sent_de}"
output_folder=output_pca_folder_path
python apply_pca_xling.py -d_remove 7 -embd 256 -langs ${langs} -word2sent ${word2sent}  -vec_path ${vec_path} -model ${model}  -output_folder ${output_folder}
```

**You can also generate multilingual SWEs as follows** (e.g. SWEs aligned across English, German, Chinese, and Japanese, which are evaluated in Table 10 and 11 in the paper).
```
langs="en de zh ja"
vec_path="output_english_folder_path/vec.txt output_german_folder_path/vec.txt output_chinese_folder_path/vec.txt output_japanese_folder_path/vec.txt"
model="Alibaba-NLP/gte-multilingual-base"
word2sent="${word2sent_en} ${word2sent_de} ${word2sent_zh} ${word2sent_ja}"
output_folder=output_pca_folder_path
python apply_pca_xling.py -d_remove 7 -embd 256 -langs ${langs} -word2sent ${word2sent}  -vec_path ${vec_path} -model ${model}  -output_folder ${output_folder}
```

## 3. Fine-tune SWEs with Contrastive Learning
Prepare **"en.txt" and "de.txt"**, where each line is a sentence that is parallel (translation) to each language (hence, both files must have the same numbner of lines). These files are used for contrastive learning. In our paper, we use [CCMatrix](https://opus.nlpl.eu/CCMatrix/corpus/version/CCMatrix) as in Step 1.

```
vec_path=output_pca_folder_path/vec.txt
lang=ende
output_folder=final_output_folder_path
model="Alibaba-NLP/gte-multilingual-base"
parallel_sents="en.txt de.txt"
CUDA_VISIBLE_DEVICES=0 python train_xling.py -parallel_sents ${parallel_sents} -lang ${lang} -epoch 15 -bs 128 -model ${model} -vec_path ${vec_path} -output_folder ${output_folder}
```

**Note: The code used in Step 3 is designed for training bilingual SWEs (as described in our paper), but can be easily extended to mulitlingual training by feeding paralell sentences of multiple language pairs and jointly minimising the contrastive learning loss.**

# Citation
If you use our code or models, please cite our paper as follows:
```
@inproceedings{wada-etal-2025-static,
    title = "Static Word Embeddings for Sentence Semantic Representation",
    author = "Wada, Takashi  and
      Hirakawa, Yuki  and
      Shimizu, Ryotaro  and
      Kawashima, Takahiro  and
      Saito, Yuki",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.316/",
    pages = "6206--6222",
    ISBN = "979-8-89176-332-6",
}
```
