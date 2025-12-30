# Historical Code

This directory contains the original SWE4Semantics implementation from the EMNLP 2025 paper:
"Static Word Embeddings for Sentence Semantic Representation"

## Original Files

- `extract_embs.py` - Extract contextual embeddings from transformer models
- `apply_pca.py` - Apply sentence-level PCA for English embeddings
- `apply_pca_xling.py` - Apply PCA for cross-lingual embeddings
- `train.py` - Knowledge distillation training for English
- `train_xling.py` - Contrastive learning for cross-lingual embeddings
- `example_en.py` - Usage example for English embeddings
- `example_xling.py` - Usage example for cross-lingual embeddings
- `util.py` - Utility functions (load_w2v, encode_text)
- `requirements.txt` - Original dependencies

## Citation

```bibtex
@inproceedings{wada-etal-2025-static,
    title = "Static Word Embeddings for Sentence Semantic Representation",
    author = "Wada, Takashi and Hirakawa, Yuki and Shimizu, Ryotaro and Kawashima, Takahiro and Saito, Yuki",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
    url = "https://aclanthology.org/2025.emnlp-main.316/",
}
```

## Note

This code is preserved for reference. The new implementation in `src/qwen3_static_embeddings/`
builds upon these ideas but uses Qwen3-Embedding models and a modernized codebase.
