"""Qwen3 model loading utilities."""

import torch
from transformers import AutoModel, AutoTokenizer


def load_tokenizer(
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    trust_remote_code: bool = True,
):
    """
    Load the Qwen3 tokenizer.

    Args:
        model_name: HuggingFace model name
        trust_remote_code: Whether to trust remote code

    Returns:
        Loaded tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    return tokenizer


def load_model(
    model_name: str = "Qwen/Qwen3-Embedding-0.6B",
    device: str = "cuda",
    dtype: str = "float16",
    trust_remote_code: bool = True,
):
    """
    Load the Qwen3 embedding model.

    Args:
        model_name: HuggingFace model name
        device: Device to load model on ("cuda" or "cpu")
        dtype: Data type ("float32", "float16", or "bfloat16")
        trust_remote_code: Whether to trust remote code

    Returns:
        Loaded model in eval mode
    """
    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
    )

    model = model.to(device)
    model.eval()

    return model


def get_hidden_dim(model) -> int:
    """Get the hidden dimension of the model."""
    # Try common attribute names
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "hidden_size"):
            return config.hidden_size
        if hasattr(config, "d_model"):
            return config.d_model

    raise ValueError("Could not determine hidden dimension from model config")
