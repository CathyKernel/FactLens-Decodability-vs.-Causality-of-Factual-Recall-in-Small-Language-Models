"""Model and tokenizer loading with architecture-agnostic aliases."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#: Convenience aliases used by configs and the ``--model`` CLI flag.
MODEL_ALIASES = {
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
}


def resolve_model_name(name: str) -> str:
    """Expand an alias to a HuggingFace hub id (pass-through if unknown)."""
    return MODEL_ALIASES.get(name, name)


def _from_pretrained_compat(model_name: str, dtype: torch.dtype):
    """Load a causal LM, handling the ``torch_dtype`` -> ``dtype`` API rename.

    transformers v4.5x renamed ``torch_dtype`` to ``dtype``; older versions
    only accept ``torch_dtype``. Trying ``dtype`` first future-proofs the
    code, and the TypeError fallback keeps older environments working.
    """
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


def load_model_and_tokenizer(
    name_or_path: str,
    dtype: str = "float32",
    device: str = "cpu",
):
    """Load a causal LM and its fast tokenizer in evaluation mode.

    Args:
        name_or_path: Hub id or local path; aliases like ``"gpt2"`` work.
        dtype: One of ``"float32"``, ``"float64"``, ``"bfloat16"``, ``"float16"``.
            Analysis defaults to float32 for numerical reproducibility of
            logit-difference metrics.
        device: ``"cpu"`` or a CUDA device string.

    Returns:
        ``(model, tokenizer)`` with the model in eval mode on ``device``.
    """
    model_name = resolve_model_name(name_or_path)
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not tokenizer.is_fast:
        raise ValueError(
            f"Tokenizer for {model_name} is not fast; offset mapping is required."
        )
    model = _from_pretrained_compat(model_name, torch_dtype)
    model.eval()
    model.to(device)
    # Deterministic eval: disable TF32 matmuls so runs are reproducible.
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return model, tokenizer
