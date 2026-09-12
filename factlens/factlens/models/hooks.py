"""From-scratch forward-hook machinery for transformer internals.

FactLens never uses ``nnsight``, ``transformer_lens``, or ``pyvene``; every
intervention is a plain ``nn.Module.register_forward_hook`` implemented here.
The module provides:

* :class:`ModuleMap` — architecture-agnostic discovery of transformer
  blocks, the final norm, embeddings, and the LM head across GPT-2,
  GPT-NeoX/Pythia, and Llama/Qwen style checkpoints.
* :func:`make_capture_hook` — save a module's output hidden states.
* :func:`make_patch_hook` — replace activations at chosen positions
  (activation patching; source and target positions may differ, which
  matters when clean and corrupt prompts tokenize to different lengths).
* :func:`make_noise_hook` — additive Gaussian corruption of the embedding
  stream at chosen positions (ROME-style causal tracing).
* :func:`run_with_hooks` — a single forward pass with hooks attached and
  guaranteed removal afterwards.

Residual-site naming convention (used everywhere in FactLens)
-------------------------------------------------------------
``"embed"``     output of the embedding layer          (site 0)
``"block{i}"``  output of the i-th block, 1 <= i <= L  (sites 1..L)
``"block{i}/attn"`` / ``"block{i}/mlp"``  sublayer outputs
The final norm and LM head are *not* sites: they are applied by
:class:`~factlens.lens.logit_lens.LogitLens` when projecting a site's
hidden state to vocabulary logits.
"""

from __future__ import annotations

import re

import torch
from torch import nn

# --- architecture discovery patterns --------------------------------------

_BLOCK_PATTERNS = [
    re.compile(r"^transformer\.h\.(\d+)$"),      # GPT-2
    re.compile(r"^gpt_neox\.layers\.(\d+)$"),    # GPT-NeoX / Pythia
    re.compile(r"^model\.layers\.(\d+)$"),       # Llama / Qwen / Mistral
]
_FINAL_NORM_CANDIDATES = ["transformer.ln_f", "gpt_neox.final_layer_norm", "model.norm"]
_EMBED_CANDIDATES = ["transformer.wte", "gpt_neox.embed_in", "model.embed_tokens"]
_LM_HEAD_CANDIDATES = ["lm_head", "embed_out"]  # GPTNeoX uses embed_out
_ATTN_CANDIDATES = ["attn", "self_attn", "attention"]
_MLP_CANDIDATES = ["mlp", "feed_forward"]


def _first_existing(named: dict[str, nn.Module], candidates: list[str]) -> nn.Module:
    for name in candidates:
        if name in named:
            return named[name]
    raise ValueError(
        f"Could not locate module; tried candidates {candidates}. "
        "This architecture may be unsupported."
    )


class ModuleMap:
    """Architecture-agnostic map of a causal LM's internals.

    Attributes:
        blocks: Transformer blocks in depth order.
        num_layers: Number of blocks ``L``.
        final_norm: The norm applied to the residual stream before unembedding.
        embed: The input embedding module.
        lm_head: The unembedding module (dedicated head, or tied embeddings).
        norm_kind: ``"rms"`` or ``"layer"`` — the final norm's family.
        norm_eps: The final norm's epsilon.
        d_model: Residual stream width.
        vocab_size: Output vocabulary size.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        named = dict(model.named_modules())

        found: list[tuple[int, str, nn.Module]] = []
        for name, module in named.items():
            for pattern in _BLOCK_PATTERNS:
                match = pattern.match(name)
                if match:
                    found.append((int(match.group(1)), name, module))
                    break
        found.sort(key=lambda item: item[0])
        if not found:
            raise ValueError("No transformer blocks found for this checkpoint.")
        depths = [item[0] for item in found]
        if depths != list(range(len(found))):
            raise ValueError(f"Block depths are not contiguous: {depths}.")

        self.blocks: list[nn.Module] = [item[2] for item in found]
        self.block_names: list[str] = [item[1] for item in found]
        self.num_layers: int = len(self.blocks)

        self.final_norm: nn.Module = _first_existing(named, _FINAL_NORM_CANDIDATES)
        self.embed: nn.Module = _first_existing(named, _EMBED_CANDIDATES)

        if hasattr(model, "lm_head") and isinstance(getattr(model, "lm_head"), nn.Module):
            self.lm_head: nn.Module = model.lm_head
        else:
            self.lm_head = _first_existing(named, _LM_HEAD_CANDIDATES + _EMBED_CANDIDATES)

        # Qwen/Llama wrap RMSNorm in their own class (e.g. Qwen2RMSNorm), so
        # dispatch on the class *name* rather than isinstance(nn.RMSNorm).
        self.norm_kind: str = (
            "rms" if "RMSNorm" in type(self.final_norm).__name__ else "layer"
        )
        self.norm_eps: float = float(getattr(self.final_norm, "eps", 1e-5))
        self.d_model: int = self.embed.weight.shape[1]
        self.vocab_size: int = self.lm_head.weight.shape[0]

    # -- site resolution -----------------------------------------------------

    def residual_sites(self) -> list[str]:
        """All residual-stream sites: ``embed`` plus ``block1..blockL``."""
        return ["embed"] + [f"block{i}" for i in range(1, self.num_layers + 1)]

    def resolve_site(self, site: str) -> nn.Module:
        """Map a site name to the module whose *output* the site refers to."""
        if site == "embed":
            return self.embed
        match = re.match(r"^block(\d+)(?:/(attn|mlp))?$", site)
        if not match:
            raise ValueError(f"Malformed site name: '{site}'.")
        depth = int(match.group(1))
        sub = match.group(2)
        if not 1 <= depth <= self.num_layers:
            raise ValueError(
                f"Block index {depth} out of range 1..{self.num_layers} in '{site}'."
            )
        block = self.blocks[depth - 1]
        if sub is None:
            return block
        candidates = _ATTN_CANDIDATES if sub == "attn" else _MLP_CANDIDATES
        for attr in candidates:
            if hasattr(block, attr):
                return getattr(block, attr)
        raise ValueError(f"Block has no {'attention' if sub == 'attn' else 'MLP'} submodule.")


# --- generic output (un)wrapping -------------------------------------------


def unwrap_hidden(output) -> torch.Tensor:
    """Extract the hidden-state tensor from a module output."""
    if isinstance(output, tuple):
        return output[0]
    return output


def rewrap_hidden(output, new_hidden: torch.Tensor):
    """Rebuild a module output, substituting the hidden-state tensor."""
    if isinstance(output, tuple):
        return (new_hidden,) + tuple(output[1:])
    return new_hidden


# --- hook factories ---------------------------------------------------------


def make_capture_hook(store: dict, key: str):
    """Save the module's output hidden states under ``store[key]``."""

    def hook(module: nn.Module, inputs, output):
        store[key] = unwrap_hidden(output).detach()

    return hook


def make_capture_many(
    module_map: ModuleMap, sites: list[str], store: dict
) -> list[tuple[nn.Module, object]]:
    """Build (module, hook) pairs capturing several sites into one store."""
    pairs = []
    for site in sites:
        module = module_map.resolve_site(site)
        pairs.append((module, make_capture_hook(store, site)))
    return pairs


def make_patch_hook(source: torch.Tensor, source_positions, target_positions):
    """Replace activations at ``target_positions`` with captured values.

    Args:
        source: Hidden states captured from another forward pass, shaped
            ``(1, T_src, d)``.
        source_positions: Positions to copy *from* (in ``source``).
        target_positions: Positions to copy *to* (in the live forward pass).

    The two position lists may differ in both content and length, which is
    essential when clean and corrupt prompts tokenize differently, e.g.
    patching the last subject token of "The capital of France is" into
    "The capital of Germany is".
    """

    def hook(module: nn.Module, inputs, output):
        hidden = unwrap_hidden(output)
        patched = hidden.clone()
        src = source.to(device=hidden.device, dtype=hidden.dtype)
        patched[:, target_positions, :] = src[:, source_positions, :]
        return rewrap_hidden(output, patched)

    return hook


def make_noise_hook(std: float, positions):
    """Add Gaussian noise to the module output at ``positions``.

    Args:
        std: Noise standard deviation (pass ``embedding_std * multiplier``).
        positions: Sequence positions to corrupt, or ``None`` for all.
    """

    def hook(module: nn.Module, inputs, output):
        hidden = unwrap_hidden(output)
        noise = torch.randn(hidden.shape, device=hidden.device, dtype=hidden.dtype) * std
        if positions is not None:
            mask = torch.zeros(
                hidden.shape[:2], dtype=torch.bool, device=hidden.device
            )  # (B, T)
            mask[:, list(positions)] = True
            noise = noise * mask.unsqueeze(-1)
        return rewrap_hidden(output, hidden + noise)

    return hook


# --- execution ---------------------------------------------------------------


@torch.no_grad()
def run_with_hooks(model: nn.Module, input_ids: torch.Tensor, fwd_hooks: list):
    """One forward pass with forward hooks attached, then removed.

    Args:
        input_ids: Tensor of shape ``(1, T)`` already on the model's device.
        fwd_hooks: Iterable of ``(module, hook_fn)`` pairs.

    Returns:
        The model's normal output (e.g. ``CausalLMOutputWithPast``).
    """
    handles = [module.register_forward_hook(fn) for module, fn in fwd_hooks]
    try:
        output = model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return output


@torch.no_grad()
def capture_sites(
    model: nn.Module,
    module_map: ModuleMap,
    input_ids: torch.Tensor,
    sites: list[str],
) -> tuple[dict, torch.Tensor]:
    """Run a forward pass capturing the given sites.

    Returns:
        ``(store, logits)`` where ``store[site]`` is a detached
        ``(1, T, d)`` tensor and ``logits`` are the model's output logits.
    """
    store: dict = {}
    hooks = make_capture_many(module_map, sites, store)
    output = run_with_hooks(model, input_ids, hooks)
    return store, output.logits
