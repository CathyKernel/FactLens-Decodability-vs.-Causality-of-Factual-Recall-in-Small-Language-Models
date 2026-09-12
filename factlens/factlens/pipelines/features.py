"""One-shot hidden-state extraction for probing.

A single forward pass per example captures every residual site; we slice
the requested readout positions and cache everything as float32 numpy
arrays. Probes at *all* sites and *all* templates then reuse this cache,
so the probing experiment costs one model pass plus L+1 tiny linear fits
per (relation, position).
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..models.hooks import make_capture_many, run_with_hooks
from .context import ExperimentContext

#: Readout positions supported by the feature cache.
POSITIONS = ("subject_end", "prompt_end")


def _position_index(example, position: str) -> int:
    if position == "subject_end":
        return example.subject_end_pos
    if position == "prompt_end":
        return example.last_pos
    raise ValueError(f"Unknown position '{position}'.")


@torch.no_grad()
def collect_hidden_features(
    ctx: ExperimentContext,
    positions: tuple[str, ...] = POSITIONS,
    progress: bool = True,
) -> dict:
    """Capture residual states at the requested positions for every example.

    Returns:
        ``{"features": {position: (n_examples, n_sites, d) float32},
        "sites": [...], "examples": [FactExample, ...]}`` — row ``i`` of the
        feature arrays corresponds to ``examples[i]`` (dataset order).
    """
    sites = ctx.module_map.residual_sites()
    n = len(ctx.dataset.examples)
    n_sites = len(sites)
    d = ctx.module_map.d_model

    buffers = {
        position: np.zeros((n, n_sites, d), dtype=np.float32) for position in positions
    }

    store: dict = {}
    hooks = make_capture_many(ctx.module_map, sites, store)
    for i, ex in enumerate(
        tqdm(ctx.dataset.examples, desc="features", disable=not progress)
    ):
        ids = torch.tensor([ex.input_ids], device=ctx.device)
        run_with_hooks(ctx.model, ids, hooks)
        for si, site in enumerate(sites):
            hidden = store[site][0]  # (T, d)
            for position in positions:
                pos = _position_index(ex, position)
                buffers[position][i, si] = hidden[pos].detach().to(torch.float32).cpu().numpy()

    return {
        "features": buffers,
        "sites": sites,
        "examples": list(ctx.dataset.examples),
    }
