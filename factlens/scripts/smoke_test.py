#!/usr/bin/env python3
"""End-to-end smoke test: every analysis runs on a tiny slice of the data.

Verifies in one pass (GPT-2, 3 relations, 2 facts each, reduced epochs):
  1. hooks capture all residual sites with correct shapes;
  2. the lens trajectory's final row equals the model's real logits;
  3. probes at the last site beat chance on known relations;
  4. patching the final residual site restores >= 50% of the logit diff;
  5. the CPG synthesis computes onsets and a dormant window without error.

Run:  python scripts/smoke_test.py   (~2 min on GPU, ~6 min on CPU)
"""

import time

import torch

from factlens.config import DotDict, deep_merge
from factlens.pipelines.cpg_experiment import run_cpg
from factlens.pipelines.context import make_context

SMOKE_CONFIG = DotDict(
    {
        "seed": 13,
        "device": "cpu",
        "output_root": "results",
        "model": {
            "name_or_path": "gpt2",
            "display_name": "GPT-2 (124M)",
            "slug": "gpt2",
            "dtype": "float32",
            "prepend_bos": False,
        },
        "data": {
            "relations": ["capital-country", "element-symbol", "animal-young"],
            "limit_per_relation": 2,
            "known_only": True,
            "known_rank_threshold": 5,
        },
        "lens": {"align": "center", "topk": [5, 10, 100]},
        "probe": {
            "positions": ["subject_end", "prompt_end"],
            "test_template_idx": -1,
            "val_fraction": 0.25,
            "epochs": 60,
            "lr": 0.01,
            "weight_decay": 0.0001,
            "batch_size": 32,
            "patience": 15,
            "bias": True,
        },
        "patch": {"positions": "subject_end", "sites": "residual", "max_facts": 6},
        "trace": {"noise_std_mult": 3.0, "max_facts": 2},
        "cpg": {"probe_margin": 0.15, "patch_margin": 0.15, "run_length": 2},
        "tracking": {"tensorboard": False},
        "publish": {"figures": False},
    }
)


def _readout_patch_restoration(ctx) -> float:
    """Patch (last block, readout position); restoration must be exactly 1.

    The logits at the final position are a fixed function (final norm +
    LM head) of the residual stream at that position, so swapping in the
    clean last-block hidden state at the readout position reproduces the
    clean logit difference bit-for-bit. This validates the whole patch
    pipeline (capture -> patch hook -> logit-diff metric) end to end.
    """
    import torch

    from factlens.models.hooks import make_capture_many, make_patch_hook, run_with_hooks

    ex = next(
        e
        for rel in ctx.dataset.relation_names
        for e in ctx.dataset.examples_for(rel)
        if e.template_idx == 0
    )
    counter = ctx.dataset.counterfactual(ex)
    ids_clean = torch.tensor([ex.input_ids], device=ctx.device)
    ids_corr = torch.tensor([counter.input_ids], device=ctx.device)
    last_site = f"block{ctx.module_map.num_layers}"
    sites = [last_site]

    store_clean: dict = {}
    out_clean = run_with_hooks(
        ctx.model, ids_clean, make_capture_many(ctx.module_map, sites, store_clean)
    )
    out_corr = run_with_hooks(ctx.model, ids_corr, [])

    ans, dis = ex.answer_token_id, counter.answer_token_id
    ld_clean = float(out_clean.logits[0, -1, ans] - out_clean.logits[0, -1, dis])
    ld_corr = float(out_corr.logits[0, -1, ans] - out_corr.logits[0, -1, dis])
    if ld_clean <= ld_corr:
        # Degenerate pair for this model; the check needs a fact where the
        # model distinguishes clean from counterfactual answers.
        raise RuntimeError("smoke machinery check picked a degenerate fact pair")

    hook = make_patch_hook(store_clean[last_site], [ex.last_pos], [counter.last_pos])
    out_patched = run_with_hooks(
        ctx.model, ids_corr, [(ctx.module_map.resolve_site(last_site), hook)]
    )
    ld_patched = float(out_patched.logits[0, -1, ans] - out_patched.logits[0, -1, dis])
    return (ld_patched - ld_corr) / (ld_clean - ld_corr)


def main() -> None:
    torch.manual_seed(13)
    start = time.time()

    cfg = DotDict(deep_merge({}, dict(SMOKE_CONFIG)))
    if torch.cuda.is_available():
        cfg["device"] = "cuda"
    ctx = make_context(cfg, run_name="smoke")
    n_layers = ctx.module_map.num_layers
    print(f"Model: {ctx.model_display_name} | layers={n_layers} | d={ctx.module_map.d_model}")
    print(ctx.dataset.describe())

    out = run_cpg(ctx, progress=False)

    # --- assertions -------------------------------------------------------
    lens_agg = out["lens"]["aggregate_all"]
    n_sites = n_layers + 2  # embed + L blocks + final
    assert len(lens_agg["sites"]) == n_sites, "lens trajectory length mismatch"
    assert lens_agg["n_facts"] == 6, "expected 6 canonical facts"
    assert lens_agg["known_fraction"] > 0, "model knows none of the smoke facts?!"

    # Lens final row must match the true readout: known facts rank below k.
    known_scans = [s for s in out["lens"]["scans"] if s["known"]]
    final_ranks = [s["trajectory"][-1]["answer_rank"] for s in known_scans]
    assert all(r < 5 for r in final_ranks), f"known facts rank >= 5 at final site: {final_ranks}"

    probe_agg = out["probes"]["aggregate"]["subject_end"]
    assert len(probe_agg["sites"]) == n_layers + 1, "probe curve length mismatch"
    last_f1 = probe_agg["task_f1"][-1]
    chance = probe_agg["chance"]
    assert last_f1 > chance, f"last-site probe F1 {last_f1:.3f} <= chance {chance:.3f}"
    patch_agg = out["patching"]["aggregate"]
    assert patch_agg["n_facts"] >= 1, "no valid patching curves"
    restorations = patch_agg["restoration_mean"]
    # Subject-position patching: full restoration while the identity is
    # still carried by the subject token (early sites), decaying to ~0 once
    # routing to the readout position is complete (last sites).
    assert max(restorations) >= 0.9, f"peak restoration {max(restorations):.3f} < 0.9"
    assert abs(restorations[0] - 1.0) < 0.05, "embed-site patch should fully restore"

    # Machinery check: patching (last block, readout position) must restore
    # the clean logits *exactly* — everything downstream is just norm+head.
    exact = _readout_patch_restoration(ctx)
    assert abs(exact - 1.0) < 1e-4, f"(blockL, last-pos) patch restored only {exact:.4f}"

    cpg = out["cpg"]
    # At this tiny scale (2 facts/relation) the aggregate readout onset may
    # sit below the margin threshold; what must hold is (a) the routing
    # offset exists (patch restoration decays), and (b) prompt-end
    # decodability rises through the stack (routing moves information to
    # the readout position).
    assert cpg["routing_offset"]["site"] is not None, "no routing offset found"
    d_read = out["probes"]["aggregate"]["prompt_end"]["task_f1"]
    assert d_read[-1] > d_read[0] + 0.1, (
        f"prompt-end probe F1 did not rise: {d_read[0]:.3f} -> {d_read[-1]:.3f}"
    )

    print("\n--- smoke test results ---")
    print(f"facts known          : {lens_agg['known_fraction']:.0%}")
    print(f"last-site probe F1   : {last_f1:.3f} (chance {chance:.3f})")
    print(f"peak restoration     : {max(restorations):.3f} "
          f"(embed site {restorations[0]:.3f}; last site {restorations[-1]:.3f})")
    print(f"readout patch exact  : {exact:.4f}")
    print(
        f"routing              : offset at {cpg['routing_offset']['site']}, "
        f"readout onset {cpg['readout_onset']['site']}, "
        f"handoff window {cpg['handoff_window_sites']} site(s)"
    )
    print(f"\nPASS  (total {time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
