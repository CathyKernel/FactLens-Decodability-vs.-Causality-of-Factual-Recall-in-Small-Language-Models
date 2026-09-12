"""Experiment 1 — logit-lens trajectories of factual recall.

For every canonical-template fact, run one forward pass capturing all
residual sites, then project each site through the lens and record the
answer token's logit and rank. The aggregated trajectory shows where in
the layer stack the answer first surfaces in the unembedding space.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from ..analysis.curves import aggregate_lens_scans
from ..analysis.figures import plot_lens_trajectory
from ..lens.logit_lens import LogitLens, estimate_site_means, scan_example
from .context import ExperimentContext, publish_figure


def run_lens(
    ctx: ExperimentContext,
    template_idx: int = 0,
    known_only: bool = True,
    progress: bool = True,
) -> dict:
    """Scan all facts; aggregate; save artifacts and figures.

    Returns:
        ``{"scans": [...], "aggregate_all": {...}, "aggregate_known": {...},
        "known_flags": {...}}``
    """
    lens_cfg = ctx.cfg.get("lens", {})
    topk = tuple(lens_cfg.get("topk", (5, 10, 100)))
    known_rank = int(ctx.cfg.get("data", {}).get("known_rank_threshold", 5))

    examples = [
        ex
        for rel in ctx.dataset.relation_names
        for ex in ctx.dataset.examples_for(rel)
        if ex.template_idx == template_idx
    ]

    # Mean-centered lens: subtract per-site dataset means before the final
    # norm (lens.align: "center"); "none" gives the raw logit lens.
    site_means = None
    if lens_cfg.get("align", "center") == "center":
        site_means = estimate_site_means(
            ctx.model, ctx.module_map, examples, device=ctx.device
        )
    lens = LogitLens(ctx.module_map, site_means=site_means)

    scans = []
    for ex in tqdm(examples, desc="lens", disable=not progress):
        ids = torch.tensor([ex.input_ids], device=ctx.device)
        scans.append(
            scan_example(
                ctx.model, ctx.module_map, lens, ex, ids,
                known_rank_threshold=known_rank,
            )
        )

    known_flags = {(s["relation"], s["subject"]): s["known"] for s in scans}
    ctx.known_flags = known_flags

    aggregate_all = aggregate_lens_scans(scans, topk=topk)
    known_scans = [s for s in scans if s["known"]]
    aggregate_known = (
        aggregate_lens_scans(known_scans, topk=topk) if known_scans else aggregate_all
    )

    if ctx.tracker is not None:
        ctx.tracker.save_json("lens_scans", scans)
        ctx.tracker.save_json("lens_aggregate_all", aggregate_all)
        ctx.tracker.save_json("lens_aggregate_known", aggregate_known)
        ctx.tracker.log(
            {
                "lens": {
                    "n_facts": aggregate_all["n_facts"],
                    "known_fraction": aggregate_all["known_fraction"],
                    "final_top1_accuracy": aggregate_all["top1_accuracy"][-1],
                }
            }
        )
        ctx.tracker.artifact_path("lens_trajectory").with_suffix(".png")
        path = str(ctx.tracker.artifact_path("lens_trajectory", suffix=".png"))
        plot_lens_trajectory(
            aggregate_known if known_only else aggregate_all,
            model_name=ctx.model_display_name,
            out_path=path,
        )
        public = publish_figure(ctx, "lens_trajectory")
        if public:
            plot_lens_trajectory(
                aggregate_known if known_only else aggregate_all,
                model_name=ctx.model_display_name,
                out_path=public,
            )

    return {
        "scans": scans,
        "aggregate_all": aggregate_all,
        "aggregate_known": aggregate_known,
        "known_flags": known_flags,
    }
