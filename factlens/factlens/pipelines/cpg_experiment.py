"""Experiment 5 — the Causal-Probe Gap synthesis.

Runs lens (for known-fact flags and the emergence trajectory), probes at
both readout positions (decodability), and patching (causality) in one
process with a single model load, then computes the CPG curves, the
readout onset, the routing offset, and the handoff window. This is the
headline experiment of the repo.
"""

from __future__ import annotations

from ..analysis.cpg import cpg_to_markdown, compute_cpg, first_fall_below, first_rise_above
from ..analysis.figures import plot_cpg
from .context import ExperimentContext, publish_figure
from .lens_experiment import run_lens
from .patch_experiment import run_patching
from .probe_experiment import run_probes


def run_cpg(
    ctx: ExperimentContext,
    progress: bool = True,
) -> dict:
    """Full pipeline: lens -> probes -> patching -> CPG.

    The synthesis uses the probe curves at both positions (``subject_end``
    and ``prompt_end``, from ``probe.positions``) together with the patching
    curve. See :mod:`factlens.analysis.cpg` for the definitions.
    """
    # 1) Lens scan: also produces known-fact flags used downstream.
    lens_out = run_lens(ctx, progress=progress)

    # 2) Probes at both readout positions (decodability).
    probe_out = run_probes(ctx, progress=progress)

    # 3) Activation patching (causality) on known facts.
    patch_out = run_patching(ctx, known_flags=lens_out["known_flags"], progress=progress)

    # 4) Synthesis over the two configured probe positions.
    positions = list(ctx.cfg.get("probe", {}).get("positions", ["subject_end", "prompt_end"]))
    if len(positions) < 2:
        # A single configured position is allowed; the synthesis then uses
        # it for both D curves (documented in the output).
        positions = positions + positions[:1]
    agg_subj = probe_out["aggregate"][positions[0]]
    agg_read = probe_out["aggregate"][positions[-1]]

    cpg_cfg = ctx.cfg.get("cpg", {})
    cpg = compute_cpg(
        sites=agg_subj["sites"],
        d_subject=agg_subj["task_f1"],
        control_subject=agg_subj["control_f1"],
        d_read=agg_read["task_f1"],
        control_read=agg_read["control_f1"],
        chance=agg_subj["chance"],
        causality=patch_out["aggregate"]["restoration_mean"],
        probe_margin=float(cpg_cfg.get("probe_margin", 0.15)),
        patch_margin=float(cpg_cfg.get("patch_margin", 0.2)),
        run_length=int(cpg_cfg.get("run_length", 2)),
    )

    # Per-relation handoff statistics where curves are defined.
    per_relation = {}
    probe_margin = float(cpg_cfg.get("probe_margin", 0.15))
    patch_margin = float(cpg_cfg.get("patch_margin", 0.2))
    for rel, sub in agg_read["per_relation"].items():
        rel_patch = patch_out["aggregate"]["per_relation"].get(rel)
        if rel_patch is None or rel_patch["sites"] != sub["sites"]:
            continue
        r_on = first_rise_above(sub["task_f1"], sub["chance"] + probe_margin)
        c_off = first_fall_below(rel_patch["restoration_mean"], patch_margin)
        per_relation[rel] = {
            "readout_onset": sub["sites"][r_on] if r_on is not None else None,
            "routing_offset": sub["sites"][c_off] if c_off is not None else None,
            "handoff_window_sites": (
                (c_off - r_on) if (r_on is not None and c_off is not None) else None
            ),
        }

    if ctx.tracker is not None:
        ctx.tracker.save_json("cpg", cpg)
        ctx.tracker.save_json("cpg_per_relation", per_relation)
        md_path = ctx.tracker.artifact_path("cpg_table", suffix=".md")
        md_path.write_text(cpg_to_markdown(cpg, model_name=ctx.model_display_name))
        ctx.tracker.log(
            {
                "cpg": {
                    "readout_onset": cpg["readout_onset"]["site"],
                    "routing_offset": cpg["routing_offset"]["site"],
                    "handoff_window_sites": cpg["handoff_window_sites"],
                    "peak_dormant": cpg["peak_dormant"]["value"],
                }
            }
        )
        path = str(ctx.tracker.artifact_path("cpg", suffix=".png"))
        plot_cpg(cpg, model_name=ctx.model_display_name, out_path=path)
        public = publish_figure(ctx, "cpg")
        if public:
            plot_cpg(cpg, model_name=ctx.model_display_name, out_path=public)

    return {
        "lens": lens_out,
        "probes": probe_out,
        "patching": patch_out,
        "cpg": cpg,
        "cpg_per_relation": per_relation,
    }
