"""Experiments 3 & 4 — activation patching and causal tracing."""

from __future__ import annotations

from ..analysis.curves import aggregate_patch_curves
from ..analysis.figures import plot_patch_curves, plot_trace_heatmap
from ..causality.patching import PatchAnalyzer
from ..causality.tracing import CausalTracer
from .context import ExperimentContext, publish_figure


def run_patching(
    ctx: ExperimentContext,
    known_flags: dict | None = None,
    template_idx: int = 0,
    progress: bool = True,
) -> dict:
    """Layerwise subject-swap patching over (known) facts.

    Returns:
        ``{"curves": [...], "aggregate": {...}, "n_skipped": int}``
    """
    patch_cfg = ctx.cfg.get("patch", {})
    analyzer = PatchAnalyzer(
        ctx.model,
        ctx.module_map,
        patch_positions=patch_cfg.get("positions", "subject_end"),
        sublayers=patch_cfg.get("sites", "residual") == "residual+sublayers",
    )
    curves, skipped = analyzer.dataset_curves(
        ctx.dataset,
        template_idx=template_idx,
        known_only=bool(ctx.cfg.get("data", {}).get("known_only", True)),
        known_flags=known_flags if known_flags is not None else ctx.known_flags,
        max_facts=patch_cfg.get("max_facts"),
        progress=progress,
    )
    if not curves:
        raise RuntimeError(
            "No valid patching curves — the model failed the direction check on "
            "every fact. Try another model or set data.known_only=false for probes."
        )
    aggregate = aggregate_patch_curves(curves)

    if ctx.tracker is not None:
        ctx.tracker.save_json("patch_curves", curves)
        ctx.tracker.save_json("patch_aggregate", aggregate)
        ctx.tracker.log(
            {
                "patch": {
                    "n_facts": aggregate["n_facts"],
                    "n_skipped": skipped,
                    "peak_site": aggregate["sites"][
                        int(max(range(len(aggregate["restoration_mean"])),
                                key=lambda i: aggregate["restoration_mean"][i]))
                    ],
                }
            }
        )
        path = str(ctx.tracker.artifact_path("patch_curves", suffix=".png"))
        plot_patch_curves(
            aggregate, model_name=ctx.model_display_name, out_path=path, per_relation=True
        )
        public = publish_figure(ctx, "patch_curves")
        if public:
            plot_patch_curves(aggregate, model_name=ctx.model_display_name, out_path=public)

    return {"curves": curves, "aggregate": aggregate, "n_skipped": skipped}


def run_tracing(
    ctx: ExperimentContext,
    known_flags: dict | None = None,
    template_idx: int = 0,
    progress: bool = True,
) -> list[dict]:
    """ROME-style noise-and-restore traces for a handful of facts."""
    trace_cfg = ctx.cfg.get("trace", {})
    tracer = CausalTracer(
        ctx.model, ctx.module_map, noise_std_mult=float(trace_cfg.get("noise_std_mult", 3.0))
    )
    tracer.attach_tokenizer(ctx.tokenizer)
    traces = tracer.dataset_traces(
        ctx.dataset,
        template_idx=template_idx,
        max_facts=trace_cfg.get("max_facts", 8),
        known_flags=known_flags if known_flags is not None else ctx.known_flags,
        progress=progress,
    )
    if ctx.tracker is not None:
        ctx.tracker.save_json("causal_traces", traces)
        for i, trace in enumerate(traces[:3]):
            path = str(ctx.tracker.artifact_path(f"trace_{i}", suffix=".png"))
            plot_trace_heatmap(trace, model_name=ctx.model_display_name, out_path=path)
            public = publish_figure(ctx, f"trace_{i}")
            if public:
                plot_trace_heatmap(trace, model_name=ctx.model_display_name, out_path=public)
    return traces
