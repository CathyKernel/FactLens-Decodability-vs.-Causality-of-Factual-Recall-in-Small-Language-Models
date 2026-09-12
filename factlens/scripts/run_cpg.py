#!/usr/bin/env python3
"""The headline experiment: Causal-Probe Gap synthesis.

Runs lens -> probes (both positions) -> patching in one process, then
computes per-site CPG curves, the readout onset, the routing offset, and
the handoff window.

Usage:
    python scripts/run_cpg.py --config configs/experiments/cpg.yaml
    python scripts/run_cpg.py --config configs/experiments/cpg.yaml \
        --model qwen2.5-0.5b --device cuda
"""

from factlens.analysis.cpg import cpg_to_markdown
from factlens.pipelines.cli import load_cli_config
from factlens.pipelines.context import make_context
from factlens.pipelines.cpg_experiment import run_cpg


def main() -> None:
    cfg, args = load_cli_config(None, description=__doc__)
    slug = cfg.model.get("slug", str(cfg.model.name_or_path).replace("/", "_"))
    ctx = make_context(cfg, run_name=f"cpg_{slug}")
    print(ctx.dataset.describe())

    out = run_cpg(ctx, progress=not args.no_progress)

    cpg = out["cpg"]
    print(f"\n=== Causal-Probe Gap — {ctx.model_display_name} ===")
    print(cpg_to_markdown(cpg, model_name=ctx.model_display_name))

    print("\nPer-relation handoff statistics:")
    for rel, stats in out["cpg_per_relation"].items():
        print(
            f"  {rel:<18} readout-onset {str(stats['readout_onset']):>8}  "
            f"routing-offset {str(stats['routing_offset']):>8}  "
            f"handoff {stats['handoff_window_sites']}"
        )

    if ctx.tracker is not None:
        print(f"\nArtifacts: {ctx.tracker.summary()['run_dir']}")


if __name__ == "__main__":
    main()
