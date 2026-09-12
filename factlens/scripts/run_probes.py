#!/usr/bin/env python3
"""Layerwise linear probes with Hewitt-Liang control tasks.

Usage:
    python scripts/run_probes.py --config configs/experiments/probes.yaml
    python scripts/run_probes.py --config configs/experiments/probes.yaml \
        --model pythia-410m --set probe.epochs=150
"""

from factlens.pipelines.cli import load_cli_config
from factlens.pipelines.context import make_context
from factlens.pipelines.probe_experiment import run_probes


def main() -> None:
    cfg, args = load_cli_config(None, description=__doc__)
    slug = cfg.model.get("slug", str(cfg.model.name_or_path).replace("/", "_"))
    ctx = make_context(cfg, run_name=f"probes_{slug}")
    print(ctx.dataset.describe())

    out = run_probes(ctx, progress=not args.no_progress)

    for position, agg in out["aggregate"].items():
        print(f"\n=== Decodability curve (probe at {position}) ===")
        print(f"chance (uniform, avg over relations): {agg['chance']:.3f}")
        for i, site in enumerate(agg["sites"]):
            auc = agg["task_auc"][i]
            auc_str = f"{auc:.3f}" if auc is not None else "  n/a"
            print(
                f"  {site:>8}  task-F1 {agg['task_f1'][i]:.3f}  "
                f"control-F1 {agg['control_f1'][i]:.3f}  "
                f"selectivity {agg['selectivity'][i]:+.3f}  AUC {auc_str}"
            )

    if ctx.tracker is not None:
        print(f"\nArtifacts: {ctx.tracker.summary()['run_dir']}")


if __name__ == "__main__":
    main()
