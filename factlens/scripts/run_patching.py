#!/usr/bin/env python3
"""Activation patching (experiment 'patching') or causal tracing ('tracing').

Usage:
    python scripts/run_patching.py --config configs/experiments/patching.yaml
    python scripts/run_patching.py --config configs/experiments/tracing.yaml \
        --model gpt2 --device cuda
"""

from factlens.pipelines.cli import load_cli_config
from factlens.pipelines.context import compute_known_flags, make_context
from factlens.pipelines.patch_experiment import run_patching, run_tracing


def main() -> None:
    cfg, args = load_cli_config(None, description=__doc__)
    slug = cfg.model.get("slug", str(cfg.model.name_or_path).replace("/", "_"))
    experiment = cfg.get("experiment", {}).get("name", "patching")
    ctx = make_context(cfg, run_name=f"{experiment}_{slug}")
    print(ctx.dataset.describe())

    known_flags = compute_known_flags(ctx, progress=not args.no_progress)
    n_known = sum(known_flags.values())
    print(f"\nKnown facts: {n_known}/{len(known_flags)} "
          f"({n_known / max(len(known_flags), 1):.1%})")

    if experiment == "tracing":
        traces = run_tracing(ctx, known_flags=known_flags, progress=not args.no_progress)
        print(f"\nCausal traces completed: {len(traces)}")
        for trace in traces[:3]:
            peak = max(
                max(row) for row in trace["restoration"]
            )
            print(f"  '{trace['prompt'][:48]}...' peak restoration {peak:.2f}")
    else:
        out = run_patching(ctx, known_flags=known_flags, progress=not args.no_progress)
        agg = out["aggregate"]
        print(f"\n=== Causality curve ({agg['n_facts']} facts, "
              f"{out['n_skipped']} skipped) ===")
        for i, site in enumerate(agg["sites"]):
            print(f"  {site:>8}  restoration {agg['restoration_mean'][i]:+.3f}")

    if ctx.tracker is not None:
        print(f"\nArtifacts: {ctx.tracker.summary()['run_dir']}")


if __name__ == "__main__":
    main()
