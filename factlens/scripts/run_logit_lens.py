#!/usr/bin/env python3
"""Logit-lens trajectory analysis of factual recall.

Usage:
    python scripts/run_logit_lens.py --config configs/experiments/logit_lens.yaml
    python scripts/run_logit_lens.py --config configs/experiments/logit_lens.yaml \
        --model qwen2.5-0.5b --device cuda
"""

from factlens.pipelines.cli import load_cli_config
from factlens.pipelines.context import make_context
from factlens.pipelines.lens_experiment import run_lens


def main() -> None:
    cfg, args = load_cli_config(None, description=__doc__)
    slug = cfg.model.get("slug", str(cfg.model.name_or_path).replace("/", "_"))
    ctx = make_context(cfg, run_name=f"lens_{slug}")
    print(ctx.dataset.describe())

    out = run_lens(ctx, progress=not args.no_progress)

    agg = out["aggregate_known"]
    print(f"\nFacts scanned: {out['aggregate_all']['n_facts']} "
          f"(known: {out['aggregate_all']['known_fraction']:.1%})")
    print(f"Answer top-1 accuracy at final site: {agg['top1_accuracy'][-1]:.3f}")
    sites = agg["sites"]
    for i, site in enumerate(sites):
        print(
            f"  {site:>8}  answer-logit {agg['answer_logit_mean'][i]:+7.2f}  "
            f"top-5 {agg['in_top_k']['5'][i]:.2f}  "
            f"top-100 {agg['in_top_k']['100'][i]:.2f}"
        )

    if ctx.tracker is not None:
        print(f"\nArtifacts: {ctx.tracker.summary()['run_dir']}")


if __name__ == "__main__":
    main()
