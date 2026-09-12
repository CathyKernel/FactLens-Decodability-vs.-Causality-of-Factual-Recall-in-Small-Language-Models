"""Shared CLI plumbing for the ``scripts/run_*.py`` entrypoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import DotDict, apply_overrides, deep_merge, load_config


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config", required=True, help="YAML config path (see configs/experiments/)"
    )
    parser.add_argument(
        "--model",
        help="Model slug (gpt2 | pythia-410m | qwen2.5-0.5b); overrides the config's model section",
    )
    parser.add_argument("--device", help="cpu | cuda | cuda:0 ...")
    parser.add_argument(
        "--limit", type=int, help="Limit facts per relation (quick iterations)"
    )
    parser.add_argument(
        "--relations", help="Comma-separated relation names (default: all)"
    )
    parser.add_argument(
        "--set", dest="overrides", action="append", default=[],
        metavar="KEY=VALUE", help="Dot-path config override, e.g. probe.lr=0.01",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm bars")
    return parser


def load_cli_config(argv: list[str] | None, description: str):
    """Parse CLI args, load + merge configs, apply overrides.

    Returns:
        ``(cfg, args)`` where ``cfg`` is a merged :class:`DotDict`.
    """
    args = build_parser(description).parse_args(argv)

    cfg: DotDict = load_config(args.config)

    if args.model:
        model_path = (
            Path(args.config).resolve().parent.parent / "models" / f"{args.model}.yaml"
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model config at {model_path}. Expected one of: "
                "gpt2, pythia-410m, pythia-160m, qwen2.5-0.5b."
            )
        cfg = DotDict(deep_merge(dict(cfg), dict(load_config(model_path))))

    if args.device:
        cfg["device"] = args.device
    if args.limit is not None:
        cfg.setdefault("data", {})["limit_per_relation"] = args.limit
    if args.relations:
        cfg.setdefault("data", {})["relations"] = [
            r.strip() for r in args.relations.split(",") if r.strip()
        ]
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)

    return cfg, args
