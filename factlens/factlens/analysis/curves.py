"""Aggregation of per-fact analysis results into per-site curves.

All outputs are plain JSON-serializable dicts (lists of floats keyed by
site), which keeps experiment artifacts inspectable and language-agnostic.
"""

from __future__ import annotations

import numpy as np


def residual_site_labels(n_layers: int) -> list[str]:
    """``["embed", "blk1", ..., "blkL"]`` — the shared x-axis of all curves."""
    return ["embed"] + [f"blk{i}" for i in range(1, n_layers + 1)]


def _mean_sem(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    sem = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return float(arr.mean()), sem


def aggregate_lens_scans(scans: list[dict], topk: list[int] = (5, 10, 100)) -> dict:
    """Aggregate lens trajectories over facts.

    Args:
        scans: Output of :func:`factlens.lens.logit_lens.scan_example`.

    Returns:
        ``{"sites": [...], "answer_logit_mean": [...], "answer_logit_sem": [...],
        "answer_rank_mean": [...], "answer_rank_median": [...],
        "top1_accuracy": [...], "in_top_k": {k: [...]}, "n_facts": int,
        "known_fraction": float}``
    """
    if not scans:
        raise ValueError("No scans to aggregate.")
    sites = [row["site"] for row in scans[0]["trajectory"]]
    agg = {
        "sites": sites,
        "answer_logit_mean": [],
        "answer_logit_sem": [],
        "answer_rank_mean": [],
        "answer_rank_median": [],
        "top1_accuracy": [],
        "in_top_k": {str(k): [] for k in topk},
    }
    for si, _site in enumerate(sites):
        logits = [s["trajectory"][si]["answer_logit"] for s in scans]
        ranks = [s["trajectory"][si]["answer_rank"] for s in scans]
        top1 = [float(s["trajectory"][si]["answer_rank"] == 0) for s in scans]
        mean, sem = _mean_sem(logits)
        agg["answer_logit_mean"].append(mean)
        agg["answer_logit_sem"].append(sem)
        agg["answer_rank_mean"].append(float(np.mean(ranks)))
        agg["answer_rank_median"].append(float(np.median(ranks)))
        agg["top1_accuracy"].append(float(np.mean(top1)))
        for k in topk:
            agg["in_top_k"][str(k)].append(float(np.mean([r < k for r in ranks])))
    agg["n_facts"] = len(scans)
    agg["known_fraction"] = float(np.mean([s["known"] for s in scans]))
    return agg


def aggregate_patch_curves(curves: list[dict]) -> dict:
    """Aggregate per-fact restoration curves.

    Returns:
        ``{"sites": [...], "restoration_mean": [...], "restoration_sem": [...],
        "per_relation": {rel: {"sites": [...], "restoration_mean": [...]}},
        "n_facts": int}``
    """
    if not curves:
        raise ValueError("No patch curves to aggregate.")
    sites = list(curves[0]["restoration"].keys())
    agg = {"sites": sites, "restoration_mean": [], "restoration_sem": []}
    for site in sites:
        vals = [c["restoration"][site] for c in curves]
        mean, sem = _mean_sem(vals)
        agg["restoration_mean"].append(mean)
        agg["restoration_sem"].append(sem)
    per_relation: dict[str, dict] = {}
    for rel in sorted({c["relation"] for c in curves}):
        rel_curves = [c for c in curves if c["relation"] == rel]
        per_relation[rel] = {
            "sites": sites,
            "restoration_mean": [
                float(np.mean([c["restoration"][s] for c in rel_curves])) for s in sites
            ],
            "n_facts": len(rel_curves),
        }
    agg["per_relation"] = per_relation
    agg["n_facts"] = len(curves)
    return agg


def aggregate_probe_results(results: list[dict]) -> dict:
    """Aggregate per-(relation, site, position) probe scores.

    Each result dict must contain: ``relation``, ``site``, ``position``,
    ``task_f1``, ``control_f1``, ``task_auc``, ``chance``.

    Returns:
        ``{"positions": {pos: {"sites": [...], "task_f1": [...],
        "control_f1": [...], "selectivity": [...], "task_auc": [...],
        "chance": float, "per_relation": {...}}}}``
    """
    if not results:
        raise ValueError("No probe results to aggregate.")
    out: dict[str, dict] = {}
    for position in sorted({r["position"] for r in results}):
        rows = [r for r in results if r["position"] == position]
        sites = [r["site"] for r in rows if r["relation"] == rows[0]["relation"]]
        # Site list must be identical across relations (it is: residual sites).
        agg = {
            "sites": sites,
            "task_f1": [],
            "control_f1": [],
            "selectivity": [],
            "task_auc": [],
            "chance": float(np.mean([r["chance"] for r in rows])),
        }
        for si, site in enumerate(sites):
            f1s = [r["task_f1"] for r in rows if r["site"] == site]
            cf1s = [r["control_f1"] for r in rows if r["site"] == site]
            aucs = [r["task_auc"] for r in rows if r["site"] == site and r["task_auc"] is not None]
            agg["task_f1"].append(float(np.mean(f1s)))
            agg["control_f1"].append(float(np.mean(cf1s)))
            agg["selectivity"].append(float(np.mean(f1s) - np.mean(cf1s)))
            agg["task_auc"].append(float(np.mean(aucs)) if aucs else None)
        agg["per_relation"] = {}
        for rel in sorted({r["relation"] for r in rows}):
            rel_rows = [r for r in rows if r["relation"] == rel]
            agg["per_relation"][rel] = {
                "sites": [r["site"] for r in rel_rows],
                "task_f1": [r["task_f1"] for r in rel_rows],
                "control_f1": [r["control_f1"] for r in rel_rows],
                "chance": rel_rows[0]["chance"],
            }
        out[position] = agg
    return out
