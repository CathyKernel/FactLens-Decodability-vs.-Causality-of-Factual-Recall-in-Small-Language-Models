"""Experiment 2 — layerwise linear probes with control tasks.

For each relation, site, and readout position, train a from-scratch linear
probe on template-split data and evaluate on a held-out template. Train an
identical probe on Hewitt-Liang control labels and report selectivity.

The decodability curve D(k) (test macro-F1 at site k, averaged over
relations) is one half of the Causal-Probe Gap; the control curve tells
us how much of D is surface cue rather than fact.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ..analysis.curves import aggregate_probe_results
from ..analysis.figures import plot_probe_curves
from ..data.control import ControlLabeler
from ..probes.linear import LinearProbe
from ..probes.metrics import macro_f1, one_vs_rest_auc
from .context import ExperimentContext, publish_figure
from .features import collect_hidden_features


def _probe_settings(cfg: dict) -> dict:
    probe_cfg = cfg.get("probe", {})
    return {
        "epochs": int(probe_cfg.get("epochs", 300)),
        "lr": float(probe_cfg.get("lr", 1e-3)),
        "weight_decay": float(probe_cfg.get("weight_decay", 1e-4)),
        "batch_size": int(probe_cfg.get("batch_size", 64)),
        "patience": int(probe_cfg.get("patience", 30)),
        "bias": bool(probe_cfg.get("bias", True)),
        "test_template_idx": int(probe_cfg.get("test_template_idx", -1)),
        "val_fraction": float(probe_cfg.get("val_fraction", 0.25)),
        "positions": probe_cfg.get("positions", ["subject_end", "prompt_end"]),
    }


def run_probes(
    ctx: ExperimentContext,
    progress: bool = True,
    verbose_probes: bool = False,
) -> dict:
    """Train probes for every (relation, site, position); aggregate curves.

    Returns:
        ``{"results": [per-(relation, site, position) rows], "aggregate": {...}}``
    """
    settings = _probe_settings(ctx.cfg)
    seed = int(ctx.cfg.get("seed", 13))

    cache = collect_hidden_features(
        ctx, positions=tuple(settings["positions"]), progress=progress
    )
    sites = cache["sites"]
    examples = cache["examples"]
    labeler = ControlLabeler(ctx.dataset, seed=seed)

    results: list[dict] = []
    device = ctx.device
    d_model = ctx.module_map.d_model

    relations = ctx.dataset.relations
    for rel in tqdm(relations, desc="probing", disable=not progress):
        # Label space from the *effective* facts (respects truncation).
        objects = ctx.dataset.label_space(rel.name)
        n_classes = len(objects)
        object_index = {obj: i for i, obj in enumerate(objects)}

        rel_rows = [i for i, ex in enumerate(examples) if ex.relation == rel.name]
        y = np.array([object_index[examples[i].object] for i in rel_rows], dtype=np.int64)
        y_ctrl = np.array([labeler.label(examples[i]) for i in rel_rows], dtype=np.int64)
        splits = ctx.dataset.probe_splits(
            rel.name,
            test_template_idx=settings["test_template_idx"],
            val_fraction=settings["val_fraction"],
            seed=seed,
        )
        tr, va, te = (np.array(splits["train"]), np.array(splits["val"]), np.array(splits["test"]))

        for position in settings["positions"]:
            feats = cache["features"][position]  # (n_all, n_sites, d)
            for si, site in enumerate(sites):
                X = torch.tensor(feats[rel_rows, si, :], dtype=torch.float32, device=device)
                y_t = torch.tensor(y, dtype=torch.long, device=device)
                y_c = torch.tensor(y_ctrl, dtype=torch.long, device=device)

                # Task probe.
                probe = LinearProbe(
                    d_model, n_classes, bias=settings["bias"],
                    lr=settings["lr"], weight_decay=settings["weight_decay"],
                ).to(device)
                probe.fit(
                    X[tr], y_t[tr], X[va], y_t[va],
                    epochs=settings["epochs"], batch_size=settings["batch_size"],
                    patience=settings["patience"], verbose=verbose_probes,
                )
                pred = probe.predict(X[te])
                proba = probe.predict_proba(X[te])
                task_f1 = macro_f1(y[te], pred, n_classes)
                task_auc = one_vs_rest_auc(proba, y[te], n_classes)

                # Control probe: identical pipeline, control labels.
                ctrl_probe = LinearProbe(
                    d_model, n_classes, bias=settings["bias"],
                    lr=settings["lr"], weight_decay=settings["weight_decay"],
                ).to(device)
                ctrl_probe.fit(
                    X[tr], y_c[tr], X[va], y_c[va],
                    epochs=settings["epochs"], batch_size=settings["batch_size"],
                    patience=settings["patience"], verbose=False,
                )
                ctrl_f1 = macro_f1(y_ctrl[te], ctrl_probe.predict(X[te]), n_classes)

                results.append(
                    {
                        "relation": rel.name,
                        "site": site,
                        "position": position,
                        "task_f1": task_f1,
                        "control_f1": ctrl_f1,
                        "selectivity": task_f1 - ctrl_f1,
                        "task_auc": task_auc,
                        "chance": 1.0 / n_classes,
                        "n_train": int(len(tr)),
                        "n_test": int(len(te)),
                    }
                )
                if ctx.tracker is not None:
                    ctx.tracker.log(
                        {
                            "probe": {
                                "relation": rel.name,
                                "site": site,
                                "position": position,
                                "task_f1": task_f1,
                                "control_f1": ctrl_f1,
                            }
                        }
                    )

    aggregate = aggregate_probe_results(results)

    if ctx.tracker is not None:
        ctx.tracker.save_json("probe_results", results)
        ctx.tracker.save_json("probe_aggregate", aggregate)
        path = str(ctx.tracker.artifact_path("probe_curves", suffix=".png"))
        plot_probe_curves(aggregate, model_name=ctx.model_display_name, out_path=path)
        public = publish_figure(ctx, "probe_curves")
        if public:
            plot_probe_curves(aggregate, model_name=ctx.model_display_name, out_path=public)

    return {"results": results, "aggregate": aggregate}
