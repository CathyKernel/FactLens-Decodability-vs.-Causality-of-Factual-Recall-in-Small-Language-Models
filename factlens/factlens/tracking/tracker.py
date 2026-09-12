"""A lightweight experiment tracker: run directories, JSONL metrics, figures.

No wandb/tensorboard dependency is required. Runs are plain directories

    results/<run_name>/<timestamp>/
        config.json          frozen merged config
        metrics.jsonl        one JSON record per event
        *.json               analysis artifacts (curves, CPG tables)
        *.png                figures
        checkpoints/         probe weights

An optional TensorBoard writer is enabled when ``tracking.tensorboard`` is
true and ``torch.utils.tensorboard`` is importable; JSONL logging always
stays on, so runs remain fully inspectable without any extra tooling.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class RunTracker:
    """Filesystem-backed experiment tracker."""

    def __init__(
        self,
        run_name: str,
        root: str | Path = "results",
        config: dict | None = None,
        use_tensorboard: bool = False,
    ):
        self.run_name = run_name
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / run_name / stamp
        self.checkpoint_dir = self.dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self._metrics_path = self.dir / "metrics.jsonl"
        self._tb = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(log_dir=str(self.dir / "tb"))
            except ImportError:
                print("[tracker] tensorboard not installed; JSONL logging only")
        if config is not None:
            self.save_json("config", _to_plain(config))
        print(f"[tracker] run directory: {self.dir}")

    # -- artifacts -----------------------------------------------------------

    def save_json(self, name: str, obj: Any) -> Path:
        """Write a JSON artifact (name without extension)."""
        path = self.dir / f"{name}.json"
        path.write_text(json.dumps(_to_plain(obj), indent=2))
        return path

    def artifact_path(self, name: str, suffix: str = ".json") -> Path:
        return self.dir / f"{name}{suffix}"

    def save_torch(self, name: str, obj) -> Path:
        import torch

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(obj, path)
        return path

    # -- metrics ---------------------------------------------------------------

    def log(self, record: dict, step: int | None = None):
        """Append one metrics record to ``metrics.jsonl``."""
        entry = {"t": time.time()}
        if step is not None:
            entry["step"] = step
        entry.update(_to_plain(record))
        with self._metrics_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        if self._tb is not None:
            for key, value in entry.items():
                if isinstance(value, (int, float)) and key not in ("t", "step"):
                    self._tb.add_scalar(key, value, step if step is not None else 0)

    def summary(self) -> dict:
        """List artifacts produced by this run."""
        files = sorted(p.name for p in self.dir.rglob("*") if p.is_file())
        return {"run_dir": str(self.dir), "files": files}


def _to_plain(obj: Any) -> Any:
    """Recursively convert numpy/torch scalars and Paths to JSON-safe types."""
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj
