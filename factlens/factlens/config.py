"""A minimal YAML config system: dot-access, defaults chaining, deep merge.

Experiment configs reference shared defaults plus a model config:

    # configs/experiments/probes.yaml
    _defaults: [../default.yaml, ../models/gpt2.yaml]
    experiment:
      name: probes

``_defaults`` paths resolve relative to the config file's directory and are
merged left-to-right; the file's own keys win last. This gives hydra-style
composition without adding a dependency.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


class DotDict(dict):
    """A dict with attribute access (``cfg.model.name``)."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as err:
            raise AttributeError(key) from err
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[key] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml(path: str | Path) -> dict:
    with open(path) as handle:
        return yaml.safe_load(handle) or {}


def load_config(path: str | Path) -> DotDict:
    """Load a config, resolving its ``_defaults`` chain recursively."""
    path = Path(path).resolve()
    doc = load_yaml(path)
    defaults = doc.pop("_defaults", [])
    merged: dict = {}
    for default_path in defaults:
        if not Path(default_path).is_absolute():
            default_path = path.parent / default_path
        merged = deep_merge(merged, load_config(default_path))
    merged = deep_merge(merged, doc)
    return DotDict(merged)


def save_config(cfg: dict, path: str | Path):
    Path(path).write_text(json.dumps(_plain(cfg), indent=2))


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def apply_overrides(cfg: DotDict, overrides: list[str]) -> DotDict:
    """Apply ``key=value`` dot-path overrides from the command line.

    Example: ``apply_overrides(cfg, ["probe.lr=0.01", "device=cuda"])``
    """
    for item in overrides:
        key, _, raw = item.partition("=")
        if not key or _ == "":
            raise ValueError(f"Bad override '{item}', expected key=value")
        node: dict = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = _parse_value(raw)
    return cfg


def _parse_value(raw: str):
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        return [_parse_value(piece.strip()) for piece in inner.split(",")] if inner else []
    return raw
