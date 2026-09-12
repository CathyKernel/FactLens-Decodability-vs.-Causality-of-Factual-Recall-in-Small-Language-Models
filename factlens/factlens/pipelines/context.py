"""Shared setup: build model + dataset + tracker from a merged config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import PreTrainedTokenizerBase

from ..config import DotDict
from ..data.dataset import FactDataset
from ..data.relations import get_fact_bank
from ..models.hooks import ModuleMap
from ..models.loader import load_model_and_tokenizer
from ..tracking.tracker import RunTracker
from ..utils.common import pick_device, seed_everything


@dataclass
class ExperimentContext:
    """Everything an experiment needs, built once and shared."""

    cfg: DotDict
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    module_map: ModuleMap
    dataset: FactDataset
    device: str
    tracker: RunTracker | None = None
    known_flags: dict = field(default_factory=dict)

    @property
    def model_display_name(self) -> str:
        return self.cfg.model.get("display_name", self.cfg.model.name_or_path)

    @property
    def model_slug(self) -> str:
        return self.cfg.model.get("slug", self.cfg.model.name_or_path.replace("/", "_"))


def make_context(
    cfg: DotDict,
    run_name: str,
    use_tracker: bool = True,
) -> ExperimentContext:
    """Load model, tokenizer, and dataset; optionally create a tracker."""
    seed_everything(int(cfg.get("seed", 13)))
    device = pick_device(cfg.get("device", "cpu"))

    model, tokenizer = load_model_and_tokenizer(
        cfg.model.name_or_path, dtype=cfg.model.get("dtype", "float32"), device=device
    )
    module_map = ModuleMap(model)

    relations = cfg.data.get("relations") or None
    relation_objs = get_fact_bank(relations)
    dataset = FactDataset(
        relations=relation_objs,
        tokenizer=tokenizer,
        prepend_bos=bool(cfg.model.get("prepend_bos", False)),
        limit_per_relation=cfg.data.get("limit_per_relation"),
    )

    tracker = None
    if use_tracker:
        tracker = RunTracker(
            run_name=run_name,
            root=cfg.get("output_root", "results"),
            config=cfg,
            use_tensorboard=bool(cfg.get("tracking", {}).get("tensorboard", False)),
        )

    return ExperimentContext(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        module_map=module_map,
        dataset=dataset,
        device=device,
        tracker=tracker,
    )


@torch.no_grad()
def compute_known_flags(
    ctx: ExperimentContext, template_idx: int = 0, progress: bool = True
) -> dict:
    """Rank check per fact: is the answer's first token within the top-k?

    A fact counts as known when the answer token's rank at the final
    readout is below ``data.known_rank_threshold`` (default 5). Greedy
    top-1 equality is too strict for small models, whose next-token
    distributions often place a determiner (" the", " a") above the entity.

    Returns a dict keyed ``(relation, subject)`` -> bool. Used to restrict
    lens/patching analyses to facts the model actually knows (configurable
    via ``data.known_only``).
    """
    if ctx.known_flags:
        return ctx.known_flags
    from tqdm import tqdm

    threshold = int(ctx.cfg.get("data", {}).get("known_rank_threshold", 5))
    flags: dict = {}
    examples = [
        ex
        for rel in ctx.dataset.relation_names
        for ex in ctx.dataset.examples_for(rel)
        if ex.template_idx == template_idx
    ]
    for ex in tqdm(examples, desc="known-check", disable=not progress):
        ids = torch.tensor([ex.input_ids], device=ctx.device)
        logits = ctx.model(input_ids=ids).logits[0, -1]
        answer_logit = logits[ex.answer_token_id]
        rank = int((logits > answer_logit).sum())
        flags[(ex.relation, ex.subject)] = rank < threshold
    ctx.known_flags = flags
    return flags


def publish_figure(ctx: ExperimentContext, name: str) -> str | None:
    """Return a path in the public figures dir when ``publish.figures`` is on."""
    publish = ctx.cfg.get("publish", {})
    if not publish.get("figures", False):
        return None
    out_dir = Path(publish.get("dir", "figures"))
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{ctx.model_slug}_{name}.png")
