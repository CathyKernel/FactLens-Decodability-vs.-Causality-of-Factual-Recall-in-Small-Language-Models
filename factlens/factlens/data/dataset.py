"""Fact dataset construction: tokenization, span location, and splits.

The dataset turns each ``(relation, fact, template)`` triple into a
:class:`FactExample` that carries

* the tokenized prompt (optionally prepended with a BOS token — Pythia
  models were trained with one, GPT-2 and Qwen2 were not),
* the *subject token span* found via character offset mapping, so analysis
  can target the last subject token (where factual knowledge is expected to
  be written, cf. Geva et al. 2022) or the final prompt token (where the
  answer is about to be read out),
* the *answer first-token id*, i.e. the id of the first token of
  ``" " + object`` — the token whose logit we track through the lens and
  whose restoration we measure when patching.

All bookkeeping is deterministic given the tokenizer and ``prepend_bos``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from transformers import PreTrainedTokenizerBase

from .relations import Relation, get_fact_bank


@dataclass
class FactExample:
    """One rendered prompt with analysis metadata."""

    relation: str
    subject: str
    object: str
    template_idx: int
    prompt: str
    input_ids: list[int]
    subject_token_span: tuple[int, int]  # [start, end) indices over input_ids
    answer_ids: list[int]  # token ids of " " + object
    fact_idx: int  # position of the fact within its relation

    @property
    def answer_token_id(self) -> int:
        """First token of the object, used as the tracked logit."""
        return self.answer_ids[0]

    @property
    def subject_end_pos(self) -> int:
        """Index of the last token covered by the subject span."""
        return self.subject_token_span[1] - 1

    @property
    def last_pos(self) -> int:
        """Index of the final prompt token (the answer readout position)."""
        return len(self.input_ids) - 1

    @property
    def label(self) -> str:
        return self.object


def _locate_subject_span(
    prompt: str, subject: str, offsets: list[tuple[int, int]]
) -> tuple[int, int]:
    """Map the subject's character span onto token indices.

    Uses the fast tokenizer's ``offset_mapping``: a token belongs to the
    subject iff its character interval overlaps the subject interval.
    """
    if prompt.count(subject) != 1:
        raise ValueError(
            f"Subject '{subject}' must occur exactly once in '{prompt}' "
            f"(found {prompt.count(subject)})."
        )
    c0 = prompt.index(subject)
    c1 = c0 + len(subject)
    inside = [
        i
        for i, (s, e) in enumerate(offsets)
        if e > c0 and s < c1 and e > s  # e > s skips zero-length spans
    ]
    if not inside:
        raise ValueError(f"Could not align subject '{subject}' to tokens in '{prompt}'.")
    return inside[0], inside[-1] + 1


class FactDataset:
    """A tokenized bank of factual prompts with lookup helpers.

    Parameters:
        relations: List of :class:`~factlens.data.relations.Relation`.
        tokenizer: A *fast* tokenizer (offset mapping support required).
        prepend_bos: Prepend the tokenizer's BOS token to every prompt.
            Required for Pythia; leave off for GPT-2 / Qwen2.
        limit_per_relation: Optionally truncate each relation to the first
            N facts (used by the smoke test and quick iterations).
    """

    def __init__(
        self,
        relations: list[Relation] | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        prepend_bos: bool = False,
        limit_per_relation: int | None = None,
    ):
        if tokenizer is None:
            raise ValueError("A tokenizer is required (fast tokenizer preferred).")
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError(
                "FactLens requires a fast tokenizer for offset mapping; "
                f"got {type(tokenizer).__name__}."
            )
        self.tokenizer = tokenizer
        self.prepend_bos = prepend_bos
        self.relations = list(relations) if relations is not None else get_fact_bank()

        self.examples: list[FactExample] = []
        self._by_relation: dict[str, list[FactExample]] = {}
        self._index: dict[tuple[str, str, int], FactExample] = {}
        self._effective_facts: dict[str, list[tuple[str, str]]] = {}

        for rel in self.relations:
            facts = rel.facts
            if limit_per_relation is not None:
                facts = facts[:limit_per_relation]
            self._effective_facts[rel.name] = facts
            rel_examples: list[FactExample] = []
            for fact_idx, (subject, obj) in enumerate(facts):
                for t_idx, template in enumerate(rel.templates):
                    ex = self._build_example(rel, subject, obj, fact_idx, t_idx)
                    rel_examples.append(ex)
                    self.examples.append(ex)
                    self._index[(rel.name, subject, t_idx)] = ex
            self._by_relation[rel.name] = rel_examples

    # -- construction ------------------------------------------------------

    def _build_example(
        self, rel: Relation, subject: str, obj: str, fact_idx: int, t_idx: int
    ) -> FactExample:
        prompt = rel.templates[t_idx].format(S=subject)
        enc = self.tokenizer(
            prompt, add_special_tokens=False, return_offsets_mapping=True
        )
        ids: list[int] = list(enc["input_ids"])
        offsets = enc["offset_mapping"]
        span = _locate_subject_span(prompt, subject, offsets)

        if self.prepend_bos:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                raise ValueError("prepend_bos=True but tokenizer has no BOS token.")
            ids = [bos] + ids
            span = (span[0] + 1, span[1] + 1)

        answer_ids = self.tokenizer.encode(" " + obj, add_special_tokens=False)
        if not answer_ids:
            raise ValueError(f"Empty answer encoding for object '{obj}'.")

        return FactExample(
            relation=rel.name,
            subject=subject,
            object=obj,
            template_idx=t_idx,
            prompt=prompt,
            input_ids=ids,
            subject_token_span=span,
            answer_ids=answer_ids,
            fact_idx=fact_idx,
        )

    # -- lookups -----------------------------------------------------------

    @property
    def relation_names(self) -> list[str]:
        return [r.name for r in self.relations]

    def relation(self, name: str) -> Relation:
        for r in self.relations:
            if r.name == name:
                return r
        raise KeyError(f"Relation '{name}' not in dataset.")

    def examples_for(self, relation_name: str) -> list[FactExample]:
        return self._by_relation[relation_name]

    def label_space(self, relation_name: str) -> list[str]:
        """Unique objects among the relation's *effective* facts, in order.

        The probing label space is defined by the facts actually present in
        the dataset (respecting ``limit_per_relation``), not by the full
        relation — otherwise macro-F1 is capped by classes that never occur.
        """
        out: list[str] = []
        for ex in self._by_relation[relation_name]:
            if ex.object not in out:
                out.append(ex.object)
        return out

    def example(self, relation: str, subject: str, template_idx: int) -> FactExample:
        return self._index[(relation, subject, template_idx)]

    def counterfactual(self, ex: FactExample) -> FactExample:
        """The next fact (cyclically) under the same template.

        Swapping the subject with a same-relation counterfactual keeps the
        prompt grammatical while changing the intended fact — the standard
        corruption for activation patching on factual recall. Cycles over
        the *effective* fact list, so truncation via ``limit_per_relation``
        is respected.
        """
        facts = self._effective_facts[ex.relation]
        n = len(facts)
        next_idx = (ex.fact_idx + 1) % n
        next_subject = facts[next_idx][0]
        return self.example(ex.relation, next_subject, ex.template_idx)

    # -- probe splits -------------------------------------------------------

    def probe_splits(
        self,
        relation_name: str,
        test_template_idx: int = -1,
        val_fraction: float = 0.25,
        seed: int = 13,
    ) -> dict[str, list[int]]:
        """Template-holdout splits over this dataset's examples.

        The test split uses a held-out *template* (surface form), not held-out
        facts, so it measures whether the probe reads the relation, not the
        specific wording. Remaining examples are split per class into
        train/val.

        Returns:
            ``{"train": [...], "val": [...], "test": [...]}`` lists of example
            indices into :attr:`examples_for(relation_name)`.
        """
        rel_examples = self.examples_for(relation_name)
        n_templates = len(self.relation(relation_name).templates)
        if test_template_idx < 0:
            test_template_idx = n_templates + test_template_idx

        by_label: dict[str, list[int]] = {}
        for i, ex in enumerate(rel_examples):
            if ex.template_idx == test_template_idx:
                continue
            by_label.setdefault(ex.object, []).append(i)

        rng = random.Random(seed)
        train: list[int] = []
        val: list[int] = []
        for obj, idxs in sorted(by_label.items()):
            idxs = sorted(idxs)
            rng.shuffle(idxs)
            n_val = max(1, int(round(len(idxs) * val_fraction)))
            val.extend(idxs[:n_val])
            train.extend(idxs[n_val:])

        test = [i for i, ex in enumerate(rel_examples) if ex.template_idx == test_template_idx]
        return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}

    # -- summary -------------------------------------------------------------

    def describe(self) -> str:
        lines = [
            f"FactDataset: {len(self.examples)} examples, "
            f"{len(self.relations)} relations, prepend_bos={self.prepend_bos}"
        ]
        for rel in self.relations:
            n_facts = len(self._effective_facts[rel.name])
            n_classes = len(self.label_space(rel.name))
            lines.append(
                f"  {rel.name:<18} {n_facts:>3} facts x {len(rel.templates)} templates "
                f"-> {n_classes:>2} classes"
            )
        return "\n".join(lines)
