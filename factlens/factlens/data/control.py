"""Control tasks for probing (Hewitt & Liang, 2019).

A linear probe can attain high accuracy by exploiting spurious surface
cues (token identity, position, length) rather than the property of
interest. The control task breaks the fact while preserving the structural
challenge: every *subject* is assigned a fixed random label drawn from the
same label space as the real task. The probe must now memorize an arbitrary
subject-to-label mapping that is invariant across surface templates —
structurally identical to the real task, minus the factual content.

Selectivity = task accuracy - control accuracy.

A probe whose selectivity is close to zero is reading surface statistics,
not factual knowledge; a selective probe is evidence that the relation is
linearly present at the probed site.

Reference
---------
Hewitt, J., & Liang, P. (2019). Designing and Interpreting Probes with
Control Tasks. EMNLP 2019. https://doi.org/10.18653/v1/D19-1175
"""

from __future__ import annotations

import random

from .dataset import FactDataset, FactExample


class ControlLabeler:
    """Deterministic random relabeling of subjects within each relation.

    The mapping is keyed on ``(relation, subject)`` so that all templates of
    the same fact receive the *same* control label — otherwise the control
    task would be solvable by no strategy at all, making selectivity trivial.
    """

    def __init__(self, dataset: FactDataset, seed: int = 13):
        self.dataset = dataset
        self._labels: dict[tuple[str, str], int] = {}
        rng = random.Random(seed)
        for rel in dataset.relations:
            # Shared label space with the real task: the objects of the
            # *effective* (possibly truncated) fact list.
            objects = dataset.label_space(rel.name)
            for subject, _obj in dataset._effective_facts[rel.name]:
                self._labels[(rel.name, subject)] = rng.randrange(len(objects))

    def label(self, ex: FactExample) -> int:
        """Control label id (index into the relation's object list)."""
        return self._labels[(ex.relation, ex.subject)]

    def label_name(self, ex: FactExample) -> str:
        objects = self.dataset.label_space(ex.relation)
        return objects[self.label(ex)]

    def describe(self, relation_name: str) -> str:
        """Human-readable dump of the control mapping for one relation."""
        objects = self.dataset.label_space(relation_name)
        rows = []
        for subject, _obj in self.dataset._effective_facts[relation_name]:
            rows.append(
                f"    {subject!r:>24} -> {objects[self._labels[(relation_name, subject)]]!r}"
            )
        return "\n".join(rows)
