"""Activation patching for factual recall, from scratch.

Patching measures the *causal* contribution of activations at a site: run a
corrupted forward pass, then overwrite chosen activations with values
captured from a clean pass and see how much the model's answer is restored.

Corruption scheme
-----------------
The subject is swapped for a same-relation counterfactual subject (e.g.
"The capital of France is" -> "The capital of Germany is"). This keeps the
prompt grammatical and the readout position fixed, while flipping the
intended fact. The measured readout is the **logit difference**

    ld(x) = logit(answer | x) - logit(distractor | x)

where ``answer`` is the clean fact's first object token and ``distractor``
the counterfactual fact's first object token. Patching site ``k`` yields

    restoration(k) = (ld(patched_k) - ld(corrupt)) / (ld(clean) - ld(corrupt))

which is 0 when the patch changes nothing and 1 when it fully restores the
clean behavior.

Facts where the model does not discriminate (``ld(clean) <= 0`` or
``ld(corrupt) >= 0``) are excluded and counted, following standard practice
— a patching experiment is only interpretable on facts the model knows.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from ..data.dataset import FactDataset, FactExample
from ..models.hooks import (
    ModuleMap,
    make_capture_many,
    make_patch_hook,
    run_with_hooks,
)


class PatchAnalyzer:
    """Layerwise activation patching on factual prompts.

    Args:
        model: The causal LM (eval mode, on device).
        module_map: Resolved internals.
        patch_positions: ``"subject_end"`` patches the last subject token;
            ``"subject_span"`` patches the whole subject span (skipped for
            facts whose clean/corrupt spans differ in length).
        sublayers: Also patch attention/MLP outputs in addition to the
            residual stream.
    """

    def __init__(
        self,
        model,
        module_map: ModuleMap,
        patch_positions: str = "subject_end",
        sublayers: bool = False,
    ):
        self.model = model
        self.module_map = module_map
        self.patch_positions = patch_positions
        self.sites = self._build_sites(sublayers)

    def _build_sites(self, sublayers: bool) -> list[str]:
        sites = self.module_map.residual_sites()
        if sublayers:
            for i in range(1, self.module_map.num_layers + 1):
                sites.extend([f"block{i}/attn", f"block{i}/mlp"])
        return sites

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def logit_diff(logits: torch.Tensor, answer_tok: int, distractor_tok: int) -> float:
        """``logit(answer) - logit(distractor)`` at the final prompt position."""
        row = logits[0, -1].detach().to(torch.float32)
        return float(row[answer_tok] - row[distractor_tok])

    def _positions(self, ex: FactExample) -> list[int]:
        if self.patch_positions == "subject_end":
            return [ex.subject_end_pos]
        if self.patch_positions == "subject_span":
            return list(range(*ex.subject_token_span))
        raise ValueError(f"Unknown patch_positions '{self.patch_positions}'.")

    # -- core experiment ------------------------------------------------------

    @torch.no_grad()
    def fact_curve(self, ex: FactExample, counter: FactExample) -> dict | None:
        """Per-site restoration values for one fact (or ``None`` if invalid).

        Args:
            ex: The clean factual example.
            counter: The counterfactual example (same template, swapped
                subject), e.g. from :meth:`FactDataset.counterfactual`.
        """
        ids_clean = torch.tensor([ex.input_ids], device=self.model.device)
        ids_corr = torch.tensor([counter.input_ids], device=self.model.device)
        ans_tok, dis_tok = ex.answer_token_id, counter.answer_token_id

        # Clean and corrupt passes, capturing every patch site.
        store_clean, out_clean = self._capture(ids_clean)
        store_corr, out_corr = self._capture(ids_corr)

        ld_clean = self.logit_diff(out_clean.logits, ans_tok, dis_tok)
        ld_corr = self.logit_diff(out_corr.logits, ans_tok, dis_tok)

        if ld_clean <= 0 or ld_corr >= 0:
            return None  # model fails to discriminate; uninterpretable

        src_positions = self._positions(ex)
        tgt_positions = self._positions(counter)
        if len(src_positions) != len(tgt_positions):
            # Span lengths differ across tokenizations; fall back to the
            # single last-subject-token site, which always aligns.
            if self.patch_positions == "subject_span":
                src_positions = [ex.subject_end_pos]
                tgt_positions = [counter.subject_end_pos]
            else:
                return None

        denominator = ld_clean - ld_corr

        curve: dict[str, float] = {}
        for site in self.sites:
            source = store_clean[site]
            hook = make_patch_hook(source, src_positions, tgt_positions)
            module = self.module_map.resolve_site(site)
            out_patched = run_with_hooks(self.model, ids_corr, [(module, hook)])
            ld_patched = self.logit_diff(out_patched.logits, ans_tok, dis_tok)
            curve[site] = (ld_patched - ld_corr) / denominator

        return {
            "relation": ex.relation,
            "subject": ex.subject,
            "prompt": ex.prompt,
            "counterfactual_prompt": counter.prompt,
            "answer_token_id": ans_tok,
            "distractor_token_id": dis_tok,
            "ld_clean": ld_clean,
            "ld_corrupt": ld_corr,
            "restoration": curve,
        }

    def _capture(self, input_ids: torch.Tensor):
        store: dict = {}
        hooks = make_capture_many(self.module_map, self.sites, store)
        output = run_with_hooks(self.model, input_ids, hooks)
        return store, output

    # -- dataset-level driver ---------------------------------------------------

    @torch.no_grad()
    def dataset_curves(
        self,
        dataset: FactDataset,
        template_idx: int = 0,
        known_only: bool = True,
        known_flags: dict | None = None,
        max_facts: int | None = None,
        progress: bool = True,
    ) -> tuple[list[dict], int]:
        """Run :meth:`fact_curve` over every (relation, fact) pair.

        Args:
            template_idx: Which template to analyze (canonical = 0).
            known_only: Skip facts whose final-logits top-1 is not the answer
                token; requires ``known_flags`` keyed by
                ``(relation, subject)`` (from the lens run) when provided,
                otherwise a quick top-1 check is run inline.
            max_facts: Cap on the number of facts analyzed.
            progress: Show a tqdm bar.

        Returns:
            ``(valid_curves, n_skipped)``.
        """
        curves: list[dict] = []
        skipped = 0
        examples = self._analysis_examples(dataset, template_idx)
        if max_facts is not None:
            examples = examples[:max_facts]
        iterator = tqdm(examples, desc="patching", disable=not progress)
        for ex in iterator:
            if known_only:
                known = self._is_known(ex, known_flags)
                if not known:
                    skipped += 1
                    continue
            counter = dataset.counterfactual(ex)
            result = self.fact_curve(ex, counter)
            if result is None:
                skipped += 1
            else:
                curves.append(result)
        return curves, skipped

    def _analysis_examples(self, dataset: FactDataset, template_idx: int) -> list[FactExample]:
        out: list[FactExample] = []
        for rel_name in dataset.relation_names:
            for ex in dataset.examples_for(rel_name):
                if ex.template_idx == template_idx:
                    out.append(ex)
        return out

    def _is_known(self, ex: FactExample, known_flags: dict | None) -> bool:
        if known_flags and (ex.relation, ex.subject) in known_flags:
            return bool(known_flags[(ex.relation, ex.subject)])
        ids = torch.tensor([ex.input_ids], device=self.model.device)
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits
        return int(logits[0, -1].argmax()) == ex.answer_token_id
