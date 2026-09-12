"""ROME-style causal tracing, from scratch.

Causal tracing (Meng et al., 2022) complements subject-swap patching with a
*positional* view. The clean run's subject embeddings are corrupted with
Gaussian noise; then a single (site, position) activation is restored from
the clean run, one at a time, and the change in the answer logit difference
is recorded. Plotting restoration over (site x position) yields the classic
"indirect effect" heatmap: which token's representation at which depth
carries the fact.

Differences from Meng et al.: (i) noise is applied to the embedding *output*
at the subject span only, (ii) the readout is the same answer-vs-distractor
logit difference used by :mod:`factlens.causality.patching`, and (iii) we
sweep full residual sites (``embed`` .. ``blockL``) rather than every MLP
mid-layer. The implementation is ~100 lines of hooks.

Reference
---------
Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing
Factual Associations in GPT. NeurIPS 2022. arXiv:2202.05262.
"""

from __future__ import annotations

import torch
from tqdm import tqdm

from ..data.dataset import FactDataset, FactExample
from ..models.hooks import (
    ModuleMap,
    make_capture_many,
    make_noise_hook,
    make_patch_hook,
    run_with_hooks,
)


class CausalTracer:
    """Single-position restoration traces under embedding noise."""

    def __init__(self, model, module_map: ModuleMap, noise_std_mult: float = 3.0):
        self.model = model
        self.module_map = module_map
        self.noise_std_mult = noise_std_mult
        self.sites = module_map.residual_sites()

    @staticmethod
    def logit_diff(logits: torch.Tensor, answer_tok: int, distractor_tok: int) -> float:
        row = logits[0, -1].detach().to(torch.float32)
        return float(row[answer_tok] - row[distractor_tok])

    @torch.no_grad()
    def trace_example(self, ex: FactExample, counter: FactExample, seed: int = 0) -> dict | None:
        """Trace one fact.

        Returns:
            Dict with the restoration matrix ``(n_sites, T)`` plus metadata,
            or ``None`` when the corruption or the fact is invalid.
        """
        ids = torch.tensor([ex.input_ids], device=self.model.device)
        ans_tok, dis_tok = ex.answer_token_id, counter.answer_token_id
        subject_positions = list(range(*ex.subject_token_span))

        # Clean pass, capturing every site at every position.
        store_clean, out_clean = self._capture(ids)
        ld_clean = self.logit_diff(out_clean.logits, ans_tok, dis_tok)

        # Noise scale: multiple of the clean embedding std over the prompt.
        embed_out = store_clean["embed"][0].to(torch.float32)
        noise_std = float(embed_out.std()) * self.noise_std_mult

        # Corrupted pass: noise the embedding output at the subject span.
        noise_hook = make_noise_hook(noise_std, subject_positions)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        torch.manual_seed(seed)  # reproducible randn inside the hook
        out_noised = run_with_hooks(self.model, ids, [(self.module_map.embed, noise_hook)])
        ld_noised = self.logit_diff(out_noised.logits, ans_tok, dis_tok)

        if ld_clean <= 0:
            return None  # fact unknown to the model
        if ld_noised >= ld_clean:
            # Noise failed to suppress the fact; trace is uninformative.
            return None

        n_positions = ids.shape[1]
        restoration = torch.zeros(len(self.sites), n_positions)
        for si, site in enumerate(self.sites):
            source = store_clean[site]
            module = self.module_map.resolve_site(site)
            for pos in range(n_positions):
                hook = make_patch_hook(source, [pos], [pos])
                out_patched = run_with_hooks(self.model, ids, [(module, hook)])
                ld_patched = self.logit_diff(out_patched.logits, ans_tok, dis_tok)
                restoration[si, pos] = (ld_patched - ld_noised) / (ld_clean - ld_noised)

        return {
            "relation": ex.relation,
            "subject": ex.subject,
            "prompt": ex.prompt,
            "sites": self.sites,
            "token_strs": [
                self._decode_id(t) for t in ex.input_ids
            ],
            "subject_positions": subject_positions,
            "ld_clean": ld_clean,
            "ld_noised": ld_noised,
            "restoration": restoration.tolist(),
        }

    def _decode_id(self, token_id: int) -> str:
        tok = getattr(self, "_tokenizer", None)
        if tok is None:
            return str(token_id)
        return tok.decode([token_id])

    def attach_tokenizer(self, tokenizer) -> None:
        """Attach a tokenizer for readable position labels in outputs."""
        self._tokenizer = tokenizer

    def _capture(self, input_ids: torch.Tensor):
        store: dict = {}
        hooks = make_capture_many(self.module_map, self.sites, store)
        output = run_with_hooks(self.model, input_ids, hooks)
        return store, output

    @torch.no_grad()
    def dataset_traces(
        self,
        dataset: FactDataset,
        template_idx: int = 0,
        max_facts: int | None = 8,
        known_flags: dict | None = None,
        progress: bool = True,
    ) -> list[dict]:
        """Trace up to ``max_facts`` facts known to the model.

        Returns a list of per-fact trace dicts (see :meth:`trace_example`).
        """
        traces: list[dict] = []
        candidates: list[FactExample] = []
        for rel_name in dataset.relation_names:
            for ex in dataset.examples_for(rel_name):
                if ex.template_idx == template_idx:
                    candidates.append(ex)
        if max_facts is not None:
            candidates = candidates[: max_facts * 2]  # some may be unknown
        for ex in tqdm(candidates, desc="tracing", disable=not progress):
            if known_flags is not None and not known_flags.get(
                (ex.relation, ex.subject), False
            ):
                continue
            counter = dataset.counterfactual(ex)
            trace = self.trace_example(ex, counter, seed=13)
            if trace is not None:
                traces.append(trace)
            if max_facts is not None and len(traces) >= max_facts:
                break
        return traces
