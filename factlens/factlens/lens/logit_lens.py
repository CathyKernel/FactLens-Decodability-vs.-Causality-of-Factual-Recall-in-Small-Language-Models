"""The logit lens, implemented from scratch.

The lens projects *intermediate* residual-stream states onto the vocabulary
(nostalgebraist, 2021; Geva et al., 2022):

    logits^(k) = Norm(h^(k)) @ W_U^T

where ``h^(k)`` is the residual stream after site ``k`` (embedding output or
a block output), ``Norm`` is the model's *final* norm (LayerNorm for GPT-2 /
Pythia, RMSNorm for Qwen / Llama), and ``W_U`` is the unembedding matrix.

A note on norm "reconciliation" (a subtlety worth writing down)
--------------------------------------------------------------
A popular folk remedy for lens distortion is to rescale each site's hidden
norm to the final site's norm before the final norm. For LayerNorm and
RMSNorm this is a **mathematical no-op**: both norms are invariant to
positive rescaling of their input (``Norm(a·h) = Norm(h)`` for a > 0), so
the scalar cancels before the unembedding ever sees it. We verified this
empirically; the distortion the *tuned* lens fixes is not scalar but
**translational/anisotropic** — intermediate residual distributions sit off
the origin in directions the unembedding was never calibrated for.

FactLens therefore offers the honest parameter-free stand-in: per-site
**mean centering**. A dataset-level mean hidden vector per site is
subtracted before the final norm (the translation term of an untrained
tuned lens), which sharpens mid-stack readouts without learning anything.
Set ``lens.align: none`` for the raw lens.

The trajectory always includes the model's *real* logits as its final
point (``site == "final"``), which doubles as a faithfulness check.

References
----------
nostalgebraist (2021). Interpreting GPT: the logit lens. LessWrong.
Geva, M., Schuster, R., Berant, J., & Levy, O. (2022). Transformer
Feed-Forward Layers Are Key-Value Memories. EMNLP 2022.
Belrose, N. et al. (2023). Eliciting Latent Predictions from Transformers
with the Tuned Lens. arXiv:2303.08112.
"""

from __future__ import annotations

import torch

from ..models.hooks import ModuleMap, make_capture_many, run_with_hooks


class LogitLens:
    """Project residual-stream states to vocabulary logits.

    Args:
        module_map: Resolved model internals.
        site_means: Optional mapping ``site -> (d,)`` mean hidden vector,
            subtracted before the final norm (mean-centered lens). Build
            with :func:`estimate_site_means`.
    """

    def __init__(self, module_map: ModuleMap, site_means: dict | None = None):
        # Unembedding: (vocab, d). Kept in float32 for stable projections.
        self.W_U: torch.Tensor = module_map.lm_head.weight.detach().to(torch.float32)
        norm = module_map.final_norm
        self.norm_w = norm.weight.detach().to(torch.float32) if norm.weight is not None else None
        bias = getattr(norm, "bias", None)
        self.norm_b = bias.detach().to(torch.float32) if bias is not None else None
        self.norm_kind = module_map.norm_kind
        self.norm_eps = module_map.norm_eps
        self.site_means = {
            k: v.to(torch.float32) for k, v in (site_means or {}).items()
        }

    # -- from-scratch normalization ------------------------------------------

    def apply_norm(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the model's final norm (LayerNorm or RMSNorm) by hand."""
        h = h.to(torch.float32)
        if self.norm_kind == "rms":
            h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + self.norm_eps)
            if self.norm_w is not None:
                h = h * self.norm_w
            return h
        mean = h.mean(dim=-1, keepdim=True)
        var = h.var(dim=-1, unbiased=False, keepdim=True)
        h = (h - mean) / torch.sqrt(var + self.norm_eps)
        if self.norm_w is not None:
            h = h * self.norm_w
        if self.norm_b is not None:
            h = h + self.norm_b
        return h

    # -- projection ------------------------------------------------------------

    def project(self, h: torch.Tensor, site: str | None = None) -> torch.Tensor:
        """Hidden state(s) ``(..., d)`` -> vocabulary logits ``(..., V)``.

        Args:
            site: Site name whose mean should be subtracted (mean-centered
                lens); ``None`` applies the raw lens.
        """
        h = h.to(torch.float32)
        if site is not None and site in self.site_means:
            h = h - self.site_means[site]
        return self.apply_norm(h) @ self.W_U.t()


@torch.no_grad()
def answer_rank(logits: torch.Tensor, answer_token_id: int) -> int:
    """0-based rank of the answer token in descending logit order."""
    answer_logit = logits[answer_token_id]
    return int((logits > answer_logit).sum().item())


@torch.no_grad()
def estimate_site_means(
    model,
    module_map: ModuleMap,
    examples,
    device: str = "cpu",
    max_prompts: int = 64,
) -> dict[str, torch.Tensor]:
    """Dataset-level mean hidden vector per site (for mean centering).

    One forward pass per prompt (up to ``max_prompts``), averaging every
    captured site over *all* positions. Using all positions rather than only
    the readout keeps the estimate from overfitting the template's final
    token.

    Following the tuned-lens convention, the alignment for the *final* block
    site is constrained to the identity (zero mean): the last site's hidden
    state already feeds the model's real norm+unembedding, so its lens row
    reproduces the exact readout and the trajectory converges to it.
    """
    sites = module_map.residual_sites()
    final_site = sites[-1]
    sums = {site: None for site in sites}
    n_tokens = 0
    for ex in examples[:max_prompts]:
        ids = torch.tensor([ex.input_ids], device=device)
        store: dict = {}
        hooks = make_capture_many(module_map, sites, store)
        run_with_hooks(model, ids, hooks)
        for site in sites:
            hidden = store[site][0].to(torch.float32).sum(dim=0)  # (T, d) -> (d,)
            sums[site] = hidden if sums[site] is None else sums[site] + hidden
        n_tokens += ids.shape[1]
    means = {site: (total / n_tokens) for site, total in sums.items()}
    means[final_site] = torch.zeros_like(means[final_site])  # identity constraint
    return means


@torch.no_grad()
def scan_example(
    model,
    module_map: ModuleMap,
    lens: LogitLens,
    example,
    input_ids: torch.Tensor,
    known_rank_threshold: int = 5,
) -> dict:
    """Compute the lens trajectory for one factual prompt.

    Runs a single forward pass capturing every residual site, then projects
    each site's hidden state at the readout position (the final prompt token)
    through the lens.

    Args:
        known_rank_threshold: A fact counts as *known* when the answer's
            first token ranks below this at the final readout. Greedy top-1
            is too strict for small models — GPT-2's top-1 after "The
            capital of France is" is " the", with " Paris" at rank ~4.

    Returns:
        Dict with, per site: ``answer_logit``, ``answer_rank``, ``top1``
        token id, plus the exact final logits row and a ``known`` flag.
    """
    sites = module_map.residual_sites()
    store, output = _capture_all_sites(model, module_map, input_ids, sites)

    final_logits = output.logits[0, -1].detach().to(torch.float32)
    answer_tok = example.answer_token_id

    trajectory = []
    for site in sites:
        h = store[site][0, example.last_pos]
        logits = lens.project(h, site=site)
        trajectory.append(
            {
                "site": site,
                "answer_logit": float(logits[answer_tok]),
                "answer_rank": answer_rank(logits, answer_tok),
                "top1": int(logits.argmax()),
            }
        )
    # Exact readout from the model itself.
    trajectory.append(
        {
            "site": "final",
            "answer_logit": float(final_logits[answer_tok]),
            "answer_rank": answer_rank(final_logits, answer_tok),
            "top1": int(final_logits.argmax()),
        }
    )
    return {
        "relation": example.relation,
        "subject": example.subject,
        "prompt": example.prompt,
        "answer_token_id": answer_tok,
        "known": answer_rank(final_logits, answer_tok) < known_rank_threshold,
        "trajectory": trajectory,
    }


def _capture_all_sites(
    model, module_map: ModuleMap, input_ids: torch.Tensor, sites: list[str]
):
    """Forward pass capturing all requested residual sites."""
    store: dict = {}
    hooks = make_capture_many(module_map, sites, store)
    output = run_with_hooks(model, input_ids, hooks)
    return store, output
