# FactLens — Decodability vs. Causality of Factual Recall in Small Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org/)
[![transformers](https://img.shields.io/badge/🤗%20transformers-4.40%2B-yellow.svg)](https://huggingface.co/docs/transformers)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Abstract.** Interpretability research offers two complementary lenses on where a
> transformer stores a fact: *linear probes* report what is **decodable** from hidden
> states, and *activation patching* reports what the model **causally uses**. These
> can disagree — probes routinely detect information the model never reads out
> (Belinkov, 2022). FactLens quantifies this disagreement, site by site through the
> residual stream, with the **Causal–Probe Gap (CPG)**:
>
> <p align="center"><i>CPG(k) = D(k) − C(k)</i>,&nbsp; where&nbsp; <i>D(k)</i> = held-out probe macro-F1 at site <i>k</i>,&nbsp; <i>C(k)</i> = mean patching restoration at site <i>k</i>.</p>
>
> Both quantities live in [0, 1], so their difference marks **dormant knowledge** —
> facts that are linearly present but not causally used. On a curated 140-fact
> bank across 9 relations, GPT-2 shows a measurable dormant tail: the fact stays
> decodable at the subject token (probe F1 ≈ 0.85) long after it stops being
> causally used there (patch restoration = 0.000 at the last block), with a
> sharp handoff window in between where readout decodability rises as subject-token
> causality decays — consistent with the "FFN writes, attention routes" picture of
> Geva et al. (2021; 2022). Everything — the hook
> machinery, logit lens, probes, control tasks, patching, causal tracing, and
> metrics — is implemented **from scratch** on plain `torch.nn` hooks; no
> `transformer_lens`, `nnsight`, or `sklearn` anywhere.

---

## Table of contents

1. [Why this project exists](#why-this-project-exists)
2. [The method in 60 seconds](#the-method-in-60-seconds)
3. [Repository layout](#repository-layout)
4. [Setup](#setup)
5. [Quickstart](#quickstart)
6. [Results on GPT-2 (124M)](#results-on-gpt-2-124m)
7. [Methodology details](#methodology-details)
8. [Extending the project](#extending-the-project)
9. [Limitations](#limitations)
10. [Citing](#citing)
11. [Acknowledgements](#acknowledgements)

## Why this project exists

Ask "where does GPT-2 keep the fact that Paris is the capital of France?" and the
literature gives you three different-but-related answers:

| Lens on the question | Tool | Classic finding |
|---|---|---|
| When does the answer *surface in vocabulary space*? | **logit lens** | mid-to-late layers "promote" the answer (nostalgebraist 2021; Geva et al. 2022) |
| Where is the fact *linearly readable*? | **linear probes** | often surprisingly early — sometimes at the embedding itself |
| Where does the model *actually use* it? | **activation patching** | a mid-network window; FFN layers act as key-value memories (Geva et al. 2021), ROME locates edits in mid-MLPs (Meng et al. 2022) |

Probes are cheap and powerful but famously over-report: a probe can exploit
surface statistics (which token is this? which position?) rather than the property
of interest. The probing community's answer is **control tasks** (Hewitt & Liang,
2019) and selectivity; the causal-mechanistic community's answer is patching. What
has *not* been widely quantified is the **site-level disagreement between the two**
on the same facts, same positions, same model — even though that disagreement is
exactly where "information present but unused" lives.

FactLens makes that disagreement a first-class object:

- **D(k)** — decodability: held-out macro-F1 of a from-scratch linear probe at
  residual site *k*, probed at the **last subject token**, with a Hewitt-Liang
  control task reporting selectivity, and a **template-holdout** test split (the
  probe must generalize across surface forms, not memorize wording).
- **C(k)** — causality: mean restoration from **subject-swap activation patching**
  at site *k* (corrupt "The capital of France is" → "The capital of Germany is",
  splice clean activations at the last subject token, measure how much of the
  clean-minus-corrupt logit difference returns).
- **CPG(k) = D(k) − C(k)**, plus **onsets** (the first site where each curve
  crosses its threshold) and the **dormant window** (onset difference), reported
  per relation and aggregated.

The whole toolkit is deliberately from scratch — ~1,900 lines of commented Python
on top of raw `transformers` checkpoints — because for a research artifact the
implementation *is* the claim: if you can't see the hook, you can't audit the
experiment.

## The method in 60 seconds

```
              embed   blk1 ... blk8   blk9   blk10   blk11   blk12   → norm → unembedding
 lens           —      ...    top100↑  .38    1.0     top5↑    exact   (answer promoted late)
 D_subj(k)     0.99    1.0  ...  0.97  0.96   0.94    0.88     0.85   (identity is linear early)
 D_read(k)     0.03    0.19 ...  0.07  0.19   0.45    0.50     0.54   (rises at the handoff)
 C(k)          0.81    0.76 ...  0.83  0.74   0.45    0.16     0.00   (decays through routing)
                                                        ↑        ↑
                                              readout onset   routing offset
                                              └── handoff ───┘  → dormant tail: CPG_subj → 0.85
```

Three experiments share one 9-relation, 140-fact, 3–5-template bank
(`factlens/data/relations.py`):

1. **Logit-lens trajectories** — project every residual site through the final norm
   + unembedding (mean-centered alignment); track the answer token's logit/rank.
2. **Probes with controls** — per relation × site × position, train an AdamW linear
   probe; report task F1, control F1, selectivity, one-vs-rest AUC.
3. **Causal interventions** — subject-swap patching curves (site sweep) and
   ROME-style noise tracing (site × position heatmaps).

Then `run_cpg.py` synthesizes D and C into the gap, onsets, and dormant windows.

## Repository layout

```
factlens/
├── factlens/                    # the library (no interpretability dependencies)
│   ├── data/                    # fact bank, tokenized dataset, control tasks
│   ├── models/                  # loading + ModuleMap + all hook machinery
│   ├── lens/                    # from-scratch logit lens (LayerNorm/RMSNorm by hand)
│   ├── probes/                  # linear probe + metrics (macro-F1, tie-aware AUC)
│   ├── causality/               # activation patching + causal tracing
│   ├── analysis/                # aggregation, CPG computation, figures
│   ├── pipelines/               # experiment orchestration (lens/probe/patch/cpg)
│   ├── tracking/                # JSONL + optional tensorboard tracker
│   └── utils/                   # seeding, logging, device helpers
├── configs/
│   ├── default.yaml             # shared defaults
│   ├── models/                  # gpt2 | pythia-160m | pythia-410m | qwen2.5-0.5b
│   └── experiments/             # one YAML per analysis
├── scripts/                     # CLI entrypoints + smoke test
├── notebooks/                   # 3 methodology walkthroughs
├── figures/                     # committed example figures (regenerate with `make`)
├── results/                     # run outputs (gitignored)
├── Makefile · pyproject.toml · CITATION.cff · CONTRIBUTING.md · LICENSE
```

## Setup

```bash
git clone https://github.com/<your-github-username>/factlens.git
cd factlens
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements.txt

# optional, for the notebooks:
pip install -e ".[notebooks]"
```

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1, transformers ≥ 4.40 (any recent 4.x/5.x
works — the loader handles the `torch_dtype` → `dtype` API rename). Models are
pulled from the HuggingFace hub on first use (GPT-2 is ~500 MB).

## Quickstart

```bash
# 0) end-to-end sanity check (~2 min GPU / ~6 min CPU, GPT-2, tiny data slice)
make smoke

# 1) logit-lens trajectories over the full fact bank
make lens            # or: --model pythia-410m / qwen2.5-0.5b

# 2) layerwise probes + control tasks (the D curves)
make probes

# 3) activation patching curves (the C curve) + causal-tracing heatmaps
make patching
make trace

# 4) the headline synthesis: CPG, onsets, dormant windows
make cpg
```

Every run writes to `results/<run>/<timestamp>/` — `config.json`,
`metrics.jsonl`, analysis JSONs, figures — and (by default) copies the key figure
to `figures/`. Useful flags:

```bash
python scripts/run_cpg.py --config configs/experiments/cpg.yaml \
    --model qwen2.5-0.5b \      # swap the model
    --device cuda \             # or cpu
    --limit 6 \                 # 6 facts per relation for a fast pass
    --set probe.epochs=150 \    # dot-path config overrides
    --set patch.sites=residual+sublayers
```

## Results on GPT-2 (124M)

Everything below is the committed reference run (`make cpg lens trace`, full
140-fact bank, template 0, seed 13, float32, CPU; ~4 minutes total; artifacts
regenerate into `figures/` and `results/`). Per-fact curves are noisy and the
bank is small by design — read the numbers as a reference point, not a claim
of statistical significance (see [Limitations](#limitations)).

**Fact coverage.** 48 of 140 facts are "known" (answer first-token rank < 5 at
the final readout; 34.3%). GPT-2's greedy top-1 is often a determiner — *the*
interesting coverage split is frequency: capitals/continents are recalled,
Brasilia/Canberra and element symbols are not.

**Emergence (logit lens, known facts, mean-centered lens).** The answer climbs
from rank >100 to top-5 in the last third of the stack. "Answer in top-100"
rises first (block 8: 0.17 → block 9: 0.38 → block 10: 1.00); "answer in
top-5" follows (block 10: 0.33 → block 11: 0.69 → block 12: 1.00) — the
concept enters vocabulary space diffusely, then later sites *promote* it
(layer-wise prediction refinement, Geva et al. 2022). The trajectory's final
row equals the model's real logits by construction (identity-constrained last
site), so the last block's row is exact.

**The full CPG table (this is the headline artifact):**

| site | D_subj (F1) | ctrl_subj | D_read (F1) | ctrl_read | causality | CPG_subj |
|---|---|---|---|---|---|---|
| embed | 0.987 | 0.742 | 0.026 | 0.028 | 0.810 | +0.176 |
| block1 | 1.000 | 0.734 | 0.185 | 0.152 | 0.755 | +0.245 |
| block2 | 0.957 | 0.732 | 0.154 | 0.141 | 0.738 | +0.218 |
| block3 | 0.980 | 0.741 | 0.131 | 0.127 | 0.740 | +0.240 |
| block4 | 1.000 | 0.732 | 0.065 | 0.124 | 0.808 | +0.192 |
| block5 | 0.988 | 0.747 | 0.092 | 0.097 | 0.862 | +0.126 |
| block6 | 0.983 | 0.711 | 0.101 | 0.086 | 0.822 | +0.161 |
| block7 | 0.956 | 0.745 | 0.134 | 0.065 | 0.810 | +0.145 |
| block8 | 0.975 | 0.735 | 0.073 | 0.132 | 0.827 | +0.148 |
| block9 | 0.957 | 0.736 | 0.193 | 0.250 | 0.738 | +0.219 |
| block10 | 0.944 | 0.691 | **0.449** | 0.338 | 0.449 | +0.495 |
| block11 | 0.877 | 0.685 | 0.497 | 0.242 | **0.164** | +0.714 |
| block12 | 0.849 | 0.634 | 0.544 | 0.330 | 0.000 | **+0.849** |

with aggregate onsets **readout onset = block10** (D_read crosses chance+0.15),
**routing offset = block11** (C falls below 0.20), **handoff window = 1 site**,
and **peak dormant knowledge = 0.849 at block12** (dormant tail: 2 sites).

Reading it:

1. **Identity is cheap.** D_subj ≈ 0.99 from the embedding site — the subject
   token's identity, and hence its object, is trivially linearly present
   wherever the subject is. The control (0.73) honestly shows that a Hewitt-
   Liang control is *also* largely solvable there: most of D_subj is surface
   memorization, and the +0.25 selectivity is the learned-association part.
2. **The informative decodability curve is D_read.** Near chance (0.09) for
   the first two-thirds of the stack, then 0.45 → 0.50 → 0.54 over blocks
   10–12: the fact becomes linearly accessible at the readout position only
   once attention routes it there.
3. **Causality decays through the routing window.** C ≈ 0.8 while the
   subject token carries the fact, collapsing over blocks 10–12 to exactly
   0.000 at block12 — after routing completes, the subject token's
   representation is causally dead for the readout.
4. **The gap is the story.** CPG_subj grows through the handoff and reaches
   0.849 at the last site: after routing, the fact stays *decodable* at the
   subject token while being causally unused — dormant knowledge, consistent
   with FFN key-value memories that the readout no longer consults at the
   subject position (Geva et al. 2021/2022).

**Per-relation handoff statistics** (onset → offset, handoff window):

| relation | readout onset | routing offset | handoff |
|---|---|---|---|
| animal-young | block9 | block11 | 2 |
| capital-country | block10 | block11 | 1 |
| landmark-city | block9 | block10 | 1 |
| country-continent | block2 | block2 | 0 |
| product-company | block5 | — | — |
| language-country | block10 | — | — |

country→continent (5 classes, easiest relation) is routed almost immediately;
harder relations route 6–8 sites later; product-company and language-country
never fully release the subject token in this run (C stays above the margin)
— their facts keep being re-read from the subject position late into the
stack, which is itself a measurable property.

**Causal tracing (ROME-style).** Per-fact heatmaps (`figures/gpt2_trace_*.png`)
show peak restoration (≈1.00 for capital facts) on the last subject token at
mid-depth — the same site the patch curves sweep — while early-position
restores affect the subject's own next-token prediction rather than the fact
readout.

**A methodological finding worth keeping.** Two subtleties surfaced while
building this and are now part of the toolkit's design: (i) scalar "norm
reconciliation" for the lens is a mathematical no-op — LayerNorm/RMSNorm are
positive-scale-invariant — so FactLens uses per-site mean centering (the
translation term of an untrained tuned lens) instead; (ii) GPT-2's learned
positional embeddings leak prompt *length* into the readout site (subjects
tokenize to different lengths), which can inflate early-site readout probes —
so onset detection requires a *crossing* (a rise from below threshold), not a
high start.

## Methodology details

### Sites and naming
`embed` = output of the embedding layer; `block k` = output of the k-th block
(1-indexed). Probes and patching share this axis (L+1 sites); the lens adds a
`final` point computed from the model's real logits as a faithfulness check.

### "Known" facts
A fact enters the lens/patching aggregates when its answer's first token ranks
below 5 at the final readout (`data.known_rank_threshold`). Greedy top-1 equality
is too strict for small models — GPT-2's top-1 after "The capital of France is"
is ` the`, with ` Paris` at rank ~4. Patching additionally enforces a *direction
check*: `ld(clean) > 0 > ld(corrupt)` on the answer-vs-distractor logit
difference; facts failing it are excluded and counted (patching is only
interpretable on facts the model discriminates).

### Probe protocol
Per relation and site: features are the hidden state at the last subject token
(or the final prompt token); labels index the relation's object set; the **last
template is held out for testing**; train/val are split per class (25% val).
The probe is a single `nn.Linear` trained with AdamW (lr 1e-3, weight decay 1e-4,
batch 64, ≤300 epochs, early stopping on val macro-F1, best-state restore). The
control task (Hewitt & Liang, 2019) relabels each *subject* with a fixed random
object — structurally identical, minus the fact. Selectivity = task − control.
AUC is one-vs-rest, macro-averaged, computed from softmax probabilities with
exact tie handling (Mann-Whitney ranks).

### Logit lens
`logits^(k) = Norm(h^(k) − μ^(k)) W_U^T` where `Norm` is the checkpoint's final
LayerNorm **or** RMSNorm implemented by hand (GPT-2/Pythia vs. Qwen/Llama), and
`μ^(k)` is a per-site dataset-level mean hidden vector (mean-centered lens —
the translation term of an untrained tuned lens). Scalar norm "reconciliation"
(rescaling hidden norms to the final site's) is deliberately *not* used: it is
a mathematical no-op, because both norms are positive-scale-invariant (see the
docstring of `factlens/lens/logit_lens.py` for the derivation). The final
site's alignment is identity-constrained, so its row reproduces the model's
exact logits. Set `lens.align: none` for the raw lens.

### Patching
Corruption swaps the subject with the next same-relation fact (cyclic), keeping
the template and grammar fixed. Clean activations are captured in one pass; each
patched pass splices `block k`'s clean output at the last subject token into the
corrupt run (source and target positions are resolved independently, so
differently-tokenized prompts align correctly). Restoration is normalized to [0,1]
by the clean-minus-corrupt logit difference. `patch.sites=residual+sublayers`
additionally sweeps attention and MLP outputs inside each block.

### Causal tracing (ROME-style)
Embedding-output Gaussian noise (σ = 3 × embedding std) on the subject span;
restore single (site, position) activations from the clean run and record the
restored logit-difference fraction → site × position heatmap. Differences from
Meng et al. (2022): noise at the embedding *output* on subject positions only,
logit-diff readout, full residual-site sweep.

## Extending the project

- **Cross-architecture replication** — `--model pythia-410m` / `qwen2.5-0.5b`
  runs the identical protocol on different norms/depths; the natural paper-grade
  question is whether the dormant tail and handoff window scale with depth or corpus.
- **Sublayer decomposition** — `--set patch.sites=residual+sublayers` separates
  attention from MLP contributions to C(k).
- **Tuned lens** — replace mean centering with a learned per-site affine
  (Belrose et al. 2023) and re-measure the lens onset.
- **Mass/copying decomposition** — add the "mass" metric (norm of the object
  direction) alongside rank, following Geva et al. (2022).
- **New relations** — append a `Relation` to `factlens/data/relations.py`; every
  analysis picks it up automatically.

## Limitations

- **Scale.** Conclusions are drawn from ≤0.5B-parameter models; layerwise
  physiology plausibly changes at scale.
- **Bank size.** 129 facts × 4–5 templates is deliberately auditable, not
  comprehensive; per-relation curves have 12–20 facts behind them.
- **English + Western-skewed facts.** The bank reflects Wikipedia frequency bias.
- **Lens faithfulness.** The reconciled logit lens is an approximation; the
  tuned-lens literature documents residual distortions.
- **CPG comparability.** D and C are both in [0,1] but estimate different
  quantities (separability vs. intervention effect); the gap is best read as a
  *disagreement signal*, not a physical quantity.
- **Probe capacity.** A linear probe under-fits some relations and over-fits
  small ones; we report selectivity and template-holdout generalization to bound
  both failure modes.

## Citing

If you use FactLens in your work, please cite it (BibTeX in
[CITATION.cff](CITATION.cff)):

```bibtex
@software{factlens2026,
  title  = {FactLens: Decodability vs. Causality of Factual Recall in Small Language Models},
  author = {Cathy Li},
  year   = {2026},
  url    = {https://github.com/CathyKernel/FactLens-Decodability-vs.-Causality-of-Factual-Recall-in-Small-Language-Models},
}
```

## Acknowledgements

This project stands on the shoulders of the mechanistic-interpretability
literature: the logit lens (nostalgebraist 2021; Geva et al. 2022), causal
tracing and ROME (Meng et al. 2022), control tasks (Hewitt & Liang 2019), amnesic
probing (Elazar et al. 2021), and probing methodology generally (Belinkov 2022).
See module docstrings for per-file references. 
