# Contributing to FactLens

Thanks for your interest in improving this research artifact!

## Ground rules

- **Scientific transparency over convenience.** The point of this repo is that
  every intervention is auditable in plain PyTorch. Do not add interpretability
  framework dependencies (`transformer_lens`, `nnsight`, `pyvene`) to core
  analyses — if an external tool genuinely helps, put it behind an optional
  extra and document what it replaces.
- **From-scratch means auditable.** Metrics in `factlens/probes/metrics.py`
  intentionally avoid sklearn. If you change a metric, justify it in the
  docstring and keep the numerics readable.
- **Determinism.** Experiments should reproduce under a fixed seed. Any new
  randomness must flow through `factlens.utils.common.seed_everything` or an
  explicit `random.Random(seed)`.

## Workflow

1. Fork, create a feature branch.
2. Install dev tooling: `pip install -e ".[dev]"`.
3. Make your change; keep functions documented (the docstrings are the paper's
   methods section).
4. Run the smoke test: `python scripts/smoke_test.py` — it must pass on CPU
   with GPT-2 before any PR.
5. If you touched an analysis, regenerate its example figure and commit it
   under `figures/` with the config you used.
6. Format with black + ruff (`black factlens scripts && ruff check factlens scripts`).
7. Open a PR describing *what claim the change affects* (e.g., "probe protocol",
   "patching corruption"), not just the code diff.

## Good first contributions

- New relations for the fact bank (follow the selection criteria in
  `factlens/data/relations.py`; keep facts unambiguous and time-stable).
- Model configs for other small checkpoints (follow `configs/models/*.yaml`;
  document BOS handling!).
- The tuned-lens extension described in the README's *Extending* section.
- A per-relation results table generator for the README.

## Reporting problems

Open an issue with: model slug, config file, commit hash, and the relevant
`results/<run>/<timestamp>/config.json`. Output artifacts are plain JSON —
attach them.
