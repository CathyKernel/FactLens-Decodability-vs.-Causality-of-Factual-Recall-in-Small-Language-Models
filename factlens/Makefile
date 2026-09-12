# FactLens — Makefile
# Small models only: everything below runs on a single consumer GPU (or CPU with patience).

PYTHON ?= python3
CONFIG_DIR := configs
RESULTS_DIR := results

.PHONY: help install smoke lens probes patching trace cpg all clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in editable mode with core deps
	$(PYTHON) -m pip install -e . -r requirements.txt

smoke:  ## End-to-end sanity check on GPT-2 with 4 facts (~2 min on GPU)
	$(PYTHON) scripts/smoke_test.py

lens:  ## Run logit-lens trajectory analysis (GPT-2 default)
	$(PYTHON) scripts/run_logit_lens.py --config $(CONFIG_DIR)/experiments/logit_lens.yaml

probes:  ## Run layerwise linear probes + control tasks (GPT-2 default)
	$(PYTHON) scripts/run_probes.py --config $(CONFIG_DIR)/experiments/probes.yaml

patching:  ## Run activation patching / causal tracing (GPT-2 default)
	$(PYTHON) scripts/run_patching.py --config $(CONFIG_DIR)/experiments/patching.yaml

trace:  ## Run ROME-style causal tracing heatmap (GPT-2 default)
	$(PYTHON) scripts/run_patching.py --config $(CONFIG_DIR)/experiments/tracing.yaml

cpg:  ## Full pipeline: probes + patching + CPG synthesis (GPT-2 default)
	$(PYTHON) scripts/run_cpg.py --config $(CONFIG_DIR)/experiments/cpg.yaml

all: lens probes patching cpg  ## Run every analysis for the default model

clean:  ## Remove regenerated results
	rm -rf $(RESULTS_DIR)
