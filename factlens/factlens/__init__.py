"""FactLens: decodability vs. causality of factual recall in small LLMs.

From-scratch interpretability toolkit — logit lens, linear probes with
control tasks, activation patching, and causal tracing — plus the
Causal-Probe Gap (CPG) synthesis. See README.md for the research framing.
"""

__version__ = "0.1.0"

from .analysis.cpg import compute_cpg  # noqa: F401
from .causality.patching import PatchAnalyzer  # noqa: F401
from .causality.tracing import CausalTracer  # noqa: F401
from .config import load_config  # noqa: F401
from .data.dataset import FactDataset  # noqa: F401
from .data.relations import FACT_BANK  # noqa: F401
from .lens.logit_lens import LogitLens  # noqa: F401
from .models.hooks import ModuleMap  # noqa: F401
from .probes.linear import LinearProbe  # noqa: F401
