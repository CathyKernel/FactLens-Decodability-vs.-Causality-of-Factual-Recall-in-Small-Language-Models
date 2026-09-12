"""Model loading and the from-scratch hook machinery."""

from .hooks import (  # noqa: F401
    ModuleMap,
    make_capture_hook,
    make_noise_hook,
    make_patch_hook,
    run_with_hooks,
)
from .loader import load_model_and_tokenizer  # noqa: F401
