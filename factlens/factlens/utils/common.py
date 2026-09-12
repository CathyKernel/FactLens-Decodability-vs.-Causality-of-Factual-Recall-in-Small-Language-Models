"""Shared utilities: seeding, device selection, logging."""

from __future__ import annotations

import logging
import random
import sys

import numpy as np
import torch


def seed_everything(seed: int = 13):
    """Seed python, numpy, and torch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str = "cpu") -> str:
    """Resolve the device string, falling back sensibly."""
    if requested == "cuda" and not torch.cuda.is_available():
        print("[utils] CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


def get_logger(name: str = "factlens") -> logging.Logger:
    """Module-level logger with a one-time console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
