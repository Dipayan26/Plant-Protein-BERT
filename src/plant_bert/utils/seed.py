"""Reproducibility utilities.

Setting a fixed random seed makes training runs reproducible — the same seed
produces the same weight initialization, same data shuffle order, and same
masking patterns, so you can compare experiments fairly.
"""

from __future__ import annotations

import pytorch_lightning as pl


def seed_everything(seed: int) -> None:
    """Seed all RNG sources (Python, NumPy, PyTorch, CUDA, DataLoader workers)."""
    # pl.seed_everything already seeds random, numpy, torch, and torch.cuda.
    # workers=True also seeds DataLoader worker processes.
    pl.seed_everything(seed, workers=True)
