"""Logging setup for training scripts.

setup_logger() configures the standard Python logging module so that log
messages from all modules appear with a consistent timestamp format.

init_wandb() is optional — call it to track experiments on Weights & Biases
(wandb.ai).  WandB records loss curves, hyperparameters, and system metrics
so you can compare runs visually.  Requires: pip install wandb && wandb login
"""

from __future__ import annotations

import logging

from omegaconf import DictConfig, OmegaConf


def setup_logger(cfg: DictConfig) -> None:
    """Configure Python logging with a timestamp prefix."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def init_wandb(cfg: DictConfig, job_type: str) -> None:
    """Initialize a Weights & Biases experiment tracking run.

    Call once at the start of a training script if you want WandB logging.
    The full Hydra config is uploaded as hyperparameters for reproducibility.
    """
    import wandb  # imported lazily so wandb is optional
    wandb.init(
        project=cfg.project_name,
        job_type=job_type,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
