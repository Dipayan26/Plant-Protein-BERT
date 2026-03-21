"""Logging setup: WandB and standard Python logging."""

from __future__ import annotations

import logging

import wandb
from omegaconf import DictConfig, OmegaConf


def setup_logger(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def init_wandb(cfg: DictConfig, job_type: str) -> None:
    """Initialize Weights & Biases run. Call once per script."""
    wandb.init(
        project=cfg.project_name,
        job_type=job_type,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
