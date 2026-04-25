"""Domain-adaptive continued pretraining of ESM-2 on Viridiplantae protein sequences.

Takes a pretrained ESM-2 model from HuggingFace Hub and continues MLM pretraining
on the 19.9M plant-specific sequences in outputs/processed/trembl_full/sequences.h5.
The adapted checkpoint can then be fine-tuned on downstream plant protein tasks and
compared directly against the vanilla ESM-2 baseline.

This script is completely independent of scripts/pretrain.py and shares no config
with the scratch pretraining pipeline.

Usage:
    # 8M model — fast baseline (~1 day on RTX 3060)
    python scripts/adapt_esm2.py +experiment=adapt_esm2_8m

    # 35M model — recommended starting point
    python scripts/adapt_esm2.py +experiment=adapt_esm2_35m

    # 150M model — best quality / VRAM tradeoff for full-parameter training
    python scripts/adapt_esm2.py +experiment=adapt_esm2_150m

    # Override any config value
    python scripts/adapt_esm2.py +experiment=adapt_esm2_150m training.optimizer.lr=1e-5

    # Hyperparameter sweep
    python scripts/adapt_esm2.py --multirun \\
        +experiment=adapt_esm2_150m \\
        training.optimizer.lr=5e-6,1e-5,2e-5
"""

import torch
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import EsmTokenizer, DataCollatorForLanguageModeling

from plant_bert.adapt.trainer import TokenBudgetCallback
from plant_bert.utils import seed_everything, setup_logger


@hydra.main(config_path="../configs/esm2_adapt", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    setup_logger(cfg)
    seed_everything(cfg.seed)

    # ESM-2 tokenizer is fetched from the same Hub checkpoint as the model.
    # It uses amino acid character-level tokenization (33 tokens) with
    # <cls> and <eos> boundary tokens — no custom wrapper needed.
    tokenizer = EsmTokenizer.from_pretrained(cfg.model.hub_name)

    # Instantiate PlantESM2 — downloads pretrained weights on first run,
    # cached to ~/.cache/huggingface/ for subsequent runs.
    model = instantiate(cfg.model)

    # DataCollatorForLanguageModeling: pads each batch to its longest sequence
    # (dynamic padding) and randomly masks 15% of tokens for the MLM objective.
    # Special tokens (<cls>, <eos>, <pad>, <mask>) are never masked.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.training.mlm_probability,
    )

    # Reuse the existing UniProtDataModule — it already handles HDF5 streaming,
    # train/val/test splits, and multi-worker loading.  The ESM-2 tokenizer is
    # passed in place of the custom AminoAcidTokenizer; the call signature is
    # identical (tokenizer(sequence, max_length=..., truncation=True, ...)).
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)
    datamodule.collate_fn = data_collator

    trainer_module = instantiate(
        cfg.training,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg.training,
        _recursive_=False,
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        notes=cfg.wandb.notes,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    callbacks = [
        ModelCheckpoint(**cfg.training.checkpoint),
        LearningRateMonitor(logging_interval="step"),
        TokenBudgetCallback(token_budget=cfg.training.token_budget),
    ]

    trainer = pl.Trainer(**cfg.training.trainer, callbacks=callbacks, logger=wandb_logger)

    # Log gradient norms per layer every 500 steps — useful for detecting
    # catastrophic forgetting during continued pretraining.
    wandb_logger.watch(trainer_module, log="gradients", log_freq=500, log_graph=False)

    trainer.fit(trainer_module, datamodule=datamodule)

    # Export adapted weights in HuggingFace format alongside the Lightning checkpoint.
    # This lets downstream code load the model with EsmForMaskedLM.from_pretrained().
    hf_out = cfg.training.checkpoint.dirpath + "/hf_model"
    model.save_hf_checkpoint(hf_out)


if __name__ == "__main__":
    main()
