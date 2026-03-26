"""Step 3 — MLM pretraining: train PlantProteinBERT on plant protein sequences.

This is the main training script.  It wires together:
  tokenizer   → converts protein sequences to token IDs
  model       → PlantProteinBERT (randomly initialized transformer)
  datamodule  → loads sequences from the HDF5 file in batches
  collator    → randomly masks 15% of tokens per batch (the MLM task)
  trainer     → PyTorch Lightning training loop with checkpointing and LR logging

After this runs, the model checkpoint can be used for fine-tuning on downstream tasks.

Usage:
    # Quick sanity check (~1000 steps, tiny model, small batch — verifies pipeline works)
    python scripts/pretrain.py +experiment=dev_run

    # Full ESM-2-matched pretraining runs
    python scripts/pretrain.py +experiment=pretrain_plant_8m
    python scripts/pretrain.py +experiment=pretrain_plant_35m
    python scripts/pretrain.py +experiment=pretrain_plant_150m

    # Override any config value on the command line
    python scripts/pretrain.py +experiment=pretrain_plant_8m training.trainer.devices=2
    python scripts/pretrain.py --multirun training.optimizer.lr=1e-4,4e-4  # sweep
"""

import torch
import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers import DataCollatorForLanguageModeling

from plant_bert.utils import seed_everything, setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Use TensorFloat32 on Ampere GPUs (A100, RTX 3090, etc.) for ~2x speedup
    torch.set_float32_matmul_precision("high")
    setup_logger(cfg)
    seed_everything(cfg.seed)

    tokenizer = instantiate(cfg.tokenizer)
    model = instantiate(cfg.model)

    # DataCollatorForLanguageModeling does two jobs in one call per batch:
    #   1. Dynamic padding: pads sequences to the longest one in the batch
    #      (much more efficient than padding everything to 1024).
    #   2. MLM masking: randomly replaces 15% of tokens with [MASK], a random
    #      token, or the original token.  Sets label=-100 at non-masked positions.
    # It is set as the DataLoader's collate_fn so workers do this in parallel.
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer._tokenizer,
        mlm=True,
        mlm_probability=cfg.training.mlm_probability,
    )

    datamodule = instantiate(cfg.data, tokenizer=tokenizer)
    # Set collate_fn after instantiation to avoid Hydra trying to serialize it
    datamodule.collate_fn = data_collator

    # MLMPretrainer wraps the model and defines training/validation steps
    trainer_module = instantiate(cfg.training, model=model, tokenizer=tokenizer, cfg=cfg.training, _recursive_=False)

    callbacks = [
        # Saves the best checkpoints based on val/mlm_loss
        ModelCheckpoint(**cfg.training.checkpoint),
        # Logs learning rate to TensorBoard/WandB at every step
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(**cfg.training.trainer, callbacks=callbacks)
    trainer.fit(trainer_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
