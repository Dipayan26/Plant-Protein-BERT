"""Step 4 — Fine-tuning: adapt a pretrained model to a specific protein task.

Fine-tuning takes a pretrained checkpoint (from Step 3) and continues training
it on a labeled downstream dataset.  The model learns to predict task-specific
labels while retaining the protein language knowledge from pretraining.

EarlyStopping stops training when the monitored metric stops improving for
3 consecutive validation checks, preventing overfitting.

Comparison experiments:
  - PlantProteinBERT: pretrained on plant-only data (our model)
  - ESM-2: pretrained on all UniRef50 (Meta's general-purpose model)
  Both use the same fine-tuning setup so the comparison is fair.

Usage:
    # PlantProteinBERT fine-tuning (supply pretrain checkpoint)
    python scripts/finetune.py +experiment=finetune_plant_bert_8m \
        training.pretrained_checkpoint=outputs/2026-03-21/14-00-00/checkpoints/pretrain/last.ckpt

    # ESM-2 fine-tuning (weights downloaded automatically from HuggingFace)
    python scripts/finetune.py +experiment=finetune_esm2_8m
"""

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from plant_bert.utils import seed_everything, setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logger(cfg)
    seed_everything(cfg.seed)

    tokenizer = instantiate(cfg.tokenizer)
    model = instantiate(cfg.model)       # loads pretrained weights if checkpoint is set
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)
    finetuner = instantiate(cfg.training, model=model, cfg=cfg.training, _recursive_=False)

    callbacks = [
        ModelCheckpoint(**cfg.training.checkpoint),
        # Stop training early if val metric stops improving for 3 checks
        EarlyStopping(
            monitor=cfg.training.checkpoint.monitor,
            patience=3,
            mode=cfg.training.checkpoint.mode,
        ),
    ]

    trainer = pl.Trainer(**cfg.training.trainer, callbacks=callbacks)
    trainer.fit(finetuner, datamodule=datamodule)
    trainer.test(finetuner, datamodule=datamodule)   # final evaluation on held-out test set


if __name__ == "__main__":
    main()
