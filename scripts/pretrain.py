"""Step 3: MLM pretraining.

Usage:
    # Dev sanity check (fast)
    python scripts/pretrain.py +experiment=dev_run

    # Full pretrain
    python scripts/pretrain.py +experiment=pretrain_base

    # Custom overrides
    python scripts/pretrain.py model=bert_large training.trainer.devices=-1

    # Hyperparameter sweep
    python scripts/pretrain.py --multirun model=bert_small,bert_base training.optimizer.lr=1e-4,5e-5
"""

import hydra
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from plant_bert.utils import seed_everything, setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logger(cfg)
    seed_everything(cfg.seed)

    tokenizer = instantiate(cfg.tokenizer)
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)
    trainer_module = instantiate(cfg.training, model=model, tokenizer=tokenizer, cfg=cfg.training)

    callbacks = [
        ModelCheckpoint(**cfg.training.checkpoint),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(**cfg.training.trainer, callbacks=callbacks)
    trainer.fit(trainer_module, datamodule=datamodule)


if __name__ == "__main__":
    main()
