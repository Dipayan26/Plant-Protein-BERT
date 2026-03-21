"""Step 4: Fine-tune on a downstream task.

Usage:
    python scripts/finetune.py +experiment=finetune_go_terms \
        training.pretrained_checkpoint=outputs/2026-03-21/14-00-00/checkpoints/pretrain/last.ckpt
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
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)
    finetuner = instantiate(cfg.training, model=model, cfg=cfg.training)

    callbacks = [
        ModelCheckpoint(**cfg.training.checkpoint),
        EarlyStopping(monitor=cfg.training.checkpoint.monitor, patience=3, mode=cfg.training.checkpoint.mode),
    ]

    trainer = pl.Trainer(**cfg.training.trainer, callbacks=callbacks)
    trainer.fit(finetuner, datamodule=datamodule)
    trainer.test(finetuner, datamodule=datamodule)


if __name__ == "__main__":
    main()
