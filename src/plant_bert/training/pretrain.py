"""MLM pretraining Lightning Module.

MLM masking follows BERT paper: 15% of tokens masked, of which:
  80% → [MASK], 10% → random token, 10% → unchanged.
"""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import DataCollatorForLanguageModeling


class MLMPretrainer(pl.LightningModule):
    def __init__(
        self,
        model: object,
        tokenizer: object,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer._tokenizer,
            mlm=True,
            mlm_probability=cfg.mlm_probability,
        )
        self.save_hyperparameters(ignore=["model", "tokenizer"])

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        self.log("train/mlm_loss", outputs.loss, prog_bar=True, sync_dist=True)
        return outputs.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self.model(**batch)
        loss = outputs.loss
        perplexity = math.exp(min(loss.item(), 100))
        self.log("val/mlm_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", perplexity, sync_dist=True)

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
        scheduler = instantiate(
            self.cfg.scheduler,
            optimizer=optimizer,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
