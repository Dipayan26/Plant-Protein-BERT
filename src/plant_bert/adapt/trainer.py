"""Lightning module and callbacks for ESM-2 domain-adaptive continued pretraining."""

from __future__ import annotations

import logging
import math

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class ESM2Adapter(pl.LightningModule):
    """Runs MLM continued pretraining on a pretrained ESM-2 model.

    Structurally identical to MLMPretrainer (training/pretrain.py) with one
    addition: tracks the number of tokens seen so TokenBudgetCallback can stop
    training precisely at the configured token budget rather than at a fixed
    step count.
    """

    def __init__(
        self,
        model: object,
        tokenizer: object,
        cfg: DictConfig,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model", "tokenizer"])
        self._tokens_seen: int = 0

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self.model(**batch)
        # count non-padding tokens processed this step
        self._tokens_seen += int(batch["attention_mask"].sum().item())
        self.log("train/mlm_loss", outputs.loss, prog_bar=True, sync_dist=True)
        self.log(
            "train/tokens_B",
            self._tokens_seen / 1e9,
            prog_bar=True,
            sync_dist=False,
            on_step=True,
            on_epoch=False,
        )
        return outputs.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self.model(**batch)
        loss = outputs.loss
        perplexity = math.exp(min(loss.item(), 100))
        self.log("val/mlm_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", perplexity, sync_dist=True)

        preds = outputs.logits.argmax(-1)
        mask = batch["labels"] != -100
        acc = (preds[mask] == batch["labels"][mask]).float().mean()
        self.log("val/masked_token_acc", acc, sync_dist=True)

        if batch_idx == 0 and torch.cuda.is_available():
            self.log("sys/gpu_mem_GB", torch.cuda.memory_allocated() / 1e9, sync_dist=False)
            self.log("sys/gpu_mem_peak_GB", torch.cuda.max_memory_allocated() / 1e9, sync_dist=False)
            torch.cuda.reset_peak_memory_stats()

    def configure_optimizers(self):
        optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class TokenBudgetCallback(pl.Callback):
    """Stop training after a fixed number of tokens have been processed.

    More reproducible than max_steps because the actual tokens/step varies
    with sequence length distribution.  Set token_budget in the training config;
    set trainer.max_steps to a large ceiling so Lightning does not stop first.

    Example (training yaml):
        token_budget: 1_600_000_000   # 1.6 B tokens
        trainer:
          max_steps: 999_999          # ceiling — TokenBudgetCallback stops first
    """

    def __init__(self, token_budget: int) -> None:
        self.token_budget = token_budget

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: ESM2Adapter, outputs, batch, batch_idx: int
    ) -> None:
        if pl_module._tokens_seen >= self.token_budget:
            log.info(
                "Token budget reached: %.2fB / %.2fB — stopping training.",
                pl_module._tokens_seen / 1e9,
                self.token_budget / 1e9,
            )
            trainer.should_stop = True
