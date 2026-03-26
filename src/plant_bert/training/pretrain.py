"""PyTorch Lightning module that runs Masked Language Model (MLM) pretraining.

What happens during one training step:
  1. The DataCollator (set in pretrain.py script) receives a batch of tokenized
     sequences and randomly replaces 15% of tokens with one of:
       - 80% of the time: [MASK] token  (the model must predict the original)
       - 10% of the time: a random token (forces robustness)
       - 10% of the time: the original token unchanged (anchors representations)
     It also sets label=-100 at non-masked positions so they are ignored in the loss.

  2. The masked batch is fed to the model.  For each masked position, the model
     outputs a score vector of length vocab_size (one score per possible token).

  3. Cross-entropy loss is computed between the predicted scores and the true
     token IDs — but ONLY at masked positions (label != -100).

  4. Backpropagation updates the model weights to improve those predictions.

What is perplexity?
  Perplexity = exp(average cross-entropy loss).  A perfect model has perplexity 1.
  A random model on a 30-token vocabulary would have perplexity ~30.
  Good protein language models achieve perplexity < 10 at convergence.
"""

from __future__ import annotations

import math

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


class MLMPretrainer(pl.LightningModule):
    """Lightning module that trains PlantProteinBERT with Masked Language Modeling."""

    def __init__(
        self,
        model: object,
        tokenizer: object,
        cfg: DictConfig,
        **kwargs,  # absorbs extra config keys (trainer, checkpoint, mlm_probability, etc.)
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        # save_hyperparameters stores cfg in the checkpoint for reproducibility.
        # We exclude model and tokenizer objects (not serializable).
        self.save_hyperparameters(ignore=["model", "tokenizer"])

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One batch: forward pass + compute MLM loss on masked positions."""
        outputs = self.model(**batch)
        # outputs.loss = cross-entropy averaged over all masked positions in the batch
        self.log("train/mlm_loss", outputs.loss, prog_bar=True, sync_dist=True)
        return outputs.loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Evaluate on a held-out batch.  No gradient computation (torch.no_grad is automatic)."""
        outputs = self.model(**batch)
        loss = outputs.loss
        # Perplexity = exp(loss).  Capped at exp(100) to prevent overflow on early training.
        perplexity = math.exp(min(loss.item(), 100))
        self.log("val/mlm_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/perplexity", perplexity, sync_dist=True)

    def configure_optimizers(self):
        """Build optimizer + learning rate scheduler.

        Uses AdamW (Adam with weight decay) with ESM-2 hyperparameters:
          - β2=0.98 (higher momentum than default 0.999 — better for transformers)
          - Linear warmup for first 2000 steps, then linear decay.
        """
        optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())
        scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            # "interval": "step" updates the LR every batch, not every epoch
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
