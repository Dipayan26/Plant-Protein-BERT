"""PyTorch Lightning module for fine-tuning on downstream protein tasks.

Fine-tuning pipeline:
  1. Load the pretrained PlantProteinBERT (or ESM-2) encoder.
  2. Attach a small classification head on top.
  3. Feed labeled protein sequences; the encoder extracts representations and
     the head predicts task-specific labels.
  4. Backpropagation updates both the encoder (unless frozen) and the head.

Supported task types (set via cfg.problem_type):
  multi_label_classification  — each protein can have multiple correct labels
                                e.g. GO terms: a protein can be in nucleus AND
                                bind DNA AND have kinase activity
                                Loss: BCEWithLogitsLoss (binary cross-entropy per label)
                                Metric: AUROC (area under ROC curve)

  single_label_classification — each protein has exactly one correct label
                                e.g. subcellular localization: nucleus OR cytoplasm
                                Loss: CrossEntropyLoss
                                Metric: Accuracy

freeze_encoder_layers: freeze the first N transformer layers so their weights
  are not updated during fine-tuning.  Useful when labeled data is scarce —
  prevents overfitting by keeping the pretrained representations stable.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics.classification import MultilabelAUROC, MulticlassAccuracy

from ..models.heads import SequenceClassificationHead


class FineTuner(pl.LightningModule):
    """Fine-tunes a pretrained protein encoder on a downstream classification task."""

    def __init__(
        self,
        model: object,
        cfg: DictConfig,
        **kwargs,  # absorb extra Hydra config keys (trainer, checkpoint, etc.)
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg

        # Classification head: [CLS] embedding → label scores
        self.head = SequenceClassificationHead(
            hidden_size=model.config.hidden_size,
            num_labels=cfg.num_labels,
        )

        # Set up loss function and evaluation metric based on task type
        if cfg.problem_type == "multi_label_classification":
            self.criterion = nn.BCEWithLogitsLoss()
            self.auroc = MultilabelAUROC(num_labels=cfg.num_labels)
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.accuracy = MulticlassAccuracy(num_classes=cfg.num_labels)

        # Optionally freeze the first N transformer encoder layers.
        # PlantProteinBERT stores layers at model.bert.bert.encoder.layer
        if cfg.freeze_encoder_layers > 0:
            encoder_layers = model.bert.bert.encoder.layer
            for i, layer in enumerate(encoder_layers):
                if i < cfg.freeze_encoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.save_hyperparameters(ignore=["model"])

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract labels from batch, get embedding, compute logits."""
        labels = batch.pop("labels")
        # get_sequence_embedding returns [batch_size, hidden_size]
        embedding = self.model.get_sequence_embedding(**batch)
        # head maps to [batch_size, num_labels]
        logits = self.head(embedding)
        return logits, labels

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits, labels = self._forward(batch)
        loss = self.criterion(logits, labels.float())
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits, labels = self._forward(batch)
        loss = self.criterion(logits, labels.float())
        self.log("val/loss", loss, sync_dist=True)
        if self.cfg.problem_type == "multi_label_classification":
            self.auroc.update(torch.sigmoid(logits), labels.int())
            self.log("val/auroc", self.auroc, prog_bar=True)
        else:
            self.accuracy.update(logits, labels)
            self.log("val/accuracy", self.accuracy, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        logits, labels = self._forward(batch)
        loss = self.criterion(logits, labels.float())
        self.log("test/loss", loss, sync_dist=True)
        if self.cfg.problem_type == "multi_label_classification":
            self.auroc.update(torch.sigmoid(logits), labels.int())
            self.log("test/auroc", self.auroc)
        else:
            self.accuracy.update(logits, labels)
            self.log("test/accuracy", self.accuracy)

    def configure_optimizers(self):
        """Optimize both encoder and head parameters jointly."""
        optimizer = instantiate(
            self.cfg.optimizer,
            params=list(self.model.parameters()) + list(self.head.parameters()),
        )
        if getattr(self.cfg, "scheduler", None) is not None:
            scheduler = instantiate(self.cfg.scheduler, optimizer=optimizer)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
        return optimizer
