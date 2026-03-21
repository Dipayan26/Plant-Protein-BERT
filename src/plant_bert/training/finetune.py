"""Fine-tuning Lightning Module for downstream protein tasks."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from torchmetrics.classification import MultilabelAUROC, MulticlassAccuracy

from ..models.heads import SequenceClassificationHead


class FineTuner(pl.LightningModule):
    def __init__(
        self,
        model: object,
        cfg: DictConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.head = SequenceClassificationHead(
            hidden_size=model.config.hidden_size,
            num_labels=cfg.num_labels,
        )

        if cfg.problem_type == "multi_label_classification":
            self.criterion = nn.BCEWithLogitsLoss()
            self.auroc = MultilabelAUROC(num_labels=cfg.num_labels)
        else:
            self.criterion = nn.CrossEntropyLoss()
            self.accuracy = MulticlassAccuracy(num_classes=cfg.num_labels)

        if cfg.freeze_encoder_layers > 0:
            for i, layer in enumerate(model.bert.bert.encoder.layer):
                if i < cfg.freeze_encoder_layers:
                    for param in layer.parameters():
                        param.requires_grad = False

        self.save_hyperparameters(ignore=["model"])

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        labels = batch.pop("labels")
        embedding = self.model.get_sequence_embedding(**batch)
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

    def configure_optimizers(self):
        optimizer = instantiate(
            self.cfg.optimizer,
            params=list(self.model.parameters()) + list(self.head.parameters()),
        )
        return optimizer
