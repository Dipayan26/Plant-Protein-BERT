"""Task-specific heads for fine-tuning on downstream protein tasks."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLMHead(nn.Module):
    """Masked language modelling head (already built into BertForMaskedLM; provided for reference)."""

    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.gelu(self.dense(hidden_states))
        x = self.layer_norm(x)
        return self.decoder(x)


class SequenceClassificationHead(nn.Module):
    """Pooled [CLS] → linear classifier for sequence-level tasks.

    Works for both single-label (CrossEntropy) and multi-label (BCEWithLogitsLoss) tasks.
    problem_type is used externally to select the loss function.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(cls_embedding))
