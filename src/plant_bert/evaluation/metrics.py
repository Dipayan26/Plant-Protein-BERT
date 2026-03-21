"""Evaluation metrics for protein language model assessment."""

from __future__ import annotations

import math

import torch


def compute_perplexity(loss: float) -> float:
    """Convert average cross-entropy loss to perplexity."""
    return math.exp(min(loss, 100))


def compute_mlm_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Accuracy over masked positions only (labels == -100 are ignored)."""
    mask = labels != -100
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()
