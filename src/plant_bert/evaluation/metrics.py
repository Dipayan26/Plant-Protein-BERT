"""Evaluation metrics for the protein language model.

Perplexity:
  The standard metric for language models.  It measures how "surprised" the
  model is when it sees a sequence — a lower value means the model assigns
  higher probability to the actual tokens.
  Formula: perplexity = exp(average cross-entropy loss)
  Interpretation:
    perplexity = 1    → perfect model (knows exactly which token comes next)
    perplexity = 30   → as good as random guessing over 30 tokens (bad)
    perplexity < 10   → good for protein language models at convergence

MLM Accuracy:
  The fraction of masked tokens that the model predicted correctly.
  Positions that were NOT masked (label=-100) are excluded from the count.
"""

from __future__ import annotations

import math

import torch


def compute_perplexity(loss: float) -> float:
    """Perplexity = exp(cross-entropy loss).  Capped at exp(100) to prevent overflow."""
    return math.exp(min(loss, 100))


def compute_mlm_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of masked tokens predicted correctly.

    Args:
        logits: model output scores, shape [batch, seq_len, vocab_size]
        labels: true token IDs; -100 at non-masked positions (ignored by convention)

    Returns:
        Float between 0 and 1.
    """
    mask = labels != -100   # True only at positions that were masked
    if mask.sum() == 0:
        return 0.0
    preds = logits.argmax(dim=-1)   # pick the highest-scoring token at each position
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()
