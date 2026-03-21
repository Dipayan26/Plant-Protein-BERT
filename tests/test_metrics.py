"""Tests for evaluation metric computations."""

from __future__ import annotations

import math

import torch
from plant_bert.evaluation.metrics import compute_mlm_accuracy, compute_perplexity


def test_perplexity_zero_loss():
    assert compute_perplexity(0.0) == 1.0


def test_perplexity_high_loss():
    # Loss capped at 100 to prevent overflow
    result = compute_perplexity(200.0)
    assert result == math.exp(100)


def test_mlm_accuracy_perfect():
    # All masked positions predicted correctly
    labels = torch.tensor([[-100, 2, -100, 5]])
    logits = torch.zeros(1, 4, 10)
    logits[0, 1, 2] = 100.0  # predict class 2 at position 1
    logits[0, 3, 5] = 100.0  # predict class 5 at position 3
    assert compute_mlm_accuracy(logits, labels) == 1.0


def test_mlm_accuracy_no_masked_tokens():
    labels = torch.full((1, 4), -100)
    logits = torch.zeros(1, 4, 10)
    assert compute_mlm_accuracy(logits, labels) == 0.0
