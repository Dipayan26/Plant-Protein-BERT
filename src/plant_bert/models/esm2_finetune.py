"""ESM2FineTuner: drop-in replacement for PlantProteinBERT for comparison experiments.

Loads a pretrained ESM-2 model from HuggingFace and exposes the same interface
as PlantProteinBERT so it can be used with the existing FineTuner Lightning module:
  - get_sequence_embedding(input_ids, attention_mask) → [batch, hidden_size] tensor
  - self.config.hidden_size   (read by FineTuner to build classification head)

Sequence embedding strategy: mean-pool over all non-padding token positions.
ESM-2 does not use a [CLS] token in the same way as BERT; mean-pooling is the
standard approach used in the ESM-2 literature for sequence-level tasks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import EsmModel


class _ConfigProxy:
    """Minimal config object so FineTuner can read hidden_size."""

    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size


class ESM2FineTuner(nn.Module):
    """Wraps a pretrained ESM-2 encoder for plant protein fine-tuning.

    Compatible with plant_bert.training.finetune.FineTuner.
    Input tokens must be encoded with ESM2TokenizerWrapper (tokenizer: esm2).
    """

    def __init__(
        self,
        esm2_model_name: str = "facebook/esm2_t6_8M_UR50D",
        freeze_base_model: bool = False,
    ) -> None:
        super().__init__()
        self.esm = EsmModel.from_pretrained(esm2_model_name)
        self.config = _ConfigProxy(hidden_size=self.esm.config.hidden_size)

        if freeze_base_model:
            for param in self.esm.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> object:
        return self.esm(input_ids=input_ids, attention_mask=attention_mask)

    def get_sequence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean-pool over non-padding positions → [batch_size, hidden_size]."""
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            return summed / lengths
        return hidden.mean(dim=1)
