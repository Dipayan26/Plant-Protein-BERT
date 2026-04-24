"""ESM-2 wrapper for domain-adaptive continued pretraining on plant proteins.

Loads a pretrained ESM-2 checkpoint from HuggingFace Hub and exposes the same
forward / get_sequence_embedding interface as PlantProteinBERT, so any future
fine-tuning head (FineTuner in training/finetune.py) can accept either model
without modification.

Supported hub_names:
    facebook/esm2_t6_8M_UR50D      (8M,   6  layers)
    facebook/esm2_t12_35M_UR50D    (35M,  12 layers)
    facebook/esm2_t30_150M_UR50D   (150M, 30 layers)
    facebook/esm2_t33_650M_UR50D   (650M, 33 layers) — inference / LoRA only on 12 GB
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import EsmForMaskedLM


class PlantESM2(nn.Module):
    """ESM-2 loaded from HuggingFace Hub, configured for domain-adaptive pretraining."""

    def __init__(self, hub_name: str, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.esm = EsmForMaskedLM.from_pretrained(hub_name)
        if gradient_checkpointing:
            self.esm.gradient_checkpointing_enable()
        # expose config so downstream fine-tuning heads can read hidden_size
        self.config = self.esm.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> object:
        """Standard MLM forward pass.  Returns ModelOutput with .loss and .logits."""
        return self.esm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def get_sequence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the <cls> token hidden state as a sequence-level embedding.

        Shape: [batch_size, hidden_size]
        Drop-in replacement for PlantProteinBERT.get_sequence_embedding().
        """
        out = self.esm.esm(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]

    def save_hf_checkpoint(self, path: str) -> None:
        """Save in HuggingFace format for from_pretrained() compatibility.

        Use this after training to export the adapted model independently of the
        Lightning .ckpt file (which also stores optimizer state, scheduler, etc.).
        The saved directory can be loaded with:
            EsmForMaskedLM.from_pretrained(path)
            EsmTokenizer.from_pretrained(original_hub_name)  # tokenizer is unchanged
        """
        self.esm.save_pretrained(path)
