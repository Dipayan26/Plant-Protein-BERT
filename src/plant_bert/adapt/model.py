"""ESM-2 wrapper for domain-adaptive continued pretraining on plant proteins.

Loads a pretrained ESM-2 checkpoint from HuggingFace Hub and exposes the same
forward / get_sequence_embedding interface as PlantProteinBERT, so any future
fine-tuning head (FineTuner in training/finetune.py) can accept either model
without modification.

Supported hub_names:
    facebook/esm2_t6_8M_UR50D      (8M,   6  layers) — full-param, ~1 GB VRAM
    facebook/esm2_t12_35M_UR50D    (35M,  12 layers) — full-param, ~3 GB VRAM
    facebook/esm2_t30_150M_UR50D   (150M, 30 layers) — full-param, ~9 GB VRAM (batch=8)
    facebook/esm2_t33_650M_UR50D   (650M, 33 layers) — LoRA only on 12 GB GPU

LoRA (for 650M and above):
    Pass a `lora` dict in the config; keys map directly to peft.LoraConfig args.
    Only the adapter weights are trainable — base weights are frozen.
    Adapters are merged into base weights on save_hf_checkpoint() so the
    exported model loads identically to a full-parameter model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import EsmForMaskedLM


class PlantESM2(nn.Module):
    """ESM-2 loaded from HuggingFace Hub, configured for domain-adaptive pretraining."""

    def __init__(
        self,
        hub_name: str,
        gradient_checkpointing: bool = False,
        lora: dict | None = None,
    ) -> None:
        super().__init__()
        self.esm = EsmForMaskedLM.from_pretrained(hub_name)
        if gradient_checkpointing:
            self.esm.gradient_checkpointing_enable()

        self._lora = lora is not None
        if lora is not None:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(**lora)
            self.esm = get_peft_model(self.esm, lora_cfg)
            self.esm.print_trainable_parameters()

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
        Works for both full-parameter and LoRA-wrapped models.
        Drop-in replacement for PlantProteinBERT.get_sequence_embedding().
        """
        out = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # hidden_states is a tuple of (n_layers + 1) tensors; last entry is the
        # final transformer layer output
        return out.hidden_states[-1][:, 0, :]

    def save_hf_checkpoint(self, path: str) -> None:
        """Save in HuggingFace format for from_pretrained() compatibility.

        For LoRA models, merges adapters into the base weights before saving so
        the output is a standard EsmForMaskedLM — no PEFT dependency at inference.

        The saved directory can be loaded with:
            EsmForMaskedLM.from_pretrained(path)
            EsmTokenizer.from_pretrained(original_hub_name)  # tokenizer is unchanged
        """
        if self._lora:
            # merge_and_unload() folds LoRA A×B into the base weight matrices
            # and returns a plain EsmForMaskedLM with no PEFT wrapper
            self.esm.merge_and_unload().save_pretrained(path)
        else:
            self.esm.save_pretrained(path)
