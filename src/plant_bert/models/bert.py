"""PlantProteinBERT: wraps HuggingFace BertForMaskedLM with protein-specific configuration.

Uses transformers.BertConfig constructed from Hydra model config — no reimplementation.
Exposes get_sequence_embedding() for [CLS]-token representations used in fine-tuning.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import BertConfig, BertForMaskedLM, BertModel


class PlantProteinBERT(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        max_position_embeddings: int,
        vocab_size: int,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        type_vocab_size: int = 1,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pretrained_checkpoint: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        bert_config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act="gelu",
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
        )

        if pretrained_checkpoint:
            self.bert = BertForMaskedLM.from_pretrained(pretrained_checkpoint, config=bert_config)
        else:
            self.bert = BertForMaskedLM(bert_config)

        self.config = bert_config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> object:
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def get_sequence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return [CLS] token hidden state as a fixed-size sequence representation."""
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str) -> "PlantProteinBERT":
        return torch.load(checkpoint_path, weights_only=False)
