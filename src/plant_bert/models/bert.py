"""PlantProteinBERT: our BERT model trained on plant protein sequences.

What is BERT?
  BERT (Bidirectional Encoder Representations from Transformers) is a neural
  network that reads sequences using self-attention — every position can attend
  to every other position simultaneously (unlike LSTMs which read left-to-right).
  This bidirectional context gives it a richer understanding of each position.

What is Masked Language Modeling (MLM)?
  During pretraining we randomly mask 15% of amino acid tokens and ask the model
  to predict what they were.  For example:
      Input:  A C D [MASK] G H I K L
      Target: predict "E" at position 4 (the masked token)
  The model must understand the surrounding sequence context to make this prediction.
  After training on millions of such examples, the model learns a rich representation
  of protein sequence patterns — effectively learning "protein grammar".

Architecture (configurable via configs/model/plant_bert_*.yaml):
  - Embedding layer: converts token IDs to continuous vectors
  - N transformer encoder layers (stacked self-attention + feedforward blocks)
  - MLM head: linear layer that predicts token ID at masked positions

We do NOT reimplement BERT — we use HuggingFace's BertForMaskedLM directly,
just configuring it with protein-specific dimensions that match ESM-2 sizes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM


class PlantProteinBERT(nn.Module):
    """BERT model configured for plant protein MLM pretraining.

    After pretraining, call get_sequence_embedding() to get a fixed-size
    vector per protein — used as input to the fine-tuning classification head.
    """

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

        # BertConfig holds all architecture hyperparameters.
        # We set type_vocab_size=1 because protein sequences have no sentence-type segments.
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

        # BertForMaskedLM = BERT encoder + MLM prediction head.
        # The MLM head is a linear layer: hidden_size → vocab_size.
        # It outputs one score per vocabulary token at each sequence position.
        self.bert = BertForMaskedLM(bert_config)

        if pretrained_checkpoint:
            # PyTorch Lightning saves the full training state (optimizer, steps, etc.)
            # in a checkpoint.  We only need the model weights, stored under the
            # "model.bert.*" prefix (because MLMPretrainer stores self.model = PlantProteinBERT,
            # and the BertForMaskedLM lives at self.model.bert).
            ckpt = torch.load(pretrained_checkpoint, map_location="cpu", weights_only=False)
            state_dict = ckpt.get("state_dict", ckpt)
            prefix = "model.bert."
            bert_weights = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            self.bert.load_state_dict(bert_weights)

        self.config = bert_config   # expose config so FineTuner can read hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> object:
        """Run the model.

        When labels are provided (during training), BertForMaskedLM automatically
        computes cross-entropy loss at masked positions (where labels != -100).
        Returns a ModelOutput with .loss and .logits.
        """
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def get_sequence_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Extract the [CLS] token's hidden state as a sequence-level embedding.

        BERT prepends a special [CLS] token to every sequence.  After training,
        its hidden state aggregates global sequence information and works as a
        fixed-size vector representation for the whole protein.

        Shape: [batch_size, hidden_size]
        """
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state shape: [batch, seq_len, hidden_size]
        # Index 0 along seq_len selects the [CLS] token position.
        return outputs.last_hidden_state[:, 0, :]
