"""Task-specific prediction heads used during fine-tuning.

After pretraining, the BERT encoder learns rich protein representations.
Fine-tuning adds a small "head" on top to solve a specific prediction task:
  - GO term prediction:          which biological processes does this protein perform?
  - Subcellular localization:    where in the cell is this protein found?
  - Thermostability prediction:  how heat-stable is this protein?

The head takes the [CLS] token embedding from the encoder and maps it to
a prediction score for each possible label.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLMHead(nn.Module):
    """The Masked Language Modeling prediction head.

    NOTE: This is already built into BertForMaskedLM and is NOT used separately.
    It is included here purely for reference so you can see what it looks like.

    Architecture:
      hidden_state  →  Dense(hidden_size)  →  GELU  →  LayerNorm  →  Linear(vocab_size)

    The output is a score (logit) for each vocabulary token at each sequence
    position.  Cross-entropy loss is then computed only at masked positions.
    """

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
    """Simple linear classifier on top of the [CLS] embedding.

    This head is attached to the pretrained encoder during fine-tuning.
    It maps the [CLS] hidden vector to prediction scores for each label.

    Works for both:
      multi_label_classification  — each sequence can have multiple labels
                                    (e.g. GO terms); loss = BCEWithLogitsLoss
      single_label_classification — one label per sequence (e.g. localization);
                                    loss = CrossEntropyLoss
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        """cls_embedding: [batch_size, hidden_size] → logits: [batch_size, num_labels]"""
        return self.classifier(self.dropout(cls_embedding))
