"""
BERT-based Masked Language Model (MLM) — PyTorch Implementation
================================================================

Reference paper:
  Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019).
  "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."
  Proceedings of NAACL-HLT 2019. arXiv:1810.04805
  https://arxiv.org/abs/1810.04805

This script implements:
  1. A minimal BERT encoder from scratch (multi-head attention, feed-forward, embeddings)
  2. The MLM pre-training head
  3. A training loop with the 15% masking strategy described in the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# =============================================================================
# 1. Configuration
# =============================================================================

class BertConfig:
    """Mirrors BERT-base hyperparameters from Table 1 of the paper."""

    def __init__(
        self,
        vocab_size: int = 30_522,       # WordPiece vocabulary
        hidden_size: int = 768,          # H in the paper
        num_hidden_layers: int = 12,     # L in the paper
        num_attention_heads: int = 12,   # A in the paper
        intermediate_size: int = 3072,   # feed-forward inner dim (4H)
        max_position_embeddings: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id


# =============================================================================
# 2. Embeddings  (Token + Position + Segment → LayerNorm + Dropout)
# =============================================================================

class BertEmbeddings(nn.Module):
    """
    Construct embeddings from token, position, and segment (token_type) ids.
    BERT sums all three embeddings before applying LayerNorm.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)  # sentence A / B

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Position ids are a contiguous range — register as buffer (not a parameter)
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).unsqueeze(0),  # (1, max_len)
        )

    def forward(
        self,
        input_ids: torch.Tensor,               # (batch, seq_len)
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        word_emb = self.word_embeddings(input_ids)
        pos_emb = self.position_embeddings(self.position_ids[:, :seq_len])
        type_emb = self.token_type_embeddings(token_type_ids)

        embeddings = word_emb + pos_emb + type_emb
        return self.dropout(self.layer_norm(embeddings))


# =============================================================================
# 3. Multi-Head Self-Attention
# =============================================================================

class BertSelfAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,            # (batch, seq_len, hidden)
        attention_mask: Optional[torch.Tensor] = None,  # (batch, 1, 1, seq_len)
    ) -> torch.Tensor:
        B, S, _ = hidden_states.shape

        # Project and reshape: (B, S, H) → (B, num_heads, S, head_dim)
        q = self.query(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            scores = scores + attention_mask  # mask is additive (−inf for padded positions)

        attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn_weights, v)  # (B, num_heads, S, head_dim)

        # Concatenate heads: (B, S, hidden_size)
        context = context.transpose(1, 2).contiguous().view(B, S, -1)
        return context


# =============================================================================
# 4. Transformer Block  (Attention → Add&Norm → FFN → Add&Norm)
# =============================================================================

class BertLayer(nn.Module):
    """One Transformer encoder layer."""

    def __init__(self, config: BertConfig):
        super().__init__()
        # Self-attention sub-layer
        self.attention = BertSelfAttention(config)
        self.attn_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attn_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Feed-forward sub-layer  (H → 4H → H with GELU activation)
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None):
        # Sub-layer 1: self-attention + residual
        attn_output = self.attention(hidden_states, attention_mask)
        attn_output = self.attn_dropout(self.attn_proj(attn_output))
        hidden_states = self.attn_norm(hidden_states + attn_output)

        # Sub-layer 2: feed-forward + residual
        ffn_output = self.ffn_dropout(self.ffn(hidden_states))
        hidden_states = self.ffn_norm(hidden_states + ffn_output)
        return hidden_states


# =============================================================================
# 5. Full BERT Encoder
# =============================================================================

class BertEncoder(nn.Module):
    """Stack of L Transformer layers."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class BertModel(nn.Module):
    """BERT encoder: embeddings → transformer stack → contextual representations."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        # Build additive attention mask: 0 for real tokens, −10 000 for pads
        if attention_mask is not None:
            # (batch, seq_len) → (batch, 1, 1, seq_len) for broadcasting across heads
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -1e4

        hidden_states = self.embeddings(input_ids, token_type_ids)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states  # (batch, seq_len, hidden_size)


# =============================================================================
# 6. MLM Head  (hidden → vocab logits)
# =============================================================================

class BertMLMHead(nn.Module):
    """
    Predict the original token for each masked position.
    Uses a weight-tied projection (shares weights with the word embedding table).
    """

    def __init__(self, config: BertConfig, word_embeddings_weight: torch.Tensor):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = nn.GELU()

        # Weight tying: output projection reuses the embedding matrix
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        self.decoder.weight = word_embeddings_weight  # tied

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.dense(hidden_states))
        x = self.layer_norm(x)
        logits = self.decoder(x)  # (batch, seq_len, vocab_size)
        return logits


# =============================================================================
# 7. BertForMaskedLM — Full Model
# =============================================================================

class BertForMaskedLM(nn.Module):
    """BERT encoder + MLM head. Loss computed only on masked positions."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.bert = BertModel(config)
        self.mlm_head = BertMLMHead(config, self.bert.embeddings.word_embeddings.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
            input_ids:      (batch, seq_len) — token ids with [MASK] inserted
            labels:         (batch, seq_len) — original token ids; -100 at non-masked positions
        Returns:
            loss (if labels provided), logits
        """
        hidden_states = self.bert(input_ids, token_type_ids, attention_mask)
        logits = self.mlm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,  # only backprop through masked positions
            )
        return loss, logits


# =============================================================================
# 8. Masking Strategy  (Section 3.3.1 of the paper)
# =============================================================================

def mask_tokens(
    input_ids: torch.Tensor,
    vocab_size: int,
    mask_token_id: int = 103,   # [MASK] in standard BERT tokenizer
    mlm_probability: float = 0.15,
    special_token_ids: set = frozenset({0, 101, 102}),  # [PAD], [CLS], [SEP]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the BERT masking strategy:
      - 15% of tokens are selected for prediction
      - Of those: 80% → [MASK], 10% → random token, 10% → unchanged

    Returns:
        masked_input_ids: input with masks applied
        labels:           original ids at masked positions, -100 elsewhere
    """
    labels = input_ids.clone()
    masked_input = input_ids.clone()

    # Probability matrix — don't mask special tokens
    probability_matrix = torch.full(input_ids.shape, mlm_probability)
    for sid in special_token_ids:
        probability_matrix.masked_fill_(input_ids == sid, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # only compute loss on masked tokens

    # 80% of the time → replace with [MASK]
    replace_mask = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    masked_input[replace_mask] = mask_token_id

    # 10% of the time → replace with random token
    random_replace = (
        torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
        & masked_indices
        & ~replace_mask
    )
    random_tokens = torch.randint(low=0, high=vocab_size, size=input_ids.shape)
    masked_input[random_replace] = random_tokens[random_replace]

    # Remaining 10% → keep original (already in masked_input)
    return masked_input, labels


# =============================================================================
# 9. Training Loop
# =============================================================================

def train_mlm():
    """Minimal training loop for demonstration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BertConfig()
    model = BertForMaskedLM(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Device: {device}\n")

    # ---------- Dummy data (replace with real DataLoader) ----------
    batch_size, seq_len = 4, 128
    num_steps = 50

    for step in range(1, num_steps + 1):
        # Simulate tokenized input (skip special tokens for simplicity)
        input_ids = torch.randint(1000, config.vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)

        # Apply BERT masking
        masked_input, labels = mask_tokens(input_ids, config.vocab_size)
        masked_input, labels = masked_input.to(device), labels.to(device)

        # Forward + backward
        loss, logits = model(masked_input, attention_mask=attention_mask, labels=labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            # Accuracy on masked positions only
            masked_positions = labels != -100
            preds = logits[masked_positions].argmax(dim=-1)
            acc = (preds == labels[masked_positions]).float().mean().item()
            print(f"Step {step:>3d} | Loss: {loss.item():.4f} | MLM Acc: {acc:.2%}")


# =============================================================================
# 10. Inference: Fill in the [MASK]
# =============================================================================

@torch.no_grad()
def predict_masked_token(model, input_ids, mask_positions, top_k=5):
    """
    Given input with [MASK] tokens, predict the top-k candidates
    for each masked position.
    """
    model.eval()
    _, logits = model(input_ids)

    results = {}
    for pos in mask_positions:
        probs = F.softmax(logits[0, pos], dim=-1)
        topk_probs, topk_ids = probs.topk(top_k)
        results[pos] = list(zip(topk_ids.tolist(), topk_probs.tolist()))
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    train_mlm()
