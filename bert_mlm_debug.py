"""
BERT MLM — Fully Instrumented Data Flow Tracer
================================================

This script builds a small BERT model and passes a real sentence through it,
printing the shape, dtype, and sample values at EVERY stage so you can see
exactly what happens inside the transformer.

Reference:
  Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding." NAACL-HLT. arXiv:1810.04805

We use a tiny config (2 layers, 4 heads, hidden=64) so the output is readable,
but the architecture is identical to BERT-base — just scaled down.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from textwrap import indent

torch.manual_seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# Pretty printing helpers
# ═══════════════════════════════════════════════════════════════════════════════

CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
MAGENTA = "\033[95m"
BOLD    = "\033[1m"
RESET   = "\033[0m"
DIM     = "\033[2m"

def section(title):
    width = 72
    print(f"\n{BOLD}{CYAN}{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}{RESET}\n")

def subsection(title):
    print(f"\n  {BOLD}{GREEN}── {title} ──{RESET}\n")

def trace(name, tensor, show_values=True, indent_level=4):
    """Print shape, dtype, stats, and optionally first few values of a tensor."""
    prefix = " " * indent_level
    shape_str = "×".join(str(d) for d in tensor.shape)
    print(f"{prefix}{YELLOW}{name}{RESET}  shape=({shape_str})  dtype={tensor.dtype}")
    
    if tensor.numel() > 0 and tensor.is_floating_point():
        print(f"{prefix}  {DIM}min={tensor.min().item():.4f}  max={tensor.max().item():.4f}  "
              f"mean={tensor.mean().item():.4f}  std={tensor.std().item():.4f}{RESET}")
    
    if show_values and tensor.numel() <= 200:
        # Show full tensor for small tensors
        val_str = str(tensor.data)
        for line in val_str.split('\n')[:6]:
            print(f"{prefix}  {DIM}{line}{RESET}")
    elif show_values:
        # Show first row / slice
        flat = tensor.reshape(-1)[:10]
        print(f"{prefix}  {DIM}first 10 values: [{', '.join(f'{v:.4f}' for v in flat.tolist())}]{RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Tiny vocabulary and tokenizer (hand-built for clarity)
# ═══════════════════════════════════════════════════════════════════════════════

# A minimal vocab so you can see real words flow through, not random integers
VOCAB = {
    "[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[MASK]": 3, "[UNK]": 4,
    "the": 5, "cat": 6, "sat": 7, "on": 8, "mat": 9,
    "a": 10, "dog": 11, "ran": 12, "in": 13, "park": 14,
    "bird": 15, "flew": 16, "over": 17, "tree": 18, "big": 19,
    "small": 20, "happy": 21, "is": 22, "was": 23, "and": 24,
}
ID2TOKEN = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)


def tokenize(sentence: str) -> list[int]:
    """Minimal whitespace tokenizer using our tiny vocab."""
    tokens = [VOCAB.get(w.lower(), VOCAB["[UNK]"]) for w in sentence.split()]
    return [VOCAB["[CLS]"]] + tokens + [VOCAB["[SEP]"]]


def ids_to_tokens(ids: list[int]) -> list[str]:
    return [ID2TOKEN.get(i, f"[{i}]") for i in ids]


# ═══════════════════════════════════════════════════════════════════════════════
# Model config — intentionally tiny so prints are readable
# ═══════════════════════════════════════════════════════════════════════════════

class TinyBertConfig:
    vocab_size = VOCAB_SIZE          # 25 tokens
    hidden_size = 64                 # H (BERT-base uses 768)
    num_hidden_layers = 2            # L (BERT-base uses 12)
    num_attention_heads = 4          # A (BERT-base uses 12)
    intermediate_size = 256          # 4×H feed-forward
    max_position_embeddings = 32
    hidden_dropout_prob = 0.0        # no dropout for deterministic tracing
    attention_probs_dropout_prob = 0.0
    layer_norm_eps = 1e-12
    pad_token_id = 0


# ═══════════════════════════════════════════════════════════════════════════════
# Instrumented modules — identical to BERT, but with print statements
# ═══════════════════════════════════════════════════════════════════════════════

class TracedEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.register_buffer("position_ids",
                             torch.arange(config.max_position_embeddings).unsqueeze(0))

    def forward(self, input_ids, token_type_ids=None):
        B, S = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        subsection("STEP 1: Token Embedding Lookup")
        print(f"    Each token ID → a {self.word_embeddings.embedding_dim}-dim vector")
        print(f"    Embedding table shape: ({self.word_embeddings.num_embeddings}, "
              f"{self.word_embeddings.embedding_dim})\n")
        word_emb = self.word_embeddings(input_ids)
        trace("word_embeddings", word_emb)

        subsection("STEP 2: Position Embedding Lookup")
        print(f"    Position IDs: {list(range(S))}")
        print(f"    Each position → a {self.position_embeddings.embedding_dim}-dim vector\n")
        pos_ids = self.position_ids[:, :S]
        pos_emb = self.position_embeddings(pos_ids)
        trace("position_embeddings", pos_emb)

        subsection("STEP 3: Token Type (Segment) Embedding")
        print(f"    Token type IDs: {token_type_ids[0].tolist()}")
        print(f"    (0 = sentence A, 1 = sentence B)\n")
        type_emb = self.token_type_embeddings(token_type_ids)
        trace("token_type_embeddings", type_emb)

        subsection("STEP 4: Sum All Three Embeddings")
        combined = word_emb + pos_emb + type_emb
        print(f"    word_emb + pos_emb + type_emb = combined")
        trace("summed_embeddings", combined)

        subsection("STEP 5: LayerNorm (normalize each token vector)")
        output = self.layer_norm(combined)
        print(f"    LayerNorm centers to mean≈0, scales to unit variance")
        trace("after_layer_norm", output)

        return output


class TracedSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        B, S, H = hidden_states.shape

        subsection(f"Layer {self.layer_idx} — SELF-ATTENTION")

        # ── Q, K, V projections ──
        print(f"    Project hidden → Q, K, V  via three Linear({H}, {H}) layers\n")
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        trace("Q (query)", q)
        trace("K (key)", k)
        trace("V (value)", v)

        # ── Reshape into heads ──
        print(f"    {MAGENTA}Reshape into {self.num_heads} heads × {self.head_dim} dims each{RESET}\n")
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        trace("Q reshaped (batch, heads, seq, head_dim)", q)

        # ── Attention scores ──
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        print(f"    {MAGENTA}Attention scores = (Q @ K^T) / √{self.head_dim} = (Q @ K^T) / {scale:.2f}{RESET}\n")
        trace("raw_attention_scores (batch, heads, seq, seq)", scores)

        # ── Apply mask (if any) ──
        if attention_mask is not None:
            print(f"    Apply attention mask (−10000 on padded positions)\n")
            scores = scores + attention_mask

        # ── Softmax → attention weights ──
        attn_weights = F.softmax(scores, dim=-1)
        print(f"    {MAGENTA}Softmax → each token's attention distribution over all tokens{RESET}")
        print(f"    (Each row sums to 1.0 — it's a probability distribution)\n")
        trace("attention_weights (batch, heads, seq, seq)", attn_weights)

        # Print human-readable attention for head 0
        print(f"    {BOLD}Head 0 attention matrix (who attends to whom):{RESET}")
        w = attn_weights[0, 0]  # first batch, first head
        for i in range(S):
            row = " ".join(f"{w[i, j].item():.2f}" for j in range(S))
            print(f"      {ID2TOKEN.get(0, '?'):>8s}  →  [{row}]")
        print()

        # ── Weighted sum of values ──
        context = torch.matmul(attn_weights, v)
        print(f"    {MAGENTA}Context = attn_weights @ V  (weighted sum of value vectors){RESET}\n")
        trace("context_per_head", context)

        # ── Concatenate heads ──
        context = context.transpose(1, 2).contiguous().view(B, S, -1)
        print(f"    {MAGENTA}Concatenate all heads → back to (batch, seq, {H}){RESET}\n")
        trace("context_concatenated", context)

        return context


class TracedTransformerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention = TracedSelfAttention(config, layer_idx)
        self.attn_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.ffn_up = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_down = nn.Linear(config.intermediate_size, config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        H = hidden_states.shape[-1]

        # ── Self-attention sub-layer ──
        attn_out = self.attention(hidden_states, attention_mask)

        subsection(f"Layer {self.layer_idx} — OUTPUT PROJECTION + RESIDUAL + LAYERNORM")
        projected = self.attn_proj(attn_out)
        print(f"    Linear projection: ({H}) → ({H})")
        trace("attention_projected", projected)

        residual = hidden_states + projected
        print(f"    {MAGENTA}Residual connection: input + attention_output{RESET}")
        trace("after_residual_add", residual)

        normed = self.attn_norm(residual)
        print(f"    LayerNorm → stabilizes training")
        trace("after_attn_layernorm", normed)

        # ── Feed-forward sub-layer ──
        subsection(f"Layer {self.layer_idx} — FEED-FORWARD NETWORK (FFN)")
        print(f"    FFN: Linear({H}→{self.ffn_up.out_features}) → GELU → Linear({self.ffn_up.out_features}→{H})\n")

        ffn_hidden = self.ffn_up(normed)
        trace("ffn_up_projection", ffn_hidden)

        ffn_activated = F.gelu(ffn_hidden)
        print(f"    {MAGENTA}GELU activation (smooth ReLU variant){RESET}")
        trace("after_gelu", ffn_activated)

        ffn_output = self.ffn_down(ffn_activated)
        trace("ffn_down_projection", ffn_output)

        residual2 = normed + ffn_output
        print(f"    {MAGENTA}Residual connection: layernorm_out + ffn_output{RESET}")
        trace("after_ffn_residual", residual2)

        output = self.ffn_norm(residual2)
        print(f"    Final LayerNorm for this block")
        trace("layer_output", output)

        return output


class TracedBertMLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = TracedEmbeddings(config)
        self.layers = nn.ModuleList(
            [TracedTransformerBlock(config, i) for i in range(config.num_hidden_layers)]
        )
        # MLM head
        self.mlm_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.mlm_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlm_decoder = nn.Linear(config.hidden_size, config.vocab_size)
        # Weight tying
        self.mlm_decoder.weight = self.embeddings.word_embeddings.weight

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        # ── Embeddings ──
        hidden = self.embeddings(input_ids, token_type_ids)

        # ── Attention mask ──
        if attention_mask is not None:
            attention_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -1e4

        # ── Transformer layers ──
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)

        # ── MLM Head ──
        section("MLM PREDICTION HEAD")

        subsection("Dense + GELU + LayerNorm")
        mlm_hidden = F.gelu(self.mlm_dense(hidden))
        trace("mlm_dense_output", mlm_hidden)

        mlm_normed = self.mlm_norm(mlm_hidden)
        trace("mlm_layernorm_output", mlm_normed)

        subsection("Project to vocabulary (weight-tied with embedding table)")
        logits = self.mlm_decoder(mlm_normed)
        print(f"    Shape: (batch, seq_len, vocab_size) = {tuple(logits.shape)}")
        print(f"    Each position gets a score for every token in the vocabulary\n")
        trace("logits", logits, show_values=False)

        # ── Loss ──
        loss = None
        if labels is not None:
            subsection("Cross-Entropy Loss (only on [MASK] positions)")
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            print(f"    {BOLD}Loss = {loss.item():.4f}{RESET}")
            print(f"    (Positions with label=-100 are ignored)\n")

        return loss, logits


# ═══════════════════════════════════════════════════════════════════════════════
# Main: Run the full trace
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    config = TinyBertConfig()
    model = TracedBertMLM(config)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())

    # ── Input ──
    section("INPUT PREPARATION")

    sentence = "the cat sat on the mat"
    print(f"    Original sentence:  \"{sentence}\"")

    token_ids = tokenize(sentence)
    tokens = ids_to_tokens(token_ids)
    print(f"    Tokenized:          {tokens}")
    print(f"    Token IDs:          {token_ids}\n")

    # Apply masking: replace "sat" (position 3) with [MASK]
    masked_ids = token_ids.copy()
    mask_pos = 3  # "sat" is at index 3 (after [CLS])
    original_token = masked_ids[mask_pos]
    masked_ids[mask_pos] = VOCAB["[MASK]"]

    print(f"    {RED}Masking position {mask_pos}: '{ID2TOKEN[original_token]}' → '[MASK]'{RESET}")
    print(f"    Masked IDs:         {masked_ids}")
    print(f"    Masked tokens:      {ids_to_tokens(masked_ids)}\n")

    # Build tensors
    input_ids = torch.tensor([masked_ids])       # (1, seq_len)
    attention_mask = torch.ones_like(input_ids)   # all tokens are real (no padding)
    token_type_ids = torch.zeros_like(input_ids)  # single sentence

    # Labels: -100 everywhere except the masked position
    labels = torch.full_like(input_ids, -100)
    labels[0, mask_pos] = original_token

    trace("input_ids", input_ids)
    trace("attention_mask", attention_mask)
    trace("labels", labels)

    print(f"    {DIM}Model parameters: {param_count:,} ({param_count/1e3:.1f}K){RESET}")
    print(f"    {DIM}Config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, ffn={config.intermediate_size}{RESET}\n")

    # ── Forward pass ──
    section("FORWARD PASS THROUGH EMBEDDINGS")
    section("FORWARD PASS THROUGH TRANSFORMER LAYERS")

    with torch.no_grad():
        loss, logits = model(input_ids, token_type_ids, attention_mask, labels)

    # ── Analyze predictions ──
    section("PREDICTION ANALYSIS")

    subsection(f"What does the model predict at the [MASK] position (index {mask_pos})?")

    mask_logits = logits[0, mask_pos]  # (vocab_size,)
    mask_probs = F.softmax(mask_logits, dim=-1)

    print(f"    Raw logits for position {mask_pos}:")
    for token, idx in sorted(VOCAB.items(), key=lambda x: x[1]):
        logit_val = mask_logits[idx].item()
        prob_val = mask_probs[idx].item()
        marker = f"  {RED}← TRUE ANSWER{RESET}" if idx == original_token else ""
        bar = "█" * int(prob_val * 50)
        print(f"      {token:>8s} (id={idx:2d})  logit={logit_val:+7.3f}  "
              f"prob={prob_val:.4f}  {GREEN}{bar}{RESET}{marker}")

    top5 = mask_probs.topk(5)
    print(f"\n    {BOLD}Top 5 predictions:{RESET}")
    for i in range(5):
        tid = top5.indices[i].item()
        prob = top5.values[i].item()
        correct = " ✓" if tid == original_token else ""
        print(f"      #{i+1}  '{ID2TOKEN[tid]}'  prob={prob:.4f}{correct}")

    print(f"\n    {BOLD}Ground truth: '{ID2TOKEN[original_token]}' (id={original_token}){RESET}")
    true_rank = (mask_probs > mask_probs[original_token]).sum().item() + 1
    print(f"    True token rank: {true_rank}/{VOCAB_SIZE}")
    print(f"    True token probability: {mask_probs[original_token].item():.4f}")
    print(f"\n    Note: Model has random weights — after training, the correct token")
    print(f"    would rank much higher!\n")

    # ── Gradient flow demo ──
    section("BONUS: GRADIENT FLOW (one backward pass)")

    model.train()
    input_ids_g = torch.tensor([masked_ids])
    attn_mask_g = torch.ones_like(input_ids_g)
    type_ids_g = torch.zeros_like(input_ids_g)
    labels_g = torch.full_like(input_ids_g, -100)
    labels_g[0, mask_pos] = original_token

    # Suppress prints for the gradient pass
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    loss_g, _ = model(input_ids_g, type_ids_g, attn_mask_g, labels_g)
    sys.stdout = old_stdout

    loss_g.backward()

    print(f"    Loss: {loss_g.item():.4f}\n")
    print(f"    {BOLD}Gradient norms by component:{RESET}\n")
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            bar = "█" * min(int(grad_norm * 10), 40)
            print(f"      {name:45s}  grad_norm={grad_norm:.6f}  {GREEN}{bar}{RESET}")

    print(f"\n    {DIM}Gradients flow from the loss at the [MASK] position back through")
    print(f"    the entire model. Larger gradient norms = more update in that layer.{RESET}\n")


if __name__ == "__main__":
    main()
