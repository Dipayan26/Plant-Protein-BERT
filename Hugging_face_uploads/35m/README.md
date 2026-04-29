---
language:
  - en
license: mit
tags:
  - biology
  - protein
  - esm2
  - plant
  - viridiplantae
  - masked-language-modeling
  - domain-adaptation
base_model: facebook/esm2_t12_35M_UR50D
datasets:
  - uniprot-trembl-viridiplantae
pipeline_tag: fill-mask
---

# PlantPLM-35M

<img src="Plant_PLM_logo.png" alt="Alt Text" width="800">



**ESM-2 35M parameter model continued-pretrained on Viridiplantae (plant) protein sequences.**

This is a domain-adapted version of [`facebook/esm2_t12_35M_UR50D`](https://huggingface.co/facebook/esm2_t12_35M_UR50D), fine-tuned on a curated subset of UniProt TrEMBL containing only plant-kingdom proteins. The adaptation improves representation quality for plant-specific protein tasks compared to the general-purpose ESM-2 baseline.

Part of the **[Plant-PLM](https://huggingface.co/collections/dipayan26/plant-plm)** - ESM-2 models at 8M, 35M, 150M, and 650M parameters, each adapted on the same plant protein corpus.

---

## Model Description

| Property | Value |
|---|---|
| Base model | `facebook/esm2_t12_35M_UR50D` |
| Architecture | ESM-2 · 12 layers · hidden=480 · heads=20 · FFN=1920 |
| Position embeddings | Rotary (RoPE) |
| Vocabulary | 33 tokens (20 standard + rare amino acids + special tokens) |
| Parameters | 33.5M (full-parameter continued pretraining) |
| Training objective | Masked Language Modeling (MLM, 15% masking) |

---

## Training Data

| Property | Value |
|---|---|
| Source | UniProt TrEMBL — Viridiplantae (plant kingdom) subset |
| Sequences | **19,938,415** protein sequences |
| Avg sequence length | 339 AA · median 291 AA |
| Estimated total tokens | **~6.76 billion** amino acid tokens |
| Tokens seen during training | **546 million** (≈ 0.08 passes over the full dataset) |

---

## Training Details

| Hyperparameter | Value |
|---|---|
| Training steps | 55,000 optimizer steps |
| Batch size | 64 sequences (32 per micro-batch × 2 gradient accumulation steps) |
| Optimizer | AdamW · β=(0.9, 0.98) · ε=1e-8 · weight_decay=0.01 |
| Learning rate | 2e-5 (20× lower than ESM-2 from-scratch to prevent catastrophic forgetting) |
| LR schedule | Linear warmup (500 steps) → linear decay |
| Gradient clipping | 1.0 |
| Precision | 16-bit mixed (fp16 activations, fp32 optimizer states) |


**Final metrics (validation set, 5% holdout):**

| Metric | Value |
|---|---|
| `val/mlm_loss` | 2.075 |
| `val/perplexity` | 7.96 |

---

## Downstream Task Performance (Linear Probe)

Frozen [CLS] embeddings evaluated on 2,000 reviewed *Arabidopsis thaliana* proteins from UniProt SwissProt using a logistic regression linear probe. Compared against the vanilla `facebook/esm2_t12_35M_UR50D` baseline.

| Task | Vanilla ESM-2 35M | PlantPLM-35M | Δ |
|---|---|---|---|
| Subcellular localization (9-class accuracy) | 91.87% | **94.28%** | +2.41% |
| Subcellular localization (macro-F1) | 92.57% | **94.86%** | +2.29% |
| GO-term prediction (macro-AUROC, top-50 terms) | 94.26% | **94.82%** | +0.56% |

*Test set: 332 proteins (localization) · 396 proteins (GO terms) · 9 localization classes · 50 GO terms evaluated.*

---

## Usage

```python
from transformers import EsmForMaskedLM, EsmTokenizer
import torch

model = EsmForMaskedLM.from_pretrained("dipayan26/PlantPLM-35M")
tokenizer = EsmTokenizer.from_pretrained("dipayan26/PlantPLM-35M")

# --- Masked token prediction ---
sequence = "MSPQTETKASVGFKAGVKDYKLTYYTPEYETK"
inputs = tokenizer(sequence, return_tensors="pt")

# mask one position
inputs["input_ids"][0, 5] = tokenizer.mask_token_id

with torch.no_grad():
    logits = model(**inputs).logits

masked_pos = (inputs["input_ids"] == tokenizer.mask_token_id).nonzero()[0, 1]
top5 = logits[0, masked_pos].topk(5)
print(tokenizer.convert_ids_to_tokens(top5.indices.tolist()))

# --- Sequence embedding ([CLS] token) ---
inputs = tokenizer(sequence, return_tensors="pt")
with torch.no_grad():
    hidden = model.esm(**inputs).last_hidden_state
cls_embedding = hidden[0, 0, :]   # shape: [480]
print("Embedding shape:", cls_embedding.shape)
```

---

## Intended Use

- **Plant protein function prediction** — GO term annotation, subcellular localization, signal peptide detection
- **Plant-specific protein embeddings** — clustering, retrieval, similarity search
- **Transfer learning starting point** — fine-tune on small labeled plant protein datasets
- **Baseline comparison** — benchmark against PlantPLM-8M / 150M / 650M variants

## Out-of-scope Use

- Non-plant organisms — the model has been shifted toward Viridiplantae statistics; use the original `facebook/esm2_t12_35M_UR50D` for general protein tasks
- Structural prediction — not trained for structure; use ESMFold for that

---

## Limitations

- Trained for only 0.08 passes over the plant corpus (546M / 6.76B tokens) — larger models in this collection see more of the data
- For highest downstream accuracy, the 150M variant is recommended

---

## Citation

If you use this model, please cite:

```bibtex
@misc{sarkar2026plantplm,
  author       = {Sarkar, Dipayan},
  title        = {PlantPLM: Domain-Adaptive Pretraining of ESM-2 on Viridiplantae Proteins},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/dipayan26/PlantPLM-35M}},
}
```

---

<!-- ## Training Code

[github.com/Dipayan26/Plant-Protein-BERT](https://github.com/Dipayan26/Plant-Protein-BERT) -->
