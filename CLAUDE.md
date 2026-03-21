# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Plant-Protein-BERT trains BERT-based protein language models specifically on plant proteins. The core hypothesis: domain-specific PLMs outperform general-purpose ones on plant protein tasks because plant proteins are underrepresented and evolutionarily divergent in general training data.

**Stack:** PyTorch · PyTorch Lightning · HuggingFace Transformers · Hydra · WandB
**Author:** Dipayan Sarkar

---

## Setup

```bash
pip install -e ".[dev]"
```

---

## Running the Pipeline

All scripts are Hydra entry points. Config defaults are in `configs/config.yaml`.

```bash
# Step 1 — Parse raw DAT.gz → JSONL → HDF5 (run once)
python scripts/parse_uniprot.py data=trembl_full

# Step 2 — (Optional) Train BPE tokenizer; skip for amino_acid tokenizer
python scripts/train_tokenizer.py tokenizer=bpe data=trembl_full

# Step 3 — MLM pretraining
python scripts/pretrain.py +experiment=dev_run            # fast sanity check
python scripts/pretrain.py +experiment=pretrain_base       # full run

# Step 4 — Fine-tune on downstream task
python scripts/finetune.py +experiment=finetune_go_terms \
    training.pretrained_checkpoint=outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/pretrain/last.ckpt

# Step 5 — Evaluate
python scripts/evaluate.py training.pretrained_checkpoint=<path>
```

**Hydra CLI overrides** work on any parameter:
```bash
python scripts/pretrain.py model=bert_large training.trainer.devices=-1
python scripts/pretrain.py --multirun model=bert_small,bert_base training.optimizer.lr=1e-4,5e-5
```

---

## Running Tests

```bash
pytest                        # all tests
pytest tests/test_parser.py   # single file
pytest -k "test_parse"        # pattern match
```

Tests use synthetic in-memory data — no dependency on the 18 GB TrEMBL file.

---

## Code Architecture

### Config system (`configs/`)

Hydra composes sub-configs at runtime from `configs/config.yaml`. The key hierarchy:

| Config group | Options | Controls |
|---|---|---|
| `data` | `trembl_full` | Dataset path, batch size, streaming mode |
| `tokenizer` | `amino_acid`, `bpe` | Vocab type and size |
| `model` | `bert_small`, `bert_base`, `bert_large` | Architecture dimensions |
| `training` | `pretrain`, `finetune` | LR, masking, optimizer, trainer args |
| `experiment` | `dev_run`, `pretrain_base`, `finetune_go_terms` | Named compositions that override the above |

Every training run auto-creates `outputs/YYYY-MM-DD/HH-MM-SS/` with a `.hydra/` snapshot of the exact config used — full reproducibility.

### Package (`src/plant_bert/`)

Strict layering: `utils` ← `tokenizer` ← `data` ← `models` ← `training`/`evaluation`. No upward imports.

**`data/uniprot_parser.py`** — The most critical module. Streaming generator over UniProt DAT.gz files that never loads the full file into memory. Parses ID/AC/DE/OS/OC/DR(GO)/SQ fields into `UniProtRecord` dataclasses.

**`data/preprocessing.py`** — One-time pipeline: `parse_dat_gz` → filtered JSONL → HDF5 index. The HDF5 index (`sequences.h5`) enables O(1) random access to any sequence in the 18 GB TrEMBL file during training. Results cached in `outputs/processed/`; set `data.force_reprocess=true` to re-run.

**`data/dataset.py`** — `StreamingProteinDataset` reads from the HDF5 index with lazy file open per worker, enabling O(1) random access to the full 18 GB TrEMBL file without loading it into RAM.

**`models/bert.py`** — `PlantProteinBERT` wraps `transformers.BertForMaskedLM` (no reimplementation). Exposes `get_sequence_embedding()` which returns the `[CLS]` token hidden state for fine-tuning heads.

**`training/pretrain.py`** — `MLMPretrainer(LightningModule)`: uses `DataCollatorForLanguageModeling` for masking, logs `val/mlm_loss` and `val/perplexity`.

**`training/finetune.py`** — `FineTuner(LightningModule)`: attaches `SequenceClassificationHead` to frozen/unfrozen BERT encoder. Supports `multi_label_classification` (BCEWithLogitsLoss) and `single_label_classification` (CrossEntropyLoss).

### Data files (`DATA/`)

Raw data — never modified by code. Parsers read from here, write processed outputs to `outputs/processed/`.

- `DATA/PLANT_uniprot_new/uniprot_trembl_18_GB/uniprot_trembl_plants.dat.gz` — TrEMBL plant proteins (~18 GB compressed)

### Artifacts (`artifacts/`)

Deliberately saved, distributable outputs: trained tokenizer vocab files. Committed to git (small files). Large model checkpoints go to `outputs/` (gitignored) or HuggingFace Hub.
