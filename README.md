# Plant-Protein-BERT

Domain-specific BERT-based protein language models trained exclusively on **Viridiplantae** (green plant) sequences, benchmarked against ESM-2 general-purpose models.

**Author:** Dipayan Sarkar
**Stack:** PyTorch · PyTorch Lightning · HuggingFace Transformers · Hydra · WandB

---

## Research Hypothesis

ESM-2 is trained on UniRef50, where plant proteins (Viridiplantae) occupy only **4.64% of member sequences** despite representing ~9.9% of sequence clusters. Plant proteins cluster poorly — fewer close homologs — consistent with their evolutionary divergence from the bacteria-dominated training corpus.

| Superkingdom | UniRef50 Clusters | Members |
|---|---:|---:|
| Bacteria | 50.81% | 63.13% |
| Metazoa | 18.74% | 12.80% |
| Fungi | 10.41% | 3.89% |
| **Plants (Viridiplantae)** | **9.90%** | **4.64%** |

**Core claim:** BERT models pretrained exclusively on plant sequences will outperform ESM-2 on plant-specific downstream tasks because:
1. Plant proteins are underrepresented in ESM-2's training data (~22× enrichment in our dataset)
2. Plant-specific sequence patterns (chloroplast targeting, plant secondary metabolism, cell wall biosynthesis) are diluted in a bacteria-dominated corpus
3. UniProt's "plant" subset is **15.7% contaminated** with non-Viridiplantae organisms (oomycetes, dinoflagellates) — our pipeline removes these entirely

### Comparison Matrix

| Model | Params | Pretrain data | Fine-tune on plant task |
|---|---|---|---|
| ESM-2 zero-shot | 8M / 35M / 150M | UniRef50 (all) | No |
| ESM-2 plant fine-tune | 8M / 35M / 150M | UniRef50 (all) | Yes ← strong baseline |
| **PlantBERT (ours)** | **8M / 35M / 150M** | **Plant only (19.9M seqs)** | **Yes** |

If PlantBERT beats ESM-2 plant fine-tune at matched parameter count → domain-specific pretraining from scratch is superior to adapting a biased general model.

---

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. To recreate the exact conda environment:

```bash
conda env create -f environment.yml
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## Data

### Raw data
```
DATA/PLANT_uniprot_new/uniprot_trembl_18_GB/uniprot_trembl_plants.dat.gz  (~18 GB)
```
UniProt TrEMBL plant subset. Never modified by any script.

### Processed data (built by pipeline)
```
outputs/taxonomy/nodes.dmp                  NCBI taxdump for taxonomy filtering
outputs/processed/viridiplantae_clean.fasta Clean sequences (19.9M, FASTA)
outputs/processed/trembl_full/sequences.h5  HDF5 training index (O(1) access)
```

### Filtering statistics
| Stage | Sequences |
|---|---:|
| Raw TrEMBL plant file | 25,481,103 |
| Removed — non-Viridiplantae (oomycetes, dinoflagellates, etc.) | 3,742,897 |
| Removed — length filter (< 50 or > 1024 AA) | 1,787,004 |
| **Final training set** | **19,938,415** |

---

## Pipeline

### Step 0 — Clean data

```bash
# Extract Viridiplantae sequences from 18 GB DAT.gz → FASTA (~2 hrs)
python scripts/filter_viridiplantae.py \
    --dat-gz  DATA/PLANT_uniprot_new/uniprot_trembl_18_GB/uniprot_trembl_plants.dat.gz \
    --nodes   outputs/taxonomy/nodes.dmp \
    --output  outputs/processed/viridiplantae_clean.fasta

# Convert clean FASTA → HDF5 training index (~15 min)
python scripts/fasta_to_hdf5.py \
    --fasta  outputs/processed/viridiplantae_clean.fasta \
    --output outputs/processed/trembl_full/sequences.h5
```

> Or run `python scripts/parse_uniprot.py data=trembl_full` to do everything from raw DAT.gz in one command (~2.5 hrs total).

### Step 1 — Pretrain

```bash
python scripts/pretrain.py +experiment=pretrain_plant_8m    # ~8M params
python scripts/pretrain.py +experiment=pretrain_plant_35m   # ~35M params
python scripts/pretrain.py +experiment=pretrain_plant_150m  # ~150M params

# Multi-GPU for 150M
python scripts/pretrain.py +experiment=pretrain_plant_150m \
    training.trainer.devices=-1 training.trainer.strategy=ddp

# Quick sanity check
python scripts/pretrain.py +experiment=dev_run
```

### Step 2 — Fine-tune

```bash
# PlantBERT fine-tune
python scripts/finetune.py +experiment=finetune_plant_bert_8m \
    training.pretrained_checkpoint=outputs/YYYY-MM-DD/HH-MM-SS/checkpoints/pretrain/last.ckpt \
    training.task=go_term_prediction \
    training.num_labels=500

# ESM-2 fine-tune (comparison — loads weights from HuggingFace automatically)
python scripts/finetune.py +experiment=finetune_esm2_8m \
    training.task=go_term_prediction \
    training.num_labels=500
```

### Step 3 — Evaluate

```bash
python scripts/evaluate.py \
    training.pretrained_checkpoint=outputs/.../checkpoints/finetune/best.ckpt
```

### Hydra overrides
```bash
# Override any config value at CLI
python scripts/pretrain.py +experiment=pretrain_plant_8m \
    training.optimizer.lr=2e-4 data.batch_size=64

# Hyperparameter sweep
python scripts/pretrain.py --multirun \
    +experiment=pretrain_plant_8m \
    training.optimizer.lr=4e-4,2e-4,1e-4
```

---

## Model Architectures

PlantBERT sizes exactly match ESM-2 parameter counts (Table S3, Lin et al. 2022):

| Config | Layers | Hidden | Heads | FFN | Params | Matches ESM-2 |
|---|---|---|---|---|---|---|
| `plant_bert_8m` | 6 | 320 | 20 | 1280 | ~8M | `esm2_t6_8M_UR50D` |
| `plant_bert_35m` | 12 | 480 | 20 | 1920 | ~35M | `esm2_t12_35M_UR50D` |
| `plant_bert_150m` | 30 | 640 | 20 | 2560 | ~150M | `esm2_t30_150M_UR50D` |

**Training optimizer mirrors ESM-2 exactly:**

| Hyperparameter | Value | Source |
|---|---|---|
| Adam β₁ | 0.9 | ESM-2 Section A.2.4 |
| Adam β₂ | **0.98** | ESM-2 (differs from BERT default 0.999) |
| ε | 1e-8 | ESM-2 Section A.2.4 |
| Peak LR | 4e-4 | ESM-2 Table S3 |
| Warmup | 2000 steps | ESM-2 Section A.2.4 |
| Weight decay | 0.01 | ESM-2 Table S3 |
| Dropout | 0.0 | ESM-2 Section A.2.4 |
| Masking rate | 15% | Standard MLM |

**Known differences from ESM-2:** Absolute position embeddings (ESM-2 uses RoPE); vocabulary of 30 tokens vs 33.

---

## Repository Structure

```
Plant-Protein-BERT/
│
├── src/plant_bert/               Python package
│   │
│   ├── tokenizer/
│   │   ├── amino_acid_tokenizer.py   Character-level: each AA → 1 token (30 vocab).
│   │   │                             Wraps HuggingFace PreTrainedTokenizerFast.
│   │   ├── bpe_tokenizer.py          BPE tokenizer (optional alternative)
│   │   └── esm2_tokenizer_wrapper.py Wraps HF EsmTokenizer for ESM-2 experiments
│   │
│   ├── data/
│   │   ├── uniprot_parser.py     Streaming DAT.gz parser → UniProtRecord dataclasses.
│   │   │                         Parses ID/AC/DE/OS/OX/DR(GO)/SQ. Memory-efficient.
│   │   ├── preprocessing.py      One-time pipeline: DAT.gz → taxonomy filter →
│   │   │                         length/dedup filter → HDF5 index.
│   │   ├── dataset.py            StreamingProteinDataset: HDF5-backed, O(1) random
│   │   │                         access, lazy file-open per DataLoader worker.
│   │   │                         InMemoryProteinDataset: for small datasets.
│   │   └── datamodule.py         UniProtDataModule (Lightning): train/val/test splits.
│   │
│   ├── models/
│   │   ├── bert.py               PlantProteinBERT: wraps BertForMaskedLM.
│   │   │                         get_sequence_embedding() → [CLS] hidden state.
│   │   ├── esm2_finetune.py      ESM2FineTuner: loads ESM-2 from HuggingFace.
│   │   │                         Mean-pools token representations. Drop-in
│   │   │                         replacement for PlantProteinBERT in fine-tuning.
│   │   └── heads.py              SequenceClassificationHead (dropout → linear).
│   │                             MLMHead (reference implementation).
│   │
│   ├── training/
│   │   ├── pretrain.py           MLMPretrainer (Lightning): 15% masking via HF
│   │   │                         DataCollatorForLanguageModeling. Logs val/mlm_loss
│   │   │                         and val/perplexity.
│   │   └── finetune.py           FineTuner (Lightning): encoder + classification head.
│   │                             BCEWithLogitsLoss (multi-label) or CrossEntropyLoss.
│   │                             Logs val/auroc. Early stopping on val/auroc.
│   │
│   ├── evaluation/
│   │   ├── metrics.py            compute_perplexity(loss), compute_mlm_accuracy()
│   │   └── evaluator.py          Evaluator: iterates dataloader, returns
│   │                             mlm_loss / perplexity / mlm_accuracy per split.
│   │
│   └── utils/
│       ├── seed.py               seed_everything(seed)
│       └── logging.py            setup_logger(cfg) — WandB + console
│
├── scripts/                      Hydra entry points
│   ├── parse_uniprot.py          All-in-one: DAT.gz → filter → HDF5 (~2.5 hrs)
│   ├── filter_viridiplantae.py   DAT.gz → clean FASTA (Viridiplantae only)
│   ├── fasta_to_hdf5.py          Clean FASTA → HDF5 training index (~15 min)
│   ├── train_tokenizer.py        Train BPE tokenizer (skip for amino_acid tokenizer)
│   ├── pretrain.py               MLM pretraining
│   ├── finetune.py               Downstream task fine-tuning
│   ├── evaluate.py               Evaluate a checkpoint
│   ├── analyze_uniref50_distribution.py   Generates outputs/figures/details.md
│   ├── plot_uniref50_distribution.py      Superkingdom composition figure
│   └── plot_plant_species_distribution.py Top-25 plant species figure
│
├── configs/                      Hydra config tree
│   ├── config.yaml               Root: composes data + tokenizer + model + training
│   ├── data/trembl_full.yaml     Dataset paths, batch_size, streaming mode
│   ├── tokenizer/
│   │   ├── amino_acid.yaml       30-token AA tokenizer (default)
│   │   ├── bpe.yaml              BPE tokenizer
│   │   └── esm2.yaml             HF EsmTokenizer (for ESM-2 experiments)
│   ├── model/
│   │   ├── plant_bert_8m.yaml    6L / h=320 / ~8M  — matches ESM-2 8M
│   │   ├── plant_bert_35m.yaml   12L / h=480 / ~35M — matches ESM-2 35M
│   │   ├── plant_bert_150m.yaml  30L / h=640 / ~150M — matches ESM-2 150M
│   │   ├── esm2_8m/35m/150m.yaml Load pretrained ESM-2 from HuggingFace
│   │   └── bert_small/base/large.yaml  Legacy configs (not parameter-matched)
│   ├── training/
│   │   ├── pretrain_8m/35m/150m.yaml  ESM-2 exact optimizer per model size
│   │   ├── finetune_plant_bert.yaml   Fine-tune PlantBERT (LR=1e-4, 20 epochs)
│   │   └── finetune_esm2.yaml         Fine-tune ESM-2 (same settings)
│   └── experiment/               Named one-liner run compositions
│       ├── pretrain_plant_8m/35m/150m.yaml
│       ├── finetune_plant_bert_8m/35m/150m.yaml
│       └── finetune_esm2_8m/35m/150m.yaml
│
├── tests/
│   ├── conftest.py           Synthetic in-memory DAT.gz fixture (no real data needed)
│   ├── test_parser.py        4 tests for UniProt DAT parser
│   ├── test_tokenizer.py     5 tests for AminoAcidTokenizer
│   └── test_metrics.py       4 tests for perplexity / MLM accuracy
│
├── artifacts/tokenizers/     Trained tokenizer vocab files (committed, small)
├── outputs/                  All run artifacts — gitignored
│   ├── YYYY-MM-DD/HH-MM-SS/
│   │   ├── .hydra/           Exact config snapshot per run (full reproducibility)
│   │   └── checkpoints/pretrain/  last.ckpt + top-k checkpoints
│   ├── figures/              Distribution plots + details.md
│   └── processed/trembl_full/sequences.h5
│
├── DATA/                     Raw data — never modified
├── pyproject.toml            Package + dev dependencies
├── environment.yml           Conda env spec
├── requirements.in           Direct unpinned dependencies (source of truth)
└── requirements.txt          Fully pinned snapshot (pip install -r for exact repro)
```

---

## Running Tests

```bash
pytest                        # all 13 tests
pytest tests/test_parser.py   # single file
pytest -k "test_tokenizer"    # pattern
```

No dependency on the 18 GB TrEMBL file — all tests use synthetic in-memory data.

---

## Downstream Tasks

| Task | Type | Labels | Plant-specific signal |
|---|---|---|---|
| GO term prediction | Multi-label | ~500–1500 | Photosynthesis, cell wall biosynthesis GO terms |
| Subcellular localization | Single-label | 10 | Chloroplast/plastid targeting peptides |
| Enzyme function (EC) | Multi-label | varies | Plant secondary metabolite biosynthesis |
| Secondary structure | Per-residue (3-class) | 3 | **Negative control** — ESM-2 should be competitive |

Secondary structure is a negative control: no plant-specific advantage is expected, so if PlantBERT wins here, something is wrong with the experimental setup.

---

## References

- Lin et al. (2022). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* bioRxiv 2022.07.20.500902
- Devlin et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT 2019
- UniProt Consortium. *UniProt: the Universal Protein Knowledgebase.* Nucleic Acids Research
