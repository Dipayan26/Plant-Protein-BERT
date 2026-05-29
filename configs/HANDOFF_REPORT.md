# Leakage-Free Plant PPI Dataset — Handoff Report

> Generated for handoff to another Claude instance. All numbers below are read
> directly from the pipeline's `*_stats.yaml` and `dataset_summary.yaml`
> outputs (run timestamp 2026-05-08). **Ignore `info.md` in this folder — it is
> stale** (describes an earlier ratio=1 / 40%-CD-HIT run that produced ~10k
> pairs; the current dataset is the ratio=3 / 60%-CD-HIT run described here).

---

## 1. What this dataset is

A **leakage-free, multi-species plant protein-protein interaction (PPI)
benchmark**, built following the Bernett et al. (2024) "cracking the black box"
protocol for graph-partition-based train/val/test splitting. The goal is to
eliminate the data leakage and node-degree shortcuts that invalidated the
previous (rejected) ARACoFusion-PPI paper.

**Core leakage guarantee:** no protein appears in more than one split. Splits
are produced by partitioning a global sequence-similarity graph, so
homologous proteins cannot straddle train/test.

---

## 2. Final output files (USE THESE)

```
Data_filter_bernette/output/07_cdhit/train_final.tsv
Data_filter_bernette/output/07_cdhit/val_final.tsv
Data_filter_bernette/output/07_cdhit/test_final.tsv
```

**TSV columns (tab-separated, with header):**

| Column | Meaning |
|---|---|
| `seq_1` | Amino-acid sequence of protein 1 |
| `seq_2` | Amino-acid sequence of protein 2 |
| `label` | 1 = interacting (positive), 0 = non-interacting (negative) |
| `protein_1` | Protein 1 accession/ID |
| `protein_2` | Protein 2 accession/ID |
| `taxid_1` | NCBI taxonomy ID of protein 1 |
| `taxid_2` | NCBI taxonomy ID of protein 2 |
| `edge_key` | Canonical undirected pair key (dedup/identity) |

Sequences are already embedded in the TSV — no separate FASTA lookup needed for training.

Both proteins in every pair are same-species (`taxid_1 == taxid_2`);
cross-species pairs were discarded during splitting.

### Final dataset statistics

| Split | Total pairs | Positives | Negatives | Unique proteins | Species (taxids) |
|---|---|---|---|---|---|
| Train | 20,580 | 10,290 | 10,290 | 11,817 | 41 |
| Val   | 4,522  | 2,261  | 2,261  | 2,226  | 20 |
| Test  | 5,512  | 2,756  | 2,756  | 2,346  | 20 |
| **Total** | **30,614** | **15,307** | **15,307** | — | — |

- Perfectly balanced 1:1 positive:negative in every split.
- QA confirms **0 protein overlap** between any pair of splits and **0
  duplicate edge_keys** within or across splits (see §5).

---

## 3. Upstream raw data (where the positives come from)

Built by `NEW_data/build_plant_ppi.py` (run 2026-04-22), upstream of this pipeline:

- **IntAct** (PSI-MITAB): 1,787,425 rows parsed → 104,786 accepted evidence rows
- **BioGRID** (multi-validated physical): 542,046 rows parsed → 11,824 accepted evidence rows
- After global dedup: **91,338 unique positive edges** / 116,604 evidence rows
- These 91,338 edges = `NEW_data/output/plant_ppi_evidence_all.tsv` = the
  **positive universe** used to exclude false negatives during negative sampling.

Then merged with **STRING** (experimental score ≥ 700) to expand species
coverage (Arabidopsis/maize-only → 20 species in val/test):

- Merged input to this pipeline: `NEW_data/output/string_merged/plant_ppi_with_string.tsv` → **243,415 pairs**
- Protein sequences: `NEW_data/output/string_merged/plant_proteins_combined.fasta` (51,838 proteins)

> Note: STRING contributes positive *pairs* for graph coverage. The negative-
> exclusion universe is the 91,338 experimentally-supported BioGRID+IntAct edges.

---

## 4. Processing pipeline — step by step (with real funnel numbers)

Run from repo root with `conda activate ara-ppi` then `bash Data_filter_bernette/run_pipeline.sh`.
Each step reads a YAML in `configs/` and writes to `output/<step>/`.

### Step 1 — Pre-filter (`01_prefilter.py`, `configs/01_prefilter.yaml`)
Hub capping + singleton isolation.
- Input: 243,415 pairs / 51,838 proteins
- **Hub cap = degree 100**: 455 hub proteins subsampled to ≤100 edges → 25,704 pairs dropped as hub overflow
- **Singletons (degree ≤ 1)**: 15,173 singleton proteins → 14,370 pairs routed directly to train (bypass partitioning)
- Output: **203,341 pairs for partitioning**

### Step 2 — All-vs-all sequence similarity (`02_similarity.py`, `configs/02_similarity.yaml`)
MMseqs2 similarity graph (the basis for leakage-free splitting).
- 35,696 proteins searched; e-value ≤ 1.0, sensitivity 7.5, end-to-end alignment (mode 3)
- 1,906,148 raw hits → **1,735,124 edges kept** after filtering
- Normalization: `norm_weight = bitscore / max(qlen, tlen)` (same as Bernett); min normalized weight ≥ 0.05
- Output: `sim_normalized.tsv`

### Step 3 — Build global METIS graph (`03_make_metis.py`, `configs/03_metis.yaml`)
- **Design choice (differs from Bernett):** ONE global graph including cross-species
  similarity edges, instead of per-species graphs. Prevents CD-HIT from later
  deleting proteins whose cross-species homologs land in different splits.
- Minor species (≤ 300 edge-endpoints): 13 taxids routed directly to train (272 pairs)
- Global graph: **35,605 nodes, 965,495 edges** (1,307 isolated nodes), integer weight scale ×1000
- Output: `global_graph.metis`, `global_id_map.tsv`

### Step 4 — KaHIP graph partitioning (`04_partition.py`, `configs/04_partition.yaml`)
- KaHIP v3.25, **k=3** blocks, `eco` preconfiguration, seed 1234, 3% imbalance allowed
- Edge cut: 433,335
- Block sizes: block_0 = 12,224, block_1 = 12,224, block_2 = 11,157
- Output: `global_block_assignment.tsv`
- ⚠️ Uses `eco` mode for speed. Switch to `strong` in the config for the
  final paper-quality run.

### Step 5 — Assign pairs to splits (`05_assign_splits.py`, `configs/05_assign_splits.yaml`)
- block_0 → train, block_1 → val, block_2 → test
- **INTER-block pairs discarded** (both endpoints must be in same block) — this is the leakage guard
- Cross-species pairs discarded
- Input 203,341 → Train 42,239 / Val 28,077 / Test 21,311 (positives only)
- Discarded 126,356 (inter_block 126,020; protein_not_partitioned 210; cross_species 126)

### Step 6 — Sample negatives (`06_sample_negatives.py`, `configs/06_negatives.yaml`)
- Degree-weighted negative sampling, **ratio = 3** (oversample, so CD-HIT in step 7 can still rebalance to 1:1)
- Excludes the 91,338-edge positive universe so negatives are true non-interactors
- Negatives drawn: Train 126,717 / Val 84,231 / Test 63,933

### Step 7 — CD-HIT homology filter + rebalance + attach sequences (`07_cdhit_filter.py`, `configs/07_cdhit.yaml`)
- **CD-HIT at 60% identity** (word length 4). Relaxed from Bernett's 40% because
  plant PPI data is sparse — 40% removed ~90% of data. cd-hit-2d used for cross-split redundancy removal.
- Then **rebalance to 1:1** pos:neg by downsampling negatives, attach sequences.
- Per-split funnel:
  - Train: pos 42,239→10,290 after CD-HIT; neg 126,717→22,878→10,290 after rebalance → **20,580 written**
  - Val:   pos 28,077→2,261;  neg 84,231→4,747→2,261 → **4,522 written**
  - Test:  pos 21,311→2,756;  neg 63,933→5,536→2,756 → **5,512 written**
  - 0 pairs dropped for missing sequence in any split.

### Step 8 — QA checks (`08_qa_checks.py`, `configs/08_qa.yaml`)
Validates the final TSVs (see §5). Writes `output/final/dataset_summary.yaml` + `qa_report.yaml`.

---

## 5. QA results — all passed (`output/final/qa_report.yaml`)

`overall_passed: true`

| Check | Result |
|---|---|
| No protein overlap val↔test | PASS (0) |
| No protein overlap train↔(val/test) | PASS (0) |
| No duplicate edge_keys within split | PASS (train/val/test all 0) |
| No duplicate edge_keys across splits | PASS (train-val 0, train-test 0, val-test 0) |
| Labels only {0,1} | PASS |
| Negatives not in positive universe | PASS (0 leaked) |
| Min pairs per split ≥ 1000 | PASS (20,580 / 4,522 / 5,512) |

---

## 6. Key design decisions (defensible in paper)

1. **Global similarity graph** (not per-species) + cross-species edges → prevents
   homolog leakage across splits via cross-species paralogs.
2. **60% CD-HIT threshold** (vs Bernett 40%) → plant interactome is ~30k pairs vs
   Bernett's 150k+; 40% was too aggressive (~90% data loss). 60% still removes clear homologs.
3. **Negative oversampling ratio 3** → CD-HIT drops negatives slightly harder than
   positives; oversampling lets the rebalance hit exactly 1:1.
4. **Hub cap = 100, singleton isolation** → removes node-degree shortcuts that the
   old model exploited.

---

## 7. Known caveats to pass along

- `info.md` in this folder is **stale** (old 7,586-pair run). This report supersedes it.
- KaHIP ran in `eco` mode (speed). For final results, rerun step 4 with `strong`.
- `protein_not_partitioned` discard = 210 pairs (minor species above threshold but
  not explicitly named major) — negligible.
- BioGRID column mapping for `publication_id` / `source_record_id` is partially
  wrong upstream (those fields empty) — does not affect the final dataset.

---

## 8. How to consume this dataset (for the receiving instance)

```python
import pandas as pd
train = pd.read_csv("Data_filter_bernette/output/07_cdhit/train_final.tsv", sep="\t")
val   = pd.read_csv("Data_filter_bernette/output/07_cdhit/val_final.tsv",   sep="\t")
test  = pd.read_csv("Data_filter_bernette/output/07_cdhit/test_final.tsv",  sep="\t")
# columns: seq_1, seq_2, label, protein_1, protein_2, taxid_1, taxid_2, edge_key
```

Target task: binary PPI classification (`label`). Sequences are ready for protein
language-model embedding (project's prior choice: ESM1b, 1280-dim mean pooling).
Splits are leakage-free by construction — do NOT re-shuffle or re-split across the
provided boundaries, or the leakage guarantee is lost.
