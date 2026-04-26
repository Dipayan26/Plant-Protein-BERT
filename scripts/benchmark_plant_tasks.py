"""Linear-probe benchmark: vanilla ESM-2 8M vs. plant-adapted ESM-2 8M.

Tasks
-----
  1. Subcellular localization  — 10-class single-label, metric: accuracy + macro-F1
  2. GO-term prediction        — top-50 most frequent GO terms, multi-label, metric: macro AUROC

Data
----
  ~2 000 reviewed Arabidopsis thaliana proteins fetched live from UniProt REST API,
  cached to outputs/benchmark/ so the download runs only once.

Method
------
  Frozen [CLS] embeddings → sklearn LogisticRegression (linear probe).
  No gradients through the encoder — tests what the PLM learned, not what
  the head can learn on top of either model.

Usage
-----
    python scripts/benchmark_plant_tasks.py
    python scripts/benchmark_plant_tasks.py --adapted checkpoints/esm2_adapt/8m/hf_model
    python scripts/benchmark_plant_tasks.py --max-proteins 500   # quick smoke-test
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import EsmForMaskedLM, EsmTokenizer

VANILLA = "facebook/esm2_t6_8M_UR50D"
CACHE_DIR = Path("outputs/benchmark")
CACHE_FILE = CACHE_DIR / "arabidopsis_reviewed.jsonl"

# ── Subcellular location keyword → canonical class ─────────────────────────
LOCATION_MAP = {
    "chloroplast": "Chloroplast",
    "thylakoid": "Chloroplast",
    "mitochondri": "Mitochondrion",
    "nucleus": "Nucleus",
    "cytoplasm": "Cytoplasm",
    "cytosol": "Cytoplasm",
    "plasma membrane": "Plasma membrane",
    "endoplasmic reticulum": "ER",
    "vacuol": "Vacuole",
    "golgi": "Golgi",
    "peroxisom": "Peroxisome",
    "secreted": "Secreted",
    "extracellular": "Secreted",
    "cell wall": "Cell wall",
}
# Minimum proteins per class for training
MIN_CLASS_SIZE = 30


# ── UniProt data fetching ───────────────────────────────────────────────────

def fetch_uniprot(max_proteins: int = 2000) -> list[dict]:
    """Download reviewed Arabidopsis thaliana proteins from UniProt REST API."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        records = [json.loads(l) for l in CACHE_FILE.read_text().splitlines() if l.strip()]
        if records:
            print(f"  Loaded {len(records)} proteins from cache ({CACHE_FILE})")
            return records[:max_proteins]

    print("  Fetching from UniProt REST API (Arabidopsis thaliana, reviewed)...")
    base = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": "(organism_id:3702) AND (reviewed:true)",
        "fields": "accession,sequence,go,cc_subcellular_location",
        "format": "tsv",
        "size": 500,
    }

    records: list[dict] = []
    cursor = None
    page = 0
    while len(records) < max_proteins:
        if cursor:
            params["cursor"] = cursor
        elif "cursor" in params:
            del params["cursor"]

        resp = requests.get(base, params=params, timeout=60)
        resp.raise_for_status()

        reader = csv.DictReader(io.StringIO(resp.text), delimiter="\t")
        page_records = list(reader)
        if not page_records:
            break
        records.extend(page_records)
        page += 1
        print(f"    Page {page}: {len(page_records)} proteins (total {len(records)})")

        # UniProt pagination: Link header looks like:
        # <https://...?...&cursor=XXX&...>; rel="next"
        link = resp.headers.get("Link", "")
        if 'rel="next"' not in link:
            break
        from urllib.parse import urlparse, parse_qs
        for part in link.split(","):
            if 'rel="next"' in part:
                url_part = part.split(";")[0].strip().strip("<>")
                qs = parse_qs(urlparse(url_part).query)
                cursor = qs.get("cursor", [None])[0]
                break

        time.sleep(0.3)  # be polite to the API

    records = records[:max_proteins]
    with CACHE_FILE.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Cached {len(records)} proteins to {CACHE_FILE}")
    return records


# ── Label parsing ───────────────────────────────────────────────────────────

def parse_location(raw: str) -> str | None:
    """Extract the primary canonical subcellular location from UniProt text."""
    if not raw:
        return None
    text = raw.lower()
    for keyword, label in LOCATION_MAP.items():
        if keyword in text:
            return label
    return None


def parse_go_ids(raw: str) -> list[str]:
    """Extract GO term IDs from UniProt 'Gene Ontology (GO)' TSV field.

    Format: "term name [GO:0009941]; other term [GO:0005737]; ..."
    """
    if not raw:
        return []
    import re
    return re.findall(r"GO:\d+", raw)


# ── Embedding extraction ────────────────────────────────────────────────────

@torch.no_grad()
def embed_sequences(
    model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    sequences: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """Return [CLS] embeddings for all sequences. Shape: [N, hidden_size]."""
    model.eval()
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.esm(**enc)
        cls_emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()
        all_embs.append(cls_emb)
        if (i // batch_size) % 5 == 0:
            print(f"    {i + len(batch)}/{len(sequences)}", end="\r")
    print()
    return np.vstack(all_embs)


# ── Linear probe ────────────────────────────────────────────────────────────

def linear_probe_localization(
    X: np.ndarray, y: list[str], label: str
) -> dict:
    """Train and evaluate a logistic regression on subcellular localization."""
    classes = sorted(Counter(y).keys())
    y_enc = np.array([classes.index(c) for c in y])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"    [{label}] accuracy={acc:.3f}  macro-F1={f1:.3f}  (test n={len(y_te)})")
    return {"accuracy": acc, "macro_f1": f1, "n_test": len(y_te), "n_classes": len(classes)}


def linear_probe_go(
    X: np.ndarray, y_bin: np.ndarray, label: str
) -> dict:
    """Train and evaluate one-vs-rest logistic regression for GO term prediction."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_bin, test_size=0.2, random_state=42
    )
    # One classifier per GO term
    n_labels = y_bin.shape[1]
    aucs: list[float] = []
    for j in range(n_labels):
        if y_te[:, j].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        clf.fit(X_tr, y_tr[:, j])
        prob = clf.predict_proba(X_te)[:, 1]
        aucs.append(float(roc_auc_score(y_te[:, j], prob)))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0
    print(f"    [{label}] macro-AUROC={macro_auc:.3f}  (evaluated {len(aucs)} GO terms, test n={len(y_te)})")
    return {"macro_auroc": macro_auc, "n_go_terms_evaluated": len(aucs), "n_test": len(y_te)}


# ── Main ────────────────────────────────────────────────────────────────────

def main(adapted_path: str, max_proteins: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Fetch and parse data ─────────────────────────────────────────────
    print("=== Data ===")
    records = fetch_uniprot(max_proteins)
    print(f"  Total proteins: {len(records)}")

    # Parse sequences and labels
    seqs, locs, go_ids_list = [], [], []
    for r in records:
        seq = r.get("Sequence", r.get("sequence", "")).strip()
        if not seq or len(seq) < 20:
            continue
        loc_raw = r.get("Subcellular location [CC]", r.get("cc_subcellular_location", ""))
        go_raw = r.get("Gene Ontology (GO)", r.get("go", ""))
        seqs.append(seq)
        locs.append(parse_location(loc_raw))
        go_ids_list.append(parse_go_ids(go_raw))

    # ── Localization dataset ─────────────────────────────────────────────
    loc_seqs, loc_labels = [], []
    for seq, loc in zip(seqs, locs):
        if loc is not None:
            loc_seqs.append(seq)
            loc_labels.append(loc)

    # Drop classes with too few samples for stratified split.
    # Scale minimum dynamically so small smoke-test runs don't collapse to 1 class.
    min_size = max(5, MIN_CLASS_SIZE if len(seqs) >= 500 else len(seqs) // 20)
    class_counts = Counter(loc_labels)
    valid_classes = {c for c, n in class_counts.items() if n >= min_size}
    loc_seqs_f = [s for s, l in zip(loc_seqs, loc_labels) if l in valid_classes]
    loc_labels_f = [l for l in loc_labels if l in valid_classes]

    print(f"\n  Localization task: {len(loc_seqs_f)} proteins, {len(valid_classes)} classes")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        flag = "" if cls in valid_classes else " (dropped, too few)"
        print(f"    {cls:20s} {cnt:4d}{flag}")

    # ── GO term dataset ──────────────────────────────────────────────────
    go_counter: Counter = Counter()
    for ids in go_ids_list:
        go_counter.update(ids)
    top50_go = [go for go, _ in go_counter.most_common(50)]

    go_seqs = [s for s, ids in zip(seqs, go_ids_list) if any(g in top50_go for g in ids)]
    go_labels_raw = [
        [g for g in ids if g in top50_go]
        for _, ids in zip(seqs, go_ids_list)
        if any(g in top50_go for g in ids)
    ]
    mlb = MultiLabelBinarizer(classes=top50_go)
    go_bin: np.ndarray = np.asarray(mlb.fit_transform(go_labels_raw))  # type: ignore[assignment]
    print(f"\n  GO term task: {len(go_seqs)} proteins, top-{len(top50_go)} GO terms")

    # ── Load models ──────────────────────────────────────────────────────
    print("\n=== Loading models ===")
    tokenizer = EsmTokenizer.from_pretrained(VANILLA)
    vanilla: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(VANILLA).to(device)  # type: ignore[assignment]
    adapted: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(adapted_path).to(device)  # type: ignore[assignment]

    # ── Embed ─────────────────────────────────────────────────────────────
    print("\n=== Extracting embeddings ===")

    print("  Vanilla — localization sequences:")
    emb_v_loc = embed_sequences(vanilla, tokenizer, loc_seqs_f, device)
    print("  Adapted — localization sequences:")
    emb_a_loc = embed_sequences(adapted, tokenizer, loc_seqs_f, device)

    print("  Vanilla — GO sequences:")
    emb_v_go = embed_sequences(vanilla, tokenizer, go_seqs, device)
    print("  Adapted — GO sequences:")
    emb_a_go = embed_sequences(adapted, tokenizer, go_seqs, device)

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n=== Task 1: Subcellular localization ===")
    res_v_loc = linear_probe_localization(emb_v_loc, loc_labels_f, "Vanilla")
    res_a_loc = linear_probe_localization(emb_a_loc, loc_labels_f, "Adapted")

    print("\n=== Task 2: GO-term prediction (top-50 GO terms) ===")
    res_v_go = linear_probe_go(emb_v_go, go_bin, "Vanilla")
    res_a_go = linear_probe_go(emb_a_go, go_bin, "Adapted")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Task':<35} {'Vanilla':>10} {'Adapted':>10} {'Δ':>8}")
    print("-" * 63)

    acc_v = res_v_loc["accuracy"]
    acc_a = res_a_loc["accuracy"]
    f1_v  = res_v_loc["macro_f1"]
    f1_a  = res_a_loc["macro_f1"]
    auc_v = res_v_go["macro_auroc"]
    auc_a = res_a_go["macro_auroc"]

    print(f"  {'Localization accuracy':<33} {acc_v:>10.3f} {acc_a:>10.3f} {acc_a - acc_v:>+8.3f}")
    print(f"  {'Localization macro-F1':<33} {f1_v:>10.3f} {f1_a:>10.3f} {f1_a - f1_v:>+8.3f}")
    print(f"  {'GO-term macro-AUROC':<33} {auc_v:>10.3f} {auc_a:>10.3f} {auc_a - auc_v:>+8.3f}")

    print(f"\n  Classes: {res_a_loc['n_classes']} locations  |  {res_a_go['n_go_terms_evaluated']} GO terms evaluated")
    print(f"  Test set: {res_a_loc['n_test']} (loc)  |  {res_a_go['n_test']} (GO)")

    verdict_loc = "adapted BETTER" if acc_a > acc_v else "vanilla BETTER"
    verdict_go  = "adapted BETTER" if auc_a > auc_v else "vanilla BETTER"
    print(f"\n  Localization → {verdict_loc}")
    print(f"  GO terms     → {verdict_go}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapted",
        default="checkpoints/esm2_adapt/8m/hf_model",
        help="Path to plant-adapted HF model directory",
    )
    parser.add_argument(
        "--max-proteins",
        type=int,
        default=2000,
        help="Max proteins to download from UniProt (default 2000)",
    )
    args = parser.parse_args()
    main(args.adapted, args.max_proteins)
