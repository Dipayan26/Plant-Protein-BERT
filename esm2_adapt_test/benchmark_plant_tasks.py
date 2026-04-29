"""Linear-probe benchmark: vanilla ESM-2 vs. plant-adapted ESM-2.

Tasks
-----
  1. Subcellular localization — 10-class single-label, metric: accuracy + macro-F1
  2. GO-term prediction       — top-50 most frequent GO terms, multi-label, metric: macro AUROC

Data
----
  ~2 000 reviewed Arabidopsis thaliana proteins fetched from UniProt REST API,
  cached to outputs/benchmark/ (shared across all model sizes).

Usage
-----
    python esm2_adapt_test/benchmark_plant_tasks.py --config esm2_adapt_test/configs/35m.yaml
    python esm2_adapt_test/benchmark_plant_tasks.py --config esm2_adapt_test/configs/150m.yaml
    python esm2_adapt_test/benchmark_plant_tasks.py --config esm2_adapt_test/configs/35m.yaml --max-proteins 500
"""


from __future__ import annotations
import argparse
import csv
import io
import json
import re
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import requests
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import EsmForMaskedLM, EsmTokenizer

# UniProt cache is model-agnostic — shared by all model sizes
CACHE_DIR  = Path("outputs/benchmark")
CACHE_FILE = CACHE_DIR / "arabidopsis_reviewed.jsonl"

LOCATION_MAP = {
    "chloroplast":          "Chloroplast",
    "thylakoid":            "Chloroplast",
    "mitochondri":          "Mitochondrion",
    "nucleus":              "Nucleus",
    "cytoplasm":            "Cytoplasm",
    "cytosol":              "Cytoplasm",
    "plasma membrane":      "Plasma membrane",
    "endoplasmic reticulum":"ER",
    "vacuol":               "Vacuole",
    "golgi":                "Golgi",
    "peroxisom":            "Peroxisome",
    "secreted":             "Secreted",
    "extracellular":        "Secreted",
    "cell wall":            "Cell wall",
}
MIN_CLASS_SIZE = 30


# ── Data fetching ──────────────────────────────────────────────────────────────

def fetch_uniprot(max_proteins: int = 2000) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        records = [json.loads(l) for l in CACHE_FILE.read_text().splitlines() if l.strip()]
        if records:
            print(f"  Loaded {len(records)} proteins from cache ({CACHE_FILE})")
            return records[:max_proteins]

    print("  Fetching from UniProt REST API (Arabidopsis thaliana, reviewed)...")
    params = {
        "query":  "(organism_id:3702) AND (reviewed:true)",
        "fields": "accession,sequence,go,cc_subcellular_location",
        "format": "tsv",
        "size":   500,
    }
    records: list[dict] = []
    cursor = None
    page = 0
    while len(records) < max_proteins:
        if cursor:
            params["cursor"] = cursor
        elif "cursor" in params:
            del params["cursor"]
        resp = requests.get("https://rest.uniprot.org/uniprotkb/search", params=params, timeout=60)
        resp.raise_for_status()
        page_records = list(csv.DictReader(io.StringIO(resp.text), delimiter="\t"))
        if not page_records:
            break
        records.extend(page_records)
        page += 1
        print(f"    Page {page}: {len(page_records)} proteins (total {len(records)})")
        link = resp.headers.get("Link", "")
        if 'rel="next"' not in link:
            break
        for part in link.split(","):
            if 'rel="next"' in part:
                from urllib.parse import parse_qs, urlparse
                qs = parse_qs(urlparse(part.split(";")[0].strip().strip("<>")).query)
                cursor = qs.get("cursor", [None])[0]
                break
        time.sleep(0.3)

    records = records[:max_proteins]
    CACHE_FILE.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    print(f"  Cached {len(records)} proteins to {CACHE_FILE}")
    return records


# ── Label parsing ──────────────────────────────────────────────────────────────

def parse_location(raw: str) -> str | None:
    if not raw:
        return None
    text = raw.lower()
    for keyword, label in LOCATION_MAP.items():
        if keyword in text:
            return label
    return None


def parse_go_ids(raw: str) -> list[str]:
    return re.findall(r"GO:\d+", raw) if raw else []


# ── Embedding extraction ───────────────────────────────────────────────────────

@torch.no_grad()
def embed_sequences(
    model: EsmForMaskedLM,
    tokenizer: EsmTokenizer,
    sequences: list[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    model.eval()
    all_embs = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        cls_emb = model.esm(**enc).last_hidden_state[:, 0, :].cpu().float().numpy()
        all_embs.append(cls_emb)
        if (i // batch_size) % 5 == 0:
            print(f"    {i + len(batch)}/{len(sequences)}", end="\r")
    print()
    return np.vstack(all_embs)


# ── Linear probes ──────────────────────────────────────────────────────────────

def linear_probe_localization(X: np.ndarray, y: list[str], label: str) -> dict:
    classes = sorted(Counter(y).keys())
    y_enc = np.array([classes.index(c) for c in y])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro", zero_division=0)
    print(f"    [{label}] accuracy={acc:.3f}  macro-F1={f1:.3f}  (test n={len(y_te)})")
    return {"accuracy": round(acc, 4), "macro_f1": round(f1, 4), "n_test": len(y_te), "n_classes": len(classes)}


def linear_probe_go(X: np.ndarray, y_bin: np.ndarray, label: str) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    aucs: list[float] = []
    for j in range(y_bin.shape[1]):
        if y_te[:, j].sum() == 0:
            continue
        clf = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        clf.fit(X_tr, y_tr[:, j])
        aucs.append(float(roc_auc_score(y_te[:, j], clf.predict_proba(X_te)[:, 1])))
    macro_auc = round(float(np.mean(aucs)) if aucs else 0.0, 4)
    print(f"    [{label}] macro-AUROC={macro_auc:.3f}  ({len(aucs)} GO terms, test n={len(y_te)})")
    return {"macro_auroc": macro_auc, "n_go_terms_evaluated": len(aucs), "n_test": len(y_te)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main(cfg: dict, max_proteins: int, device_str: str = "auto") -> None:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    if device.type == "cpu":
        print("WARNING: running on CPU — activate the plantbert conda env or pass --device cuda")
    model_name = cfg["model_name"]
    results_dir = Path("esm2_adapt_test/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Model : {model_name}")
    print(f"Device: {device}\n")

    # ── Data ─────────────────────────────────────────────────────────────
    print("=== Data ===")
    records = fetch_uniprot(max_proteins)
    print(f"  Total proteins: {len(records)}")

    seqs, locs, go_ids_list = [], [], []
    for r in records:
        seq = r.get("Sequence", r.get("sequence", "")).strip()
        if not seq or len(seq) < 20:
            continue
        seqs.append(seq)
        locs.append(parse_location(r.get("Subcellular location [CC]", r.get("cc_subcellular_location", ""))))
        go_ids_list.append(parse_go_ids(r.get("Gene Ontology (GO)", r.get("go", ""))))

    # Localization dataset
    loc_seqs   = [s for s, l in zip(seqs, locs) if l is not None]
    loc_labels = [l for l in locs if l is not None]
    min_size   = max(5, MIN_CLASS_SIZE if len(seqs) >= 500 else len(seqs) // 20)
    valid_cls  = {c for c, n in Counter(loc_labels).items() if n >= min_size}
    loc_seqs_f   = [s for s, l in zip(loc_seqs, loc_labels) if l in valid_cls]
    loc_labels_f = [l for l in loc_labels if l in valid_cls]
    print(f"\n  Localization: {len(loc_seqs_f)} proteins, {len(valid_cls)} classes")

    # GO dataset
    go_counter: Counter = Counter(g for ids in go_ids_list for g in ids)
    top50_go  = [g for g, _ in go_counter.most_common(50)]
    go_seqs   = [s for s, ids in zip(seqs, go_ids_list) if any(g in top50_go for g in ids)]
    go_labels = [[g for g in ids if g in top50_go] for _, ids in zip(seqs, go_ids_list) if any(g in top50_go for g in ids)]
    mlb       = MultiLabelBinarizer(classes=top50_go)
    go_bin: np.ndarray = np.asarray(mlb.fit_transform(go_labels))
    print(f"  GO terms:     {len(go_seqs)} proteins, top-{len(top50_go)} GO terms")

    # ── Models ────────────────────────────────────────────────────────────
    print("\n=== Loading models ===")
    tokenizer = EsmTokenizer.from_pretrained(cfg["vanilla_hub"])
    vanilla: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(cfg["vanilla_hub"]).to(device)
    adapted: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(cfg["adapted_path"]).to(device)

    # ── Embeddings ────────────────────────────────────────────────────────
    print("\n=== Extracting embeddings ===")
    print("  Vanilla — localization:")
    emb_v_loc = embed_sequences(vanilla, tokenizer, loc_seqs_f, device)
    print("  Adapted — localization:")
    emb_a_loc = embed_sequences(adapted, tokenizer, loc_seqs_f, device)
    print("  Vanilla — GO terms:")
    emb_v_go = embed_sequences(vanilla, tokenizer, go_seqs, device)
    print("  Adapted — GO terms:")
    emb_a_go = embed_sequences(adapted, tokenizer, go_seqs, device)

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n=== Task 1: Subcellular localization ===")
    res_v_loc = linear_probe_localization(emb_v_loc, loc_labels_f, "Vanilla")
    res_a_loc = linear_probe_localization(emb_a_loc, loc_labels_f, "Adapted")

    print("\n=== Task 2: GO-term prediction ===")
    res_v_go = linear_probe_go(emb_v_go, go_bin, "Vanilla")
    res_a_go = linear_probe_go(emb_a_go, go_bin, "Adapted")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Task':<35} {'Vanilla':>10} {'Adapted':>10} {'Δ':>8}")
    print("-" * 63)
    print(f"  {'Localization accuracy':<33} {res_v_loc['accuracy']:>10.3f} {res_a_loc['accuracy']:>10.3f} {res_a_loc['accuracy'] - res_v_loc['accuracy']:>+8.3f}")
    print(f"  {'Localization macro-F1':<33} {res_v_loc['macro_f1']:>10.3f} {res_a_loc['macro_f1']:>10.3f} {res_a_loc['macro_f1'] - res_v_loc['macro_f1']:>+8.3f}")
    print(f"  {'GO-term macro-AUROC':<33} {res_v_go['macro_auroc']:>10.3f} {res_a_go['macro_auroc']:>10.3f} {res_a_go['macro_auroc'] - res_v_go['macro_auroc']:>+8.3f}")
    print(f"\n  Localization → {'adapted BETTER' if res_a_loc['accuracy'] > res_v_loc['accuracy'] else 'vanilla BETTER'}")
    print(f"  GO terms     → {'adapted BETTER' if res_a_go['macro_auroc'] > res_v_go['macro_auroc'] else 'vanilla BETTER'}")

    # ── Save ──────────────────────────────────────────────────────────────
    results = {
        "model":      model_name,
        "date":       str(date.today()),
        "vanilla_hub": cfg["vanilla_hub"],
        "adapted_path": cfg["adapted_path"],
        "localization": {"vanilla": res_v_loc, "adapted": res_a_loc},
        "go_terms":     {"vanilla": res_v_go,  "adapted": res_a_go},
    }
    out_path = results_dir / f"{model_name}_benchmark_{date.today()}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="esm2_adapt_test/configs/35m.yaml",
        help="Path to model config YAML (e.g. esm2_adapt_test/configs/35m.yaml)",
    )
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto (default: auto)")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    max_proteins = args.max_proteins or cfg.get("max_proteins", 2000)
    main(cfg, max_proteins, args.device)
