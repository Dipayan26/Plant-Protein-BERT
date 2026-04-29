"""Sanity tests for a plant-adapted ESM-2 checkpoint.

Three tests:
  1. MLM perplexity  — adapted vs. vanilla on plant sequences
  2. Fill-mask demo  — qualitative masked-token prediction on RuBisCO
  3. Embedding similarity — [CLS] cosine distances by protein family

Usage:
    python esm2_adapt_test/test_adapted_esm2.py --config esm2_adapt_test/configs/35m.yaml
    python esm2_adapt_test/test_adapted_esm2.py --config esm2_adapt_test/configs/150m.yaml
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path

import torch
import yaml
from transformers import EsmForMaskedLM, EsmTokenizer

PLANT_SEQS = [
    # RuBisCO large subunit fragment (Arabidopsis)
    "MSPQTETKASVGFKAGVKDYKLTYYTPEYETKDTDILAAFRVTPQPGVPPEEAGAAVAAESSTGTWTTVWTDGLTSLDRYKGRCYRIECFSVQKILDSGKMDKFQKVGQNTLFQPQISDSGHTSSPLSSPESGGVHVHQTAEEAAYIDLKKILPPANSPVGGVVAGGQTMTVIAGEGQLEHLKALLRGLHSPRSGRPPTGRFKDKNVTTLTLGDLGNHPFNPNSSIEQDNAQVQALRKLGTDLFRARIVPLDRGLDAQVAIRSALRQELTADLAQAKLYAEAAAMMAYTMSPQTETKRSVGFKAGVKDYKLTYYTPEYETK",
    # RuBisCO small subunit
    "MASSIAAASSATVAPFQGLKSAASFPVSRKQNLDITSISNGGRVKFSPQTLGGLPFQTPADMVAPFTGLKNATNLVPFHGLKSSTNLVPFHGLKSSTNLVPFHGLKS",
    # Actin (plant variant)
    "MCDEDETTALVCDNGSGLVKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQSKRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREKMTQIMFETFNTPAMYVAIQAVLSLYASGRTTGIVMDSGDGVTHTVPIYEGYALPHAILRLDLAGRDLTDYLMKILTERGYSFTTTAEREIVRDIKEKLCYVALDFEQEMATAASSSSLEKSYELPDGQVITIGNERFRCPEALFQPSFLGMESCGIHETTFNSIMKCDVDIRKDLYANTVLSGGTTMYPGIADRMQKEITALAPSTMKIKIIAPPERKYSVWIGGSILASLSTFQQMWISKQEYDESGPSIVHRKCF",
    # Thioredoxin (plant)
    "MASKVYLADFYAPWCGHCKMIAPDVVAEYEKDNSCPVVEFSENVKKAFEEGKVKGQTLKLFADANGTPASQKLVEFIKRNPEGGLITPPEGDDLKLKAGQKLREELAKEGVAQAFPASQLIEDAIKEFMDEQYLKQPTGDSAIKVPLIAFSNQPEVKYVTEDGKISRDDLVLVKKLIEQTDPQKVAIASVPNQDKAAGDALTQVMVDLEDLKK",
    # E. coli — expect adapted model to NOT improve on this
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQQKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL",
]
SEQ_LABELS = ["RuBisCO-large", "RuBisCO-small", "Actin", "Thioredoxin", "E.coli (non-plant)"]


def compute_mlm_loss(model, tokenizer, sequences, device, batch_size=4):
    model.eval()
    losses = []
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i : i + batch_size]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        enc["labels"] = enc["input_ids"].clone()
        with torch.no_grad():
            out = model(**enc)
        losses.append(out.loss.item())
    return sum(losses) / len(losses)


def fill_mask_demo(model, tokenizer, sequence, mask_positions, device):
    tokens = tokenizer(sequence, return_tensors="pt")
    input_ids = tokens["input_ids"].clone().to(device)
    original_tokens = [tokenizer.convert_ids_to_tokens([input_ids[0, p].item()])[0] for p in mask_positions]
    for pos in mask_positions:
        input_ids[0, pos] = tokenizer.mask_token_id
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    preds = []
    for pos, orig in zip(mask_positions, original_tokens):
        top5 = tokenizer.convert_ids_to_tokens(logits[0, pos].topk(5).indices.tolist())
        preds.append({"position": pos, "true_token": orig, "top5": top5})
    return preds


def get_embeddings(model, tokenizer, sequences, device):
    enc = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.esm(**enc)
    return out.last_hidden_state[:, 0, :]


def cosine(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main(cfg: dict, device_str: str = "auto") -> None:
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

    print("Loading tokenizer and models...")
    tokenizer = EsmTokenizer.from_pretrained(cfg["vanilla_hub"])
    vanilla: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(cfg["vanilla_hub"]).to(device)
    adapted: EsmForMaskedLM = EsmForMaskedLM.from_pretrained(cfg["adapted_path"]).to(device)

    results: dict = {"model": model_name, "date": str(date.today()), "tests": {}}

    # ── Test 1: MLM perplexity ─────────────────────────────────────────────
    print("\n=== Test 1: MLM loss (lower = better fit to sequence distribution) ===")
    plant_seqs = PLANT_SEQS[:-1]
    nonplant   = PLANT_SEQS[-1:]
    perplexity: dict = {}

    for label, seqs in [("plant", plant_seqs), ("non_plant_ecoli", nonplant)]:
        v_loss = compute_mlm_loss(vanilla, tokenizer, seqs, device)
        a_loss = compute_mlm_loss(adapted, tokenizer, seqs, device)
        delta  = v_loss - a_loss
        perplexity[label] = {
            "vanilla_loss": round(v_loss, 4),
            "adapted_loss": round(a_loss, 4),
            "vanilla_ppl":  round(math.exp(min(v_loss, 20)), 2),
            "adapted_ppl":  round(math.exp(min(a_loss, 20)), 2),
            "delta_loss":   round(delta, 4),
            "improved":     delta > 0,
        }
        tag = "✓ adapted better" if delta > 0 else "✗ no improvement"
        print(f"\n  {label}:")
        print(f"    Vanilla  loss={v_loss:.4f}  ppl={perplexity[label]['vanilla_ppl']:.2f}")
        print(f"    Adapted  loss={a_loss:.4f}  ppl={perplexity[label]['adapted_ppl']:.2f}")
        print(f"    Δ={delta:+.4f}  {tag}")

    results["tests"]["mlm_perplexity"] = perplexity

    # ── Test 2: Fill-mask ──────────────────────────────────────────────────
    print("\n=== Test 2: Fill-mask — masked positions in RuBisCO ===")
    seq  = PLANT_SEQS[0]
    toks = tokenizer(seq)["input_ids"]
    mid  = len(toks) // 2
    positions = [mid - 5, mid, mid + 5]

    fill_results = []
    for pred in fill_mask_demo(adapted, tokenizer, seq, positions, device):
        v_pred = fill_mask_demo(vanilla, tokenizer, seq, [pred["position"]], device)[0]
        entry = {
            "position":      pred["position"],
            "true_token":    pred["true_token"],
            "adapted_top5":  pred["top5"],
            "vanilla_top5":  v_pred["top5"],
        }
        fill_results.append(entry)
        print(f"\n  Position {pred['position']}  (true: {pred['true_token']})")
        print(f"    Adapted top-5: {pred['top5']}")
        print(f"    Vanilla top-5: {v_pred['top5']}")

    results["tests"]["fill_mask"] = fill_results

    # ── Test 3: Embedding similarity ───────────────────────────────────────
    print("\n=== Test 3: [CLS] embedding cosine similarities ===")
    emb_v = get_embeddings(vanilla, tokenizer, PLANT_SEQS, device)
    emb_a = get_embeddings(adapted, tokenizer, PLANT_SEQS, device)

    ecoli_idx = len(PLANT_SEQS) - 1
    sim_results = []

    print("\n  Plant vs. E. coli (adapted should be MORE distant):")
    for i, label in enumerate(SEQ_LABELS[:-1]):
        sim_v = cosine(emb_v[i], emb_v[ecoli_idx])
        sim_a = cosine(emb_a[i], emb_a[ecoli_idx])
        sim_results.append({"pair": f"{label}_vs_ecoli", "vanilla": round(sim_v, 3), "adapted": round(sim_a, 3)})
        print(f"    {label:20s} vs E.coli → vanilla: {sim_v:.3f}  adapted: {sim_a:.3f}")

    print("\n  Plant-to-plant cosines (should be higher than plant-ecoli):")
    for i in range(len(SEQ_LABELS) - 2):
        for j in range(i + 1, len(SEQ_LABELS) - 1):
            sim_a = cosine(emb_a[i], emb_a[j])
            sim_results.append({"pair": f"{SEQ_LABELS[i]}_vs_{SEQ_LABELS[j]}", "adapted": round(sim_a, 3)})
            print(f"    {SEQ_LABELS[i]:20s} vs {SEQ_LABELS[j]:20s} → {sim_a:.3f}")

    results["tests"]["embedding_similarity"] = sim_results

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = results_dir / f"{model_name}_sanity_{date.today()}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="esm2_adapt_test/configs/35m.yaml",
        help="Path to model config YAML (e.g. esm2_adapt_test/configs/35m.yaml)",
    )
    parser.add_argument("--device", default="auto", help="cuda / cpu / auto (default: auto)")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg, args.device)
