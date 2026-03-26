"""Compare plant species distribution between UniRef50 and TrEMBL plant proteins.

Panel A: Top plant species in UniRef50 (by member sequence count)
Panel B: Top plant species in TrEMBL plant dat.gz (by sequence count)

Usage:
    python scripts/plot_plant_species_distribution.py
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

matplotlib.rcParams.update({
    "font.family":      "sans-serif",
    "font.sans-serif":  ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "pdf.fonttype":     42,
    "axes.spines.top":  False,
    "axes.spines.right":False,
})

TREMBL_PATH = Path("DATA/PLANT_uniprot_new/uniprot_trembl_18_GB/uniprot_trembl_plants.dat.gz")
UNIREF_JSON = Path("outputs/uniref50_distribution_v2.json")
OUT_DIR     = Path("outputs/figures")
TOP_N       = 25

# ---------------------------------------------------------------------------
# Helper: is this a species-level name (not a broad clade like "Mesangiospermae")
# ---------------------------------------------------------------------------
BROAD_CLADE_RE = re.compile(
    r"^(Mesangiospermae|Pentapetalae|Magnoliopsida|Embryophyta|Spermatophyta|"
    r"Poaceae|rosids|Viridiplantae|Tracheophyta|Brassicaceae|asterids|"
    r"Liliopsida|eudicots|monocots|Streptophyta|Chlorophyta|PACMAD clade|"
    r"BOP clade|Fabaceae|Solanaceae|Malvaceae|Asteraceae|Lamiaceae|"
    r"Euphorbiaceae|Rutaceae|Myrtaceae|Pinaceae|Orchidaceae|Ranunculaceae|"
    r"core eudicots|campanulids|lamiids|COM clade|ANA grade)$",
    re.IGNORECASE,
)

def is_species(name: str) -> bool:
    if BROAD_CLADE_RE.match(name):
        return False
    parts = name.split()
    if len(parts) < 2:
        return False
    # Binomial: first word capitalized, second word lowercase
    return parts[0][0].isupper() and parts[1][0].islower()


# ---------------------------------------------------------------------------
# 1. UniRef50 plant species
# ---------------------------------------------------------------------------
print("Loading UniRef50 plant data ...", flush=True)
with open(UNIREF_JSON) as f:
    uniref_data = json.load(f)

uniref_species: list[tuple[str, int]] = [
    (name, count)
    for name, count in uniref_data["top_plant_taxa_by_members"].items()
    if is_species(name)
][:TOP_N]


# ---------------------------------------------------------------------------
# 2. TrEMBL plant species — stream OS lines
# ---------------------------------------------------------------------------
TREMBL_CACHE = Path("outputs/trembl_species_counts.json")

if TREMBL_CACHE.exists():
    print(f"Loading TrEMBL species counts from cache {TREMBL_CACHE} ...", flush=True)
    with open(TREMBL_CACHE) as f:
        os_counter = Counter(json.load(f))
else:
    print(f"Streaming {TREMBL_PATH} for OS (organism) counts ...", flush=True)
    os_counter: Counter = Counter()
    current_os = None
    with gzip.open(TREMBL_PATH, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "OS":
                organism = line[5:].strip().rstrip(".")
                current_os = organism
            elif tag == "//" and current_os:
                os_counter[current_os] += 1
                current_os = None
    TREMBL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(TREMBL_CACHE, "w") as f:
        json.dump(dict(os_counter), f)
    print(f"  Cached to {TREMBL_CACHE}")

trembl_total_all  = sum(os_counter.values())
trembl_unique_spp = len(os_counter)
print(f"  Found {trembl_unique_spp:,} unique organisms, {trembl_total_all:,} total sequences")

trembl_species: list[tuple[str, int]] = os_counter.most_common(TOP_N)

# Totals for panel annotations
# UniRef50: Viridiplantae member sequences (from audit: 24,220,200)
uniref_total   = uniref_data["superkingdom_members"]["Plants (Viridiplantae)"]
uniref_species_count = uniref_data.get("plant_species_count", None)  # may not exist

# TrEMBL: Viridiplantae only (from taxonomy audit: 21,478,452 of 25,481,103 total)
trembl_viridiplantae = 21_478_452   # confirmed by OX TaxID audit


# ---------------------------------------------------------------------------
# 3. Plot — two horizontal bar charts
# ---------------------------------------------------------------------------
def shorten(name: str, maxlen: int = 32) -> str:
    if len(name) <= maxlen:
        return name
    parts = name.split()
    return " ".join(parts[:2]) + " …"


fig, axes = plt.subplots(1, 2, figsize=(14, 9))
fig.suptitle(
    "Plant species representation: UniRef50 vs TrEMBL plant proteome",
    fontsize=13, fontweight="bold", y=1.01,
)

panels = [
    (axes[0], uniref_species, "UniRef50\nmember sequences per plant species",  "#3A7D44"),
    (axes[1], trembl_species, "TrEMBL plant proteins\nsequences per species", "#2E86AB"),
]

for ax, species_data, title, base_color in panels:
    names  = [shorten(n) for n, _ in species_data]
    counts = [c for _, c in species_data]

    norm   = plt.Normalize(min(counts), max(counts))
    colors = [matplotlib.colors.to_hex(plt.cm.RdYlGn(0.35 + 0.65 * norm(c))) for c in counts]

    y = np.arange(len(names))
    bars = ax.barh(y, counts, color=colors, edgecolor="white", linewidth=0.5, height=0.72)

    # Value labels at end of each bar
    for bar, count in zip(bars, counts):
        label = f"{count/1e6:.2f}M" if count >= 1_000_000 else f"{count/1e3:.1f}K"
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left",
            fontsize=7.5, color="#222222", fontweight="bold",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of sequences")
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=8)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}K"
    ))
    ax.set_xlim(0, max(counts) * 1.22)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)


fig.tight_layout()
OUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_DIR / "plant_species_distribution.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "plant_species_distribution.png", bbox_inches="tight", dpi=300)
print(f"\nSaved:")
print(f"  {OUT_DIR / 'plant_species_distribution.pdf'}")
print(f"  {OUT_DIR / 'plant_species_distribution.png'}")
plt.show()
