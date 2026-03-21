"""Analyze organism distribution in UniRef50 to quantify plant protein representation.

Streams through the gzipped FASTA reading only header lines — no sequence data loaded.
Each header encodes: cluster size (n=), taxon name (Tax=), and TaxID (TaxID=).

Two metrics computed:
  - Cluster count: number of UniRef50 representative sequences
  - Member count: sum of n= (total sequences across all clusters, i.e. what ESM2 sees)

Usage:
    python scripts/analyze_uniref50_distribution.py \
        --fasta /home/dipayan/Documents/MOE_Bind/configs/dataset/RAW_data_download/uniref_50/uniref50.fasta.gz \
        --output outputs/uniref50_distribution.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Plant lineage identification
# Viridiplantae (TaxID 33090) — green plants. We identify plant entries using:
#   1. A broad keyword set covering taxonomic group names in Tax= strings
#   2. Common plant-associated genus/species substrings
# This is conservative: counts only what is clearly plant-derived.
# ---------------------------------------------------------------------------
PLANT_TAX_KEYWORDS = {
    # Taxonomic clade names (appear in Tax= for heterogeneous clusters)
    "viridiplantae", "streptophyta", "embryophyta", "tracheophyta",
    "spermatophyta", "angiospermae", "magnoliopsida", "liliopsida",
    "bryophyta", "marchantiophyta", "anthocerotophyta",
    "chlorophyta", "charophyta", "chlorokybophyceae",
    "gymnosperms", "gymnospermae",
    # Major crop/model plant genera
    "arabidopsis", "oryza", "zea mays", "triticum", "hordeum",
    "glycine max", "solanum", "nicotiana", "vitis", "populus",
    "brassica", "gossypium", "sorghum", "manihot", "phaseolus",
    "medicago", "lotus japonicus", "setaria", "panicum",
    "musa", "coffea", "theobroma", "camellia sinensis",
    "helianthus", "beta vulgaris", "spinacia", "lactuca",
    "capsicum", "ipomoea", "cucumis", "cucurbita",
    "pisum sativum", "cicer arietinum", "linum", "cannabis",
    "eucalyptus", "pinus", "picea", "abies", "cupressus",
    "marchantia", "physcomitrella", "selaginella",
    "chlamydomonas", "volvox", "chlorella", "coccomyxa",
    "cyanidioschyzon", "galdieria",  # red algae (not plants but often grouped)
}

# TaxID ranges / exact IDs for Viridiplantae
# Exact known plant TaxIDs (non-exhaustive; used for fast path)
KNOWN_PLANT_TAXIDS = {
    33090,   # Viridiplantae
    35493,   # Streptophyta
    3398,    # Magnoliopsida (flowering plants)
    3193,    # Embryophyta
    131221,  # Streptophytina
    3197,    # Tracheophyta
    58024,   # Spermatophyta
    3218,    # Bryophyta
    3208,    # Marchantiophyta
    3694,    # Populus
    3702,    # Arabidopsis thaliana
    4530,    # Oryza sativa
    4577,    # Zea mays
    4565,    # Triticum aestivum
    3847,    # Glycine max
    4081,    # Solanum lycopersicum
    29760,   # Vitis vinifera
    4113,    # Solanum tuberosum
    4097,    # Nicotiana tabacum
    112509,  # Hordeum vulgare
    4558,    # Sorghum bicolor
    3760,    # Malus domestica
    4641,    # Musa acuminata
    13333,   # Brassica napus
    3708,    # Brassica oleracea
    3711,    # Brassica rapa
    3888,    # Pisum sativum
    3827,    # Cicer arietinum
    3917,    # Vigna radiata
    3987,    # Linum usitatissimum
    3483,    # Cannabis sativa
    4236,    # Lactuca sativa
    4072,    # Capsicum annuum
    4432,    # Ipomoea batatas
    3659,    # Cucumis sativus
    3635,    # Gossypium hirsutum
    4232,    # Helianthus annuus
    161934,  # Beta vulgaris
    3562,    # Spinacia oleracea
    39947,   # Oryza sativa Japonica
    4555,    # Saccharum officinarum
    4232,    # Helianthus annuus
    35883,   # Medicago truncatula
}

# Broad superkingdom buckets for the full distribution table
SUPERKINGDOM_KEYWORDS = {
    "bacteria":   ["bacteria", "bacterium", "bacillus", "clostridium", "streptomyces",
                   "mycobacterium", "nocardia", "pseudomonas", "escherichia", "salmonella",
                   "staphylococcus", "lactobacillus", "bifidobacterium", "helicobacter",
                   "campylobacter", "listeria", "vibrio", "enterococcus", "klebsiella",
                   "acinetobacter", "bacteroidetes", "proteobacteria", "firmicutes",
                   "actinobacteria", "cyanobacteria", "spirochaetes"],
    "archaea":    ["archaea", "archaeon", "halobacterium", "methanobacterium",
                   "sulfolobus", "thermococcus", "pyrococcus", "crenarchaeota",
                   "euryarchaeota", "thaumarchaeota"],
    "fungi":      ["fungi", "fungus", "saccharomyces", "aspergillus", "candida",
                   "neurospora", "schizosaccharomyces", "cryptococcus", "penicillium",
                   "trichoderma", "fusarium", "ustilago", "puccinia"],
    "metazoa":    ["homo sapiens", "mus musculus", "rattus", "drosophila", "caenorhabditis",
                   "danio rerio", "vertebrata", "mammalia", "chordata", "arthropoda",
                   "nematoda", "insecta", "amphibia", "reptilia", "aves", "fish",
                   "teleost", "actinopterygii"],
    "virus":      ["virus", "viral", "phage", "bacteriophage", "retrovirus",
                   "influenza", "coronavirus", "hiv", "herpes"],
    "unclassified": ["unclassified", "environmental sample", "metagenom",
                     "uncultured", "synthetic construct", "artificial"],
}


HEADER_RE = re.compile(
    r"^>UniRef50_\S+\s+.*?\s+n=(\d+)\s+Tax=(.+?)\s+TaxID=(\d+)\s+RepID=",
)


def is_plant(tax_name: str, tax_id: int) -> bool:
    tax_lower = tax_name.lower()
    if tax_id in KNOWN_PLANT_TAXIDS:
        return True
    return any(kw in tax_lower for kw in PLANT_TAX_KEYWORDS)


def classify_superkingdom(tax_name: str) -> str:
    tax_lower = tax_name.lower()
    # Plant check first
    if any(kw in tax_lower for kw in PLANT_TAX_KEYWORDS):
        return "plants"
    for kingdom, keywords in SUPERKINGDOM_KEYWORDS.items():
        if any(kw in tax_lower for kw in keywords):
            return kingdom
    return "other_eukaryota"


def analyze(fasta_gz: Path, output_path: Path | None = None) -> None:
    # Cluster counts and member sequence counts per kingdom
    kingdom_clusters: Counter = Counter()
    kingdom_members: Counter = Counter()

    # Top organisms within plants
    plant_organism_clusters: Counter = Counter()
    plant_organism_members: Counter = Counter()

    # Top organisms globally
    global_organism_clusters: Counter = Counter()

    total_clusters = 0
    total_members = 0
    parse_errors = 0

    print(f"Streaming {fasta_gz} ...", flush=True)

    with gzip.open(fasta_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if not line.startswith(">"):
                continue

            m = HEADER_RE.match(line.rstrip())
            if not m:
                parse_errors += 1
                continue

            n_members = int(m.group(1))
            tax_name  = m.group(2).strip()
            tax_id    = int(m.group(3))

            total_clusters += 1
            total_members  += n_members

            kingdom = classify_superkingdom(tax_name)
            if is_plant(tax_name, tax_id):
                kingdom = "plants"  # override in case classify missed it

            kingdom_clusters[kingdom] += 1
            kingdom_members[kingdom]  += n_members
            global_organism_clusters[tax_name] += 1

            if kingdom == "plants":
                plant_organism_clusters[tax_name] += 1
                plant_organism_members[tax_name]  += n_members

            if total_clusters % 1_000_000 == 0:
                pct_plant_c = 100 * kingdom_clusters["plants"] / total_clusters
                pct_plant_m = 100 * kingdom_members["plants"]  / max(total_members, 1)
                print(
                    f"  {total_clusters:>10,} clusters processed | "
                    f"plants: {kingdom_clusters['plants']:,} clusters ({pct_plant_c:.2f}%) | "
                    f"members: {pct_plant_m:.2f}%",
                    flush=True,
                )

    # -----------------------------------------------------------------------
    print("\n" + "="*70)
    print(f"UNIREF50 ORGANISM DISTRIBUTION ANALYSIS")
    print("="*70)
    print(f"\nTotal UniRef50 clusters : {total_clusters:>12,}")
    print(f"Total member sequences  : {total_members:>12,}")
    print(f"Parse errors (skipped)  : {parse_errors:>12,}")

    print("\n--- By superkingdom (% of CLUSTERS) ---")
    for kingdom, count in kingdom_clusters.most_common():
        pct = 100 * count / total_clusters
        mem = kingdom_members[kingdom]
        mem_pct = 100 * mem / total_members
        print(f"  {kingdom:<20s}  {count:>10,} clusters ({pct:6.2f}%)  |  "
              f"{mem:>14,} members ({mem_pct:6.2f}%)")

    plant_c = kingdom_clusters["plants"]
    plant_m = kingdom_members["plants"]
    print(f"\n{'='*70}")
    print(f"PLANTS SUMMARY")
    print(f"  Clusters  : {plant_c:,} / {total_clusters:,}  →  {100*plant_c/total_clusters:.3f}%")
    print(f"  Members   : {plant_m:,} / {total_members:,}  →  {100*plant_m/total_members:.3f}%")
    print(f"{'='*70}")

    print(f"\n--- Top 30 plant organisms by cluster count ---")
    for org, cnt in plant_organism_clusters.most_common(30):
        mem = plant_organism_members[org]
        print(f"  {cnt:>8,} clusters  {mem:>12,} members   {org}")

    print(f"\n--- Top 20 global organisms by cluster count ---")
    for org, cnt in global_organism_clusters.most_common(20):
        print(f"  {cnt:>8,}   {org}")

    # Save JSON results
    results = {
        "total_clusters": total_clusters,
        "total_members": total_members,
        "parse_errors": parse_errors,
        "kingdom_clusters": dict(kingdom_clusters),
        "kingdom_members": dict(kingdom_members),
        "plant_cluster_pct": round(100 * plant_c / total_clusters, 4),
        "plant_member_pct": round(100 * plant_m / total_members, 4),
        "top_plant_organisms_by_clusters": dict(plant_organism_clusters.most_common(100)),
        "top_plant_organisms_by_members": dict(plant_organism_members.most_common(100)),
        "top_global_organisms": dict(global_organism_clusters.most_common(200)),
    }
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze UniRef50 organism distribution")
    parser.add_argument(
        "--fasta",
        default="/home/dipayan/Documents/MOE_Bind/configs/dataset/RAW_data_download/uniref_50/uniref50.fasta.gz",
        help="Path to uniref50.fasta.gz",
    )
    parser.add_argument(
        "--output",
        default="outputs/uniref50_distribution.json",
        help="Path to save JSON results",
    )
    args = parser.parse_args()
    analyze(Path(args.fasta), Path(args.output))


if __name__ == "__main__":
    main()
