"""Analyze organism distribution in UniRef50 using NCBI Taxonomy for exact classification.

Pipeline:
  1. Parse nodes.dmp → build TaxID→parent dict (covers all ~2.7M known TaxIDs)
  2. For each UniRef50 header, walk the taxonomy tree upward from TaxID
     until a landmark superkingdom node is reached (cached per TaxID)
  3. Count clusters and member sequences per superkingdom

Landmark TaxIDs (NCBI):
  2       Bacteria
  2157    Archaea
  10239   Viruses
  33090   Viridiplantae   ← plants
  33208   Metazoa         ← animals
  4751    Fungi
  2759    Eukaryota       ← catch-all for other eukaryotes (Sar, Amoebozoa, etc.)
  1       root            ← unclassified / environmental / synthetic

Usage:
    python scripts/analyze_uniref50_distribution.py
    python scripts/analyze_uniref50_distribution.py --taxdump outputs/taxonomy --output outputs/uniref50_distribution_v2.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Step 1 — Build taxonomy tree from nodes.dmp
# ---------------------------------------------------------------------------

# Superkingdom landmark TaxIDs — order matters for display only; tree walk is unambiguous
LANDMARKS: dict[int, str] = {
    2:     "Bacteria",
    2157:  "Archaea",
    10239: "Viruses",
    33090: "Plants (Viridiplantae)",
    33208: "Metazoa",
    4751:  "Fungi",
    2759:  "Other Eukaryota",   # Sar, Amoebozoa, Excavata, etc.
    1:     "Unclassified / Root",
}


def build_parent_map(nodes_dmp: Path) -> dict[int, int]:
    """Parse nodes.dmp into {taxid: parent_taxid}. Root (1) maps to itself."""
    parent: dict[int, int] = {}
    with nodes_dmp.open() as f:
        for line in f:
            parts = line.split("\t|\t")
            tax_id = int(parts[0])
            par_id = int(parts[1])
            parent[tax_id] = par_id
    return parent


def build_superkingdom_cache(parent: dict[int, int]) -> dict[int, str]:
    """Pre-resolve every known TaxID to its superkingdom in one pass.

    Walks each TaxID to a LANDMARK node, caching every node along the path
    so the next sibling TaxID resolves in O(1).
    """
    cache: dict[int, str] = {}

    # Seed cache with landmarks themselves
    for tid, label in LANDMARKS.items():
        cache[tid] = label

    def resolve(taxid: int) -> str:
        if taxid in cache:
            return cache[taxid]

        # Walk up, collecting path until we hit a cached node
        path: list[int] = []
        current = taxid
        while current not in cache:
            path.append(current)
            par = parent.get(current)
            if par is None or par == current:
                # Orphan or root — mark unclassified
                cache[current] = "Unclassified / Root"
                break
            current = par

        # Backfill the entire path with the resolved label
        label = cache[current]
        for node in path:
            cache[node] = label
        return label

    for taxid in parent:
        resolve(taxid)

    return cache


# ---------------------------------------------------------------------------
# Step 2 — Stream UniRef50 FASTA headers and classify
# ---------------------------------------------------------------------------

HEADER_RE = re.compile(
    r"^>UniRef50_\S+\s+.*?\s+n=(\d+)\s+Tax=(.+?)\s+TaxID=(\d+)\s+RepID="
)


def analyze(
    fasta_gz: Path,
    taxdump_dir: Path,
    output_path: Path,
) -> None:

    nodes_dmp = taxdump_dir / "nodes.dmp"
    if not nodes_dmp.exists():
        sys.exit(f"nodes.dmp not found at {nodes_dmp}. Run with --taxdump pointing to extracted taxdump dir.")

    # --- Build taxonomy lookup ---
    print("Building taxonomy tree from nodes.dmp ...", flush=True)
    parent = build_parent_map(nodes_dmp)
    print(f"  Loaded {len(parent):,} TaxIDs. Resolving superkingdoms ...", flush=True)
    superkingdom_cache = build_superkingdom_cache(parent)
    print(f"  Done. {len(superkingdom_cache):,} TaxIDs resolved.\n", flush=True)

    # --- Stream FASTA ---
    kingdom_clusters: Counter = Counter()
    kingdom_members: Counter = Counter()

    # Top organisms per kingdom
    plant_org_clusters: Counter = Counter()
    plant_org_members: Counter = Counter()
    top_global_clusters: Counter = Counter()

    # Unknown TaxIDs (not in NCBI dump — deleted/merged entries)
    unknown_taxids: Counter = Counter()

    total_clusters = 0
    total_members  = 0
    parse_errors   = 0

    print(f"Streaming {fasta_gz} ...", flush=True)

    with gzip.open(fasta_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
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

            if tax_id in superkingdom_cache:
                kingdom = superkingdom_cache[tax_id]
            else:
                kingdom = "Unclassified / Root"
                unknown_taxids[tax_id] += 1

            kingdom_clusters[kingdom] += 1
            kingdom_members[kingdom]  += n_members
            top_global_clusters[tax_name] += 1

            if kingdom == "Plants (Viridiplantae)":
                plant_org_clusters[tax_name] += 1
                plant_org_members[tax_name]  += n_members

            if total_clusters % 5_000_000 == 0:
                pct_c = 100 * kingdom_clusters["Plants (Viridiplantae)"] / total_clusters
                pct_m = 100 * kingdom_members["Plants (Viridiplantae)"]  / max(total_members, 1)
                print(
                    f"  {total_clusters:>10,} clusters | "
                    f"plants: {kingdom_clusters['Plants (Viridiplantae)']:,} ({pct_c:.2f}% clusters, {pct_m:.2f}% members)",
                    flush=True,
                )

    # --- Print results ---
    plant_c = kingdom_clusters["Plants (Viridiplantae)"]
    plant_m = kingdom_members["Plants (Viridiplantae)"]

    print("\n" + "=" * 72)
    print("UNIREF50 ORGANISM DISTRIBUTION  (NCBI Taxonomy-based)")
    print("=" * 72)
    print(f"\n  Total clusters         : {total_clusters:>12,}")
    print(f"  Total member sequences : {total_members:>12,}")
    print(f"  Parse errors (skipped) : {parse_errors:>12,}")
    print(f"  Unknown TaxIDs         : {len(unknown_taxids):>12,}  (not in current taxdump)")

    print("\n" + "-" * 72)
    print(f"  {'Superkingdom':<30s} {'Clusters':>12s} {'  %':>7s}   {'Members':>14s} {'  %':>7s}")
    print("-" * 72)
    for kingdom, count in kingdom_clusters.most_common():
        pct_c = 100 * count / total_clusters
        mem   = kingdom_members[kingdom]
        pct_m = 100 * mem / total_members
        print(f"  {kingdom:<30s} {count:>12,} {pct_c:>7.3f}%   {mem:>14,} {pct_m:>7.3f}%")
    print("-" * 72)

    print(f"\n{'=' * 72}")
    print(f"  PLANT SUMMARY")
    print(f"  Clusters : {plant_c:,} / {total_clusters:,}  →  {100*plant_c/total_clusters:.4f}%")
    print(f"  Members  : {plant_m:,} / {total_members:,}  →  {100*plant_m/total_members:.4f}%")
    print(f"{'=' * 72}")

    print(f"\n  Top 40 plant taxa by cluster count")
    print(f"  {'Clusters':>10s}  {'Members':>12s}  Taxon")
    print(f"  {'-'*10}  {'-'*12}  {'-'*40}")
    for org, cnt in plant_org_clusters.most_common(40):
        mem = plant_org_members[org]
        print(f"  {cnt:>10,}  {mem:>12,}  {org}")

    print(f"\n  Top 30 global taxa by cluster count")
    print(f"  {'Clusters':>10s}  Taxon")
    print(f"  {'-'*10}  {'-'*40}")
    for org, cnt in top_global_clusters.most_common(30):
        kingdom = superkingdom_cache.get(0, "?")  # just for display
        print(f"  {cnt:>10,}  {org}")

    # --- Save JSON ---
    results = {
        "total_clusters": total_clusters,
        "total_members": total_members,
        "parse_errors": parse_errors,
        "unknown_taxids_count": len(unknown_taxids),
        "superkingdom_clusters": dict(kingdom_clusters),
        "superkingdom_members": dict(kingdom_members),
        "plant_cluster_pct": round(100 * plant_c / total_clusters, 5),
        "plant_member_pct":  round(100 * plant_m / total_members, 5),
        "top_plant_taxa_by_clusters": dict(plant_org_clusters.most_common(100)),
        "top_plant_taxa_by_members":  dict(plant_org_members.most_common(100)),
        "top_global_taxa":            dict(top_global_clusters.most_common(200)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fasta",
        default="/home/dipayan/Documents/MOE_Bind/configs/dataset/RAW_data_download/uniref_50/uniref50.fasta.gz",
    )
    parser.add_argument("--taxdump", default="outputs/taxonomy")
    parser.add_argument("--output",  default="outputs/uniref50_distribution_v2.json")
    args = parser.parse_args()

    analyze(Path(args.fasta), Path(args.taxdump), Path(args.output))


if __name__ == "__main__":
    main()
