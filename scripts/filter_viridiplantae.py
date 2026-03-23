"""filter_viridiplantae.py — Extract clean Viridiplantae sequences from TrEMBL.

Reads the raw TrEMBL plant DAT.gz file, applies the NCBI Taxonomy-based
Viridiplantae filter (removing oomycetes, dinoflagellates, cryptophytes, etc.),
and writes a clean FASTA file ready for training or inspection.

Background (outputs/figures/details.md):
  - Raw TrEMBL plant file:       25,481,103 sequences
  - After Viridiplantae filter:  21,478,452 sequences (84.3%)
  - Removed (15.7%):             oomycetes, dinoflagellates, haptophytes

Usage:
    python scripts/filter_viridiplantae.py \
        --dat-gz  DATA/PLANT_uniprot_new/uniprot_trembl_18_GB/uniprot_trembl_plants.dat.gz \
        --nodes   outputs/taxonomy/nodes.dmp \
        --output  outputs/processed/viridiplantae_clean.fasta \
        --min-len 50 --max-len 1024

    # Check stats only (no output file)
    python scripts/filter_viridiplantae.py --dat-gz <path> --nodes <path> --stats-only
"""

from __future__ import annotations

import argparse
import gzip
import sys
from collections import Counter
from pathlib import Path


# ── Taxonomy helpers ──────────────────────────────────────────────────────────

VIRIDIPLANTAE_TAXID = 33090   # NCBI TaxID for Viridiplantae (green plants)


def load_taxonomy_parents(nodes_dmp: Path) -> dict[int, int]:
    """Parse NCBI nodes.dmp → {taxid: parent_taxid}."""
    parents: dict[int, int] = {}
    with open(nodes_dmp) as f:
        for line in f:
            parts = line.split("|")
            taxid = int(parts[0].strip())
            parent = int(parts[1].strip())
            parents[taxid] = parent
    return parents


def is_viridiplantae(taxid: int, parents: dict[int, int], cache: dict[int, bool]) -> bool:
    """Walk the taxonomy tree upward; return True if Viridiplantae is an ancestor."""
    if taxid in cache:
        return cache[taxid]

    visited: list[int] = []
    current = taxid
    while current not in cache:
        visited.append(current)
        if current == VIRIDIPLANTAE_TAXID:
            result = True
            break
        parent = parents.get(current)
        if parent is None or parent == current:  # root (taxid=1, parent=1)
            result = False
            break
        current = parent
    else:
        result = cache[current]

    for tid in visited:
        cache[tid] = result
    return result


# ── UniProt DAT.gz parser ────────────────────────────────────────────────────

def iter_records(dat_gz: Path):
    """Yield (accession, taxid, sequence) from a UniProt DAT.gz file."""
    accession = taxid = sequence_lines = None
    in_sequence = False

    opener = gzip.open if str(dat_gz).endswith(".gz") else open
    with opener(dat_gz, "rt", errors="replace") as f:
        for line in f:
            tag = line[:2]
            if tag == "AC" and accession is None:
                accession = line[5:].strip().rstrip(";").split(";")[0].strip()
            elif tag == "OX":
                # OX   NCBI_TaxID=3702; or OX   NCBI_TaxID=3702 {ECO:...};
                part = line[5:].strip()
                for token in part.split():
                    if token.startswith("NCBI_TaxID="):
                        raw = token.split("=")[1].rstrip(";").split("{")[0]
                        try:
                            taxid = int(raw)
                        except ValueError:
                            pass
                        break
            elif tag == "  " and in_sequence:
                sequence_lines.append(line.strip().replace(" ", ""))
            elif tag == "SQ":
                in_sequence = True
                sequence_lines = []
            elif line.startswith("//"):
                if accession and taxid and sequence_lines:
                    yield accession, taxid, "".join(sequence_lines)
                accession = taxid = None
                in_sequence = False
                sequence_lines = None


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Filter TrEMBL to Viridiplantae only.")
    parser.add_argument("--dat-gz", required=True, type=Path)
    parser.add_argument("--nodes", required=True, type=Path,
                        help="Path to NCBI nodes.dmp (from taxdump.tar.gz)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output FASTA path (omit for --stats-only)")
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--stats-only", action="store_true",
                        help="Print statistics without writing output")
    args = parser.parse_args()

    if not args.stats_only and args.output is None:
        print("ERROR: specify --output or --stats-only", file=sys.stderr)
        sys.exit(1)

    print(f"Loading taxonomy from {args.nodes} ...", flush=True)
    parents = load_taxonomy_parents(args.nodes)
    cache: dict[int, bool] = {1: False}   # root is not Viridiplantae
    print(f"  {len(parents):,} taxon nodes loaded.")

    out_fh = None
    if not args.stats_only and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_fh = open(args.output, "w")

    total = kept = removed_taxonomy = removed_length = 0
    removed_taxa: Counter = Counter()

    print(f"\nStreaming {args.dat_gz} ...", flush=True)
    try:
        for i, (acc, taxid, seq) in enumerate(iter_records(args.dat_gz)):
            total += 1
            if total % 500_000 == 0:
                pct = kept / total * 100
                print(f"  {total:>10,}  kept={kept:,} ({pct:.1f}%)", flush=True)

            # Length filter
            if not (args.min_len <= len(seq) <= args.max_len):
                removed_length += 1
                continue

            # Taxonomy filter
            if not is_viridiplantae(taxid, parents, cache):
                removed_taxonomy += 1
                removed_taxa[taxid] += 1
                continue

            kept += 1
            if out_fh:
                out_fh.write(f">{acc}\n{seq}\n")

    finally:
        if out_fh:
            out_fh.close()

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"  Total records parsed:            {total:>12,}")
    print(f"  Kept (Viridiplantae, len OK):     {kept:>12,}  ({kept/total*100:.1f}%)")
    print(f"  Removed — non-Viridiplantae:      {removed_taxonomy:>12,}  ({removed_taxonomy/total*100:.1f}%)")
    print(f"  Removed — length filter:          {removed_length:>12,}  ({removed_length/total*100:.1f}%)")
    print(f"{'─'*55}")

    if removed_taxa:
        print(f"\nTop 10 non-Viridiplantae TaxIDs removed:")
        for taxid, count in removed_taxa.most_common(10):
            print(f"  TaxID {taxid:>8}  →  {count:>8,} sequences")

    if out_fh and args.output:
        print(f"\nOutput written to: {args.output}")


if __name__ == "__main__":
    main()
