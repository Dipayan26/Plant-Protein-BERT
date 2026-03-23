"""fasta_to_hdf5.py — Convert a clean FASTA file to the HDF5 format used by StreamingProteinDataset.

Use this after running filter_viridiplantae.py to build the training index directly
from the already-filtered FASTA, without re-parsing the 18 GB DAT.gz.

The output HDF5 has a single dataset "sequences" of variable-length strings,
which is exactly what StreamingProteinDataset expects.

Usage:
    python scripts/fasta_to_hdf5.py \
        --fasta  outputs/processed/viridiplantae_clean.fasta \
        --output outputs/processed/trembl_full/sequences.h5 \
        --min-len 50 --max-len 1024 --max-ambiguous-frac 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

AMBIGUOUS_AA = set("BZOUXJ")


def iter_fasta(fasta_path: Path):
    """Yield (header, sequence) from a FASTA file."""
    header = seq_parts = None
    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            elif seq_parts is not None:
                seq_parts.append(line)
    if header is not None and seq_parts:
        yield header, "".join(seq_parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta",  required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--max-ambiguous-frac", type=float, default=0.1)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.fasta} ...", flush=True)
    sequences: list[str] = []
    skipped = 0

    for _, seq in tqdm(iter_fasta(args.fasta), desc="Filtering"):
        if not (args.min_len <= len(seq) <= args.max_len):
            skipped += 1
            continue
        ambig = sum(1 for aa in seq if aa in AMBIGUOUS_AA) / len(seq)
        if ambig > args.max_ambiguous_frac:
            skipped += 1
            continue
        sequences.append(seq)

    print(f"  Kept {len(sequences):,}  |  Skipped {skipped:,}")

    print(f"Writing HDF5 → {args.output} ...", flush=True)
    # h5py.string_dtype() returns str (not bytes) on read in h5py 3.x
    dt = h5py.string_dtype()
    with h5py.File(args.output, "w") as hf:
        ds = hf.create_dataset("sequences", shape=(len(sequences),), dtype=dt)
        ds[:] = np.array(sequences, dtype=object)

    print(f"Done. {len(sequences):,} sequences in {args.output}")


if __name__ == "__main__":
    main()
