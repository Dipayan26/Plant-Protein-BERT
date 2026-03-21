"""One-time preprocessing pipeline: raw DAT.gz → filtered JSONL → HDF5 index.

Run via: python scripts/parse_uniprot.py data=sprot
Outputs are cached in outputs/processed/; set force_reprocess=true to re-run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import h5py
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from .uniprot_parser import UniProtRecord, parse_dat_gz

log = logging.getLogger(__name__)

AMBIGUOUS_AA = set("BZOUXJ")


def _is_valid(record: UniProtRecord, cfg: DictConfig) -> bool:
    seq = record.sequence
    if not (cfg.min_length <= len(seq) <= cfg.max_length):
        return False
    ambiguous_frac = sum(1 for aa in seq if aa in AMBIGUOUS_AA) / len(seq)
    if ambiguous_frac > cfg.max_ambiguous_frac:
        return False
    return True


def stream_parse_to_jsonl(raw_files: list[dict], output_path: Path) -> int:
    """Parse all raw DAT.gz files and write filtered records as JSONL.

    Returns:
        Total number of sequences written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as out:
        for file_cfg in raw_files:
            log.info(f"Parsing {file_cfg['path']} (source={file_cfg['source']})")
            for record in tqdm(parse_dat_gz(file_cfg["path"], source=file_cfg["source"])):
                out.write(json.dumps({
                    "entry_name": record.entry_name,
                    "accession": record.accessions[0] if record.accessions else "",
                    "organism": record.organism,
                    "sequence": record.sequence,
                    "go_terms": record.go_terms,
                    "source": record.source,
                }) + "\n")
                count += 1
    log.info(f"Wrote {count:,} sequences to {output_path}")
    return count


def filter_and_deduplicate(input_path: Path, output_path: Path, cfg: DictConfig) -> int:
    """Apply length/ambiguity filters and remove exact sequence duplicates."""
    seen: set[str] = set()
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as inp, output_path.open("w") as out:
        for line in tqdm(inp, desc="Filtering"):
            record = json.loads(line)
            seq = record["sequence"]
            if not (cfg.min_length <= len(seq) <= cfg.max_length):
                continue
            ambiguous_frac = sum(1 for aa in seq if aa in AMBIGUOUS_AA) / len(seq)
            if ambiguous_frac > cfg.max_ambiguous_frac:
                continue
            if seq in seen:
                continue
            seen.add(seq)
            out.write(line)
            count += 1
    log.info(f"After filter/dedup: {count:,} sequences")
    return count


def build_hdf5_index(jsonl_path: Path, hdf5_path: Path) -> None:
    """Convert JSONL to HDF5 for O(1) random-access during streaming training.

    Stores variable-length byte sequences using HDF5 special dtype.
    """
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    sequences = []
    with jsonl_path.open() as f:
        for line in tqdm(f, desc="Building HDF5"):
            sequences.append(json.loads(line)["sequence"])

    dt = h5py.special_dtype(vlen=str)
    with h5py.File(hdf5_path, "w") as hf:
        ds = hf.create_dataset("sequences", shape=(len(sequences),), dtype=dt)
        ds[:] = np.array(sequences, dtype=object)
    log.info(f"HDF5 index with {len(sequences):,} sequences → {hdf5_path}")


def run_preprocessing_pipeline(cfg: DictConfig) -> None:
    """Full pipeline: parse → filter/dedup → HDF5 index."""
    processed_dir = Path(cfg.processed_dir)
    raw_jsonl = processed_dir / "raw.jsonl"
    filtered_jsonl = processed_dir / "filtered.jsonl"
    hdf5_path = processed_dir / "sequences.h5"

    if hdf5_path.exists() and not cfg.get("force_reprocess", False):
        log.info(f"Found existing {hdf5_path}, skipping preprocessing. Set force_reprocess=true to re-run.")
        return

    stream_parse_to_jsonl(cfg.raw_files, raw_jsonl)
    filter_and_deduplicate(raw_jsonl, filtered_jsonl, cfg)
    build_hdf5_index(filtered_jsonl, hdf5_path)
