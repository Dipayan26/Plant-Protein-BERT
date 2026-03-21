"""One-time preprocessing pipeline: raw DAT.gz → filtered JSONL → HDF5 index.

Run via: python scripts/parse_uniprot.py data=trembl_full
Outputs are cached in outputs/processed/; set force_reprocess=true to re-run.

Taxonomy filtering: sequences whose OX TaxID does not resolve to Viridiplantae
(NCBI TaxID 33090) are rejected. This removes the ~15.7% of oomycete,
dinoflagellate, and other non-plant sequences present in UniProt's "plant" subset.
Requires outputs/taxonomy/nodes.dmp (downloaded once by scripts/parse_uniprot.py).
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

# NCBI TaxID for Viridiplantae — the only clade we want
VIRIDIPLANTAE_TAXID = 33090


# ---------------------------------------------------------------------------
# Taxonomy filter — built once, reused across all records
# ---------------------------------------------------------------------------

def _build_viridiplantae_resolver(nodes_dmp: Path):
    """Return a function is_viridiplantae(taxid) -> bool using NCBI nodes.dmp.

    Walks the parent chain from any TaxID upward; caches every node along
    the path so sibling lookups are O(1).
    """
    log.info(f"Loading NCBI taxonomy from {nodes_dmp} ...")
    parent: dict[int, int] = {}
    with nodes_dmp.open() as f:
        for line in f:
            parts = line.split("\t|\t")
            parent[int(parts[0])] = int(parts[1])

    # Cache: True = is Viridiplantae, False = is not, None = unseen
    cache: dict[int, bool] = {VIRIDIPLANTAE_TAXID: True, 1: False}

    def is_viridiplantae(taxid: int) -> bool:
        if taxid in cache:
            return cache[taxid]
        path: list[int] = []
        current = taxid
        while current not in cache:
            path.append(current)
            par = parent.get(current)
            if par is None or par == current:
                cache[current] = False
                break
            current = par
        result = cache[current]
        for node in path:
            cache[node] = result
        return result

    log.info(f"Taxonomy loaded ({len(parent):,} TaxIDs).")
    return is_viridiplantae


def _is_valid(record: UniProtRecord, cfg: DictConfig) -> bool:
    seq = record.sequence
    if not (cfg.min_length <= len(seq) <= cfg.max_length):
        return False
    ambiguous_frac = sum(1 for aa in seq if aa in AMBIGUOUS_AA) / len(seq)
    if ambiguous_frac > cfg.max_ambiguous_frac:
        return False
    return True


def stream_parse_to_jsonl(
    raw_files: list[dict],
    output_path: Path,
    is_viridiplantae=None,
) -> tuple[int, int]:
    """Parse all raw DAT.gz files and write Viridiplantae records as JSONL.

    Args:
        raw_files: list of {path, source} dicts from Hydra config.
        output_path: destination JSONL file.
        is_viridiplantae: callable(taxid) -> bool. If None, no taxonomy
            filtering is applied (not recommended for UniProt plant subsets).

    Returns:
        (written, rejected_taxonomy) counts.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = rejected = 0

    with output_path.open("w") as out:
        for file_cfg in raw_files:
            log.info(f"Parsing {file_cfg['path']} (source={file_cfg['source']})")
            for record in tqdm(parse_dat_gz(file_cfg["path"], source=file_cfg["source"])):
                # Taxonomy gate — reject non-Viridiplantae (oomycetes, dinoflagellates, etc.)
                if is_viridiplantae is not None and record.tax_id:
                    if not is_viridiplantae(record.tax_id):
                        rejected += 1
                        continue

                out.write(json.dumps({
                    "entry_name": record.entry_name,
                    "accession":  record.accessions[0] if record.accessions else "",
                    "tax_id":     record.tax_id,
                    "organism":   record.organism,
                    "sequence":   record.sequence,
                    "go_terms":   record.go_terms,
                    "source":     record.source,
                }) + "\n")
                written += 1

    log.info(f"Wrote {written:,} Viridiplantae sequences (rejected {rejected:,} non-plant by TaxID)")
    return written, rejected


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
    """Full pipeline: parse → taxonomy filter → length/dedup filter → HDF5 index."""
    processed_dir = Path(cfg.processed_dir)
    raw_jsonl      = processed_dir / "raw.jsonl"
    filtered_jsonl = processed_dir / "filtered.jsonl"
    hdf5_path      = processed_dir / "sequences.h5"

    if hdf5_path.exists() and not cfg.get("force_reprocess", False):
        log.info(f"Found existing {hdf5_path}. Set force_reprocess=true to re-run.")
        return

    # Build Viridiplantae resolver from NCBI taxdump
    nodes_dmp = Path(cfg.get("taxonomy_nodes_dmp", "outputs/taxonomy/nodes.dmp"))
    if not nodes_dmp.exists():
        log.warning(
            f"nodes.dmp not found at {nodes_dmp}. Taxonomy filtering DISABLED — "
            f"non-plant sequences (oomycetes, dinoflagellates) will NOT be removed. "
            f"Download taxdump.tar.gz from NCBI and set taxonomy_nodes_dmp in config."
        )
        is_viridiplantae = None
    else:
        is_viridiplantae = _build_viridiplantae_resolver(nodes_dmp)

    stream_parse_to_jsonl(cfg.raw_files, raw_jsonl, is_viridiplantae)
    filter_and_deduplicate(raw_jsonl, filtered_jsonl, cfg)
    build_hdf5_index(filtered_jsonl, hdf5_path)
