"""PyTorch Dataset classes for protein sequences.

During training, PyTorch loads sequences in batches.  Each Dataset class
implements __len__ and __getitem__ so PyTorch can index into it randomly.

Two variants:
  InMemoryProteinDataset  — loads the full filtered JSONL into RAM at startup.
                            Good for small datasets like SwissProt (~57 MB).
  StreamingProteinDataset — reads sequences on-demand from an HDF5 file.
                            Required for TrEMBL (~18 GB, cannot fit in RAM).

Why padding=False and return_tensors=None in __getitem__?
  Each protein has a different length.  If we pad every sequence to max_length
  here, we waste GPU memory on padding tokens.  Instead, we return raw Python
  lists and let the DataCollator in the DataLoader batch sequences together,
  padding them only to the longest sequence in that batch (dynamic padding).
  This is significantly more memory-efficient.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
from torch.utils.data import Dataset


class InMemoryProteinDataset(Dataset):
    """Loads all sequences from a JSONL file into memory at startup.

    Use for small datasets where loading time and RAM usage are acceptable.
    The JSONL format is one JSON object per line: {"sequence": "ACDEF...", ...}
    """

    def __init__(self, jsonl_path: str | Path, tokenizer: Any, max_length: int = 1024) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: list[str] = []
        with Path(jsonl_path).open() as f:
            for line in f:
                self.records.append(json.loads(line)["sequence"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        # Return token IDs as Python lists (not tensors) — the DataCollator
        # will pad and stack them into tensors batch-by-batch.
        return self.tokenizer(
            self.records[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )


class StreamingProteinDataset(Dataset):
    """Random-access dataset backed by an HDF5 file.

    HDF5 stores sequences as an indexed array on disk.  We can read any
    sequence directly by its index (O(1) seek), without loading the whole file.

    Each DataLoader worker gets its own file handle (_hf), opened lazily on the
    first __getitem__ call in that worker process.  This avoids issues with
    file handles shared across processes (h5py is not fork-safe).

    The HDF5 file is built by:  python scripts/fasta_to_hdf5.py
    """

    def __init__(self, hdf5_path: str | Path, tokenizer: Any, max_length: int = 1024) -> None:
        self.hdf5_path = str(hdf5_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Open once just to read the total number of sequences, then close.
        with h5py.File(self.hdf5_path, "r") as hf:
            self._length = len(hf["sequences"])
        self._hf: h5py.File | None = None   # each worker opens its own handle

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        # Lazy open: first access in each worker process opens the file.
        if self._hf is None:
            self._hf = h5py.File(self.hdf5_path, "r")
        sequence = self._hf["sequences"][idx]
        # h5py 3.x returns bytes for variable-length strings; decode to str.
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        return self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
