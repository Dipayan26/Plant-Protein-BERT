"""PyTorch Dataset classes for protein sequences.

InMemoryProteinDataset  — SwissProt (~57 MB, fits in RAM).
StreamingProteinDataset — TrEMBL (~18 GB, O(1) random access via HDF5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import torch
from torch.utils.data import Dataset


class InMemoryProteinDataset(Dataset):
    """Loads the full filtered JSONL into a list at construction."""

    def __init__(self, jsonl_path: str | Path, tokenizer: Any, max_length: int = 1024) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.records: list[str] = []
        with Path(jsonl_path).open() as f:
            for line in f:
                self.records.append(json.loads(line)["sequence"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            self.records[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )


class StreamingProteinDataset(Dataset):
    """Random-access dataset backed by an HDF5 file.

    The HDF5 file is built once by preprocessing.build_hdf5_index().
    Each worker opens its own file handle to avoid concurrency issues.
    """

    def __init__(self, hdf5_path: str | Path, tokenizer: Any, max_length: int = 1024) -> None:
        self.hdf5_path = str(hdf5_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Open once to read length; workers re-open in __getitem__
        with h5py.File(self.hdf5_path, "r") as hf:
            self._length = len(hf["sequences"])
        self._hf: h5py.File | None = None

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self._hf is None:
            # Lazy open per worker process
            self._hf = h5py.File(self.hdf5_path, "r")
        sequence = self._hf["sequences"][idx]
        if isinstance(sequence, bytes):
            sequence = sequence.decode("utf-8")
        return self.tokenizer(
            sequence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
