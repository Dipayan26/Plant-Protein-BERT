"""PyTorch Lightning DataModule — manages the full data loading pipeline.

A LightningDataModule is a self-contained unit that handles:
  1. setup()         — opens the dataset and splits it into train/val/test
  2. *_dataloader()  — returns a DataLoader for each split

The DataLoader feeds batches to the model during training.  Key parameters:
  batch_size      — how many sequences per batch (larger = faster but needs more GPU RAM)
  num_workers     — how many CPU processes to load data in parallel (0 = main process)
  collate_fn      — function that converts a list of samples into a padded batch tensor.
                    Set to DataCollatorForLanguageModeling in pretrain.py, which also
                    applies MLM masking (randomly replacing tokens with [MASK]).
  persistent_workers — keeps worker processes alive between batches (faster on disk I/O)
  pin_memory      — pre-pins CPU memory for faster GPU transfer
"""

from __future__ import annotations

from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split

from .dataset import InMemoryProteinDataset, StreamingProteinDataset


class UniProtDataModule(pl.LightningDataModule):
    """DataModule for UniProt plant protein sequences.

    Supports two modes:
      streaming=True  — reads from HDF5 (required for 18 GB TrEMBL, O(1) access)
      streaming=False — loads filtered JSONL fully into RAM (only for small datasets)
    """

    def __init__(
        self,
        tokenizer: object,
        processed_dir: str,
        streaming: bool = True,
        max_length: int = 1024,
        train_split: float = 0.9,
        val_split: float = 0.05,
        test_split: float = 0.05,
        batch_size: int = 64,
        num_workers: int = 8,
        shuffle_seed: int = 42,
        collate_fn=None,
        prefetch_factor: int = 4,
        **kwargs,  # absorb unused config fields (raw_files, taxonomy_nodes_dmp, etc.)
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.processed_dir = Path(processed_dir)
        self.streaming = streaming
        self.max_length = max_length
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_seed = shuffle_seed
        self.collate_fn = collate_fn
        # prefetch_factor requires num_workers > 0 (otherwise ignored)
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None

    def setup(self, stage: str | None = None) -> None:
        """Open dataset and create reproducible train/val/test splits."""
        hdf5_path = self.processed_dir / "sequences.h5"
        jsonl_path = self.processed_dir / "filtered.jsonl"

        if self.streaming:
            full_dataset = StreamingProteinDataset(hdf5_path, self.tokenizer, self.max_length)
        else:
            full_dataset = InMemoryProteinDataset(jsonl_path, self.tokenizer, self.max_length)

        n = len(full_dataset)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)
        n_test = n - n_train - n_val

        # random_split with a fixed seed ensures the same split on every run
        generator = torch.Generator().manual_seed(self.shuffle_seed)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [n_train, n_val, n_test], generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,               # shuffle training data each epoch
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,              # no shuffle for validation/test
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
        )
