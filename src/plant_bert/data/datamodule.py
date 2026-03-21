"""Lightning DataModule wrapping the full data pipeline."""

from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from .dataset import InMemoryProteinDataset, StreamingProteinDataset


class UniProtDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, tokenizer: object) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.processed_dir = Path(cfg.processed_dir)

    def setup(self, stage: str | None = None) -> None:
        hdf5_path = self.processed_dir / "sequences.h5"
        jsonl_path = self.processed_dir / "filtered.jsonl"

        if self.cfg.streaming:
            full_dataset = StreamingProteinDataset(hdf5_path, self.tokenizer, self.cfg.max_length)
        else:
            full_dataset = InMemoryProteinDataset(jsonl_path, self.tokenizer, self.cfg.max_length)

        n = len(full_dataset)
        n_train = int(n * self.cfg.train_split)
        n_val = int(n * self.cfg.val_split)
        n_test = n - n_train - n_val

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=__import__("torch").Generator().manual_seed(self.cfg.shuffle_seed),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
