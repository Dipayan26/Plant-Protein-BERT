from .uniprot_parser import UniProtRecord, parse_dat_gz
from .dataset import InMemoryProteinDataset, StreamingProteinDataset
from .datamodule import UniProtDataModule

__all__ = [
    "UniProtRecord",
    "parse_dat_gz",
    "InMemoryProteinDataset",
    "StreamingProteinDataset",
    "UniProtDataModule",
]
