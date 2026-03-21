"""Step 1: Parse raw UniProt DAT.gz files → filtered JSONL → HDF5 index.

Usage:
    python scripts/parse_uniprot.py data=sprot
    python scripts/parse_uniprot.py data=trembl_full
    python scripts/parse_uniprot.py data=trembl_full data.force_reprocess=true
"""

import hydra
from omegaconf import DictConfig

from plant_bert.data.preprocessing import run_preprocessing_pipeline
from plant_bert.utils import setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logger(cfg)
    run_preprocessing_pipeline(cfg.data)


if __name__ == "__main__":
    main()
