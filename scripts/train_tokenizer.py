"""Step 2 (optional): Train a BPE tokenizer on the processed sequences.

Skip this step when using tokenizer=amino_acid (no training required).

Usage:
    python scripts/train_tokenizer.py tokenizer=bpe data=trembl_full
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from plant_bert.utils import setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logger(cfg)
    # Instantiating with training_corpus triggers BPE training and saves to save_path
    instantiate(cfg.tokenizer)


if __name__ == "__main__":
    main()
