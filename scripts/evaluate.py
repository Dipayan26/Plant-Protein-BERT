"""Step 5: Run evaluation on a trained checkpoint.

Usage:
    python scripts/evaluate.py \
        training.pretrained_checkpoint=outputs/.../checkpoints/finetune/best.ckpt
"""

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from plant_bert.utils import seed_everything, setup_logger


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_logger(cfg)
    seed_everything(cfg.seed)

    tokenizer = instantiate(cfg.tokenizer)
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.data, tokenizer=tokenizer)
    evaluator = instantiate(cfg.evaluation)

    datamodule.setup()
    for split in evaluator.splits_to_evaluate:
        loader = getattr(datamodule, f"{split}_dataloader")()
        results = evaluator.evaluate(model, loader, split)
        print(results)


if __name__ == "__main__":
    main()
