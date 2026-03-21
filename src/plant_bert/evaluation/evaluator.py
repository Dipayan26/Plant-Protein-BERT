"""Evaluation runner for pretrained and fine-tuned models."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from omegaconf import DictConfig
from tqdm import tqdm

from .metrics import compute_mlm_accuracy, compute_perplexity

log = logging.getLogger(__name__)


class Evaluator:
    def __init__(
        self,
        metrics: list[str],
        splits_to_evaluate: list[str],
        batch_size: int,
        output_dir: str,
        **kwargs,
    ) -> None:
        self.metrics = metrics
        self.splits_to_evaluate = splits_to_evaluate
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)

    def evaluate(self, model: object, dataloader: object, split: str) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {split}"):
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                if "perplexity" in self.metrics or "mlm_accuracy" in self.metrics:
                    total_accuracy += compute_mlm_accuracy(outputs.logits, batch.get("labels"))
                n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        results = {
            f"{split}/mlm_loss": avg_loss,
            f"{split}/perplexity": compute_perplexity(avg_loss),
            f"{split}/mlm_accuracy": total_accuracy / max(n_batches, 1),
        }
        log.info(f"Evaluation results ({split}): {results}")
        return results
