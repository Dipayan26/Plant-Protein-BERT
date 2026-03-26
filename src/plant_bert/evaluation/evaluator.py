"""Standalone evaluation runner for pretrained models.

Used by scripts/evaluate.py to benchmark a trained checkpoint on val/test splits.
Computes perplexity and MLM accuracy (see metrics.py for definitions).

For fine-tuned models evaluated during training, use the built-in Lightning
validation/test steps in finetune.py — this Evaluator is only for pretraining.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from tqdm import tqdm

from .metrics import compute_mlm_accuracy, compute_perplexity

log = logging.getLogger(__name__)


class Evaluator:
    """Runs evaluation over a dataloader and returns metric dictionaries."""

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
        """Iterate over a dataloader and compute average metrics.

        Returns a dict like {"val/mlm_loss": 2.3, "val/perplexity": 9.9, ...}
        """
        model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        n_batches = 0

        with torch.no_grad():   # no_grad disables gradient computation (saves memory)
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
