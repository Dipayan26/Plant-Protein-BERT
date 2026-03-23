"""ESM2TokenizerWrapper: thin wrapper around HuggingFace EsmTokenizer.

Exposes the same interface as AminoAcidTokenizer so the existing data pipeline
can use it transparently for ESM-2 fine-tuning experiments.
"""

from __future__ import annotations

from transformers import EsmTokenizer


class ESM2TokenizerWrapper:
    """Wraps HuggingFace EsmTokenizer to match the AminoAcidTokenizer interface."""

    def __init__(
        self,
        esm2_model_name: str = "facebook/esm2_t6_8M_UR50D",
        vocab_size: int = 33,
    ) -> None:
        self._tokenizer = EsmTokenizer.from_pretrained(esm2_model_name)
        self.vocab_size = len(self._tokenizer)  # authoritative count from HF

        # Special token IDs (matching AminoAcidTokenizer attribute names)
        self.pad_token_id = self._tokenizer.pad_token_id
        self.cls_token_id = self._tokenizer.cls_token_id
        self.sep_token_id = self._tokenizer.eos_token_id   # ESM uses <eos> as [SEP]
        self.mask_token_id = self._tokenizer.mask_token_id
        self.unk_token_id = self._tokenizer.unk_token_id

    def __call__(
        self,
        sequence: str,
        max_length: int = 1024,
        truncation: bool = True,
        padding: str = "max_length",
        return_tensors: str | None = "pt",
    ) -> dict:
        return self._tokenizer(
            sequence,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
            return_tensors=return_tensors,
        )

    def encode(self, sequence: str, **kwargs) -> list[int]:
        return self._tokenizer.encode(sequence, **kwargs)

    def decode(self, token_ids: list[int], **kwargs) -> str:
        return self._tokenizer.decode(token_ids, **kwargs)

    def save(self, path: str) -> None:
        self._tokenizer.save_pretrained(path)
