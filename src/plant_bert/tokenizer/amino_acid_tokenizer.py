"""Fixed-vocabulary amino acid tokenizer (25 AA + 5 special tokens = 30 total).

Character-level: each amino acid maps to exactly one token ID.
Wraps HuggingFace PreTrainedTokenizerFast for compatibility with transformers DataCollators.
"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")
AMBIGUOUS_AA = list("BZXUO")
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _build_vocab() -> dict[str, int]:
    tokens = SPECIAL_TOKENS + STANDARD_AA + AMBIGUOUS_AA
    return {tok: idx for idx, tok in enumerate(tokens)}


class AminoAcidTokenizer:
    """Character-level amino acid tokenizer."""

    def __init__(self, vocab_size: int = 30, save_path: str | None = None, **kwargs) -> None:
        vocab = _build_vocab()
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        # Split on individual characters — each AA becomes one token
        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        if save_path:
            self._tokenizer.save_pretrained(save_path)

    def __call__(self, sequence: str, **kwargs):
        # Insert spaces so each character is treated as a separate token
        spaced = " ".join(list(sequence))
        return self._tokenizer(spaced, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @classmethod
    def from_pretrained(cls, path: str) -> "AminoAcidTokenizer":
        obj = cls.__new__(cls)
        obj._tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        return obj
