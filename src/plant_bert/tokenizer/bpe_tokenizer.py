"""BPE tokenizer trained on plant protein sequences.

Train via: python scripts/train_tokenizer.py tokenizer=bpe data=trembl_full
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


class BPETokenizer:
    """Byte-pair encoding tokenizer for protein sequences."""

    def __init__(
        self,
        vocab_size: int = 8000,
        min_frequency: int = 10,
        special_tokens: list[str] | None = None,
        training_corpus: str | None = None,
        save_path: str | None = None,
        **kwargs,
    ) -> None:
        if special_tokens is None:
            special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")

        if training_corpus:
            trainer = BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
            )
            tokenizer.train(files=[training_corpus], trainer=trainer)

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
        spaced = " ".join(list(sequence))
        return self._tokenizer(spaced, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        obj = cls.__new__(cls)
        obj._tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        return obj
