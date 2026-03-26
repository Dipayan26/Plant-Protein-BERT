"""BPE (Byte-Pair Encoding) tokenizer — an alternative to the fixed amino-acid tokenizer.

BPE is a data-driven algorithm that learns common sub-sequences from the training
data and merges them into single tokens.  Example: if "AL" appears very often in
plant proteins, BPE may represent it as one token rather than two.

    Advantage: richer vocabulary can capture common motifs (e.g. signal peptides).
    Disadvantage: vocabulary must be trained first; harder to interpret.

For this project we use AminoAcidTokenizer by default (tokenizer: amino_acid).
Use this only if you want to experiment with sub-sequence representations.

Train via:
    python scripts/train_tokenizer.py tokenizer=bpe data=trembl_full
"""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


class BPETokenizer:
    """Byte-Pair Encoding tokenizer for protein sequences.

    If training_corpus is provided, BPE merges are learned from the corpus
    and saved to save_path.  If not provided, the tokenizer is untrained
    (only useful when loading from a pre-saved path via from_pretrained).
    """

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
        """Tokenize a protein sequence into sub-word token IDs."""
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
