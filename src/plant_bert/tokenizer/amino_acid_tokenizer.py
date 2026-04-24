"""Character-level amino acid tokenizer.

Protein sequences are strings of amino acid letters (e.g. "ACDEFGHIKL").
Before a neural network can process them, each letter must be converted to
a number (a "token ID").  This tokenizer does that one-to-one mapping:

    Vocabulary (30 tokens total):
      5 special tokens  — [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4
      20 standard AAs   — A=5, C=6, D=7, … Y=24
      5 ambiguous AAs   — B=25, Z=26, X=27, U=28, O=29

    [MASK] is the token used during Masked Language Modeling (MLM) training:
      the model sees "ACD[MASK]GHIKL" and must predict that position 4 = "E".

Wraps HuggingFace PreTrainedTokenizerFast so it works seamlessly with
transformers DataCollators (which handle the actual masking during training).
"""

from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast


STANDARD_AA = list("ACDEFGHIKLMNPQRSTVWY")   # 20 canonical amino acids
AMBIGUOUS_AA = list("BZXUO")                  # rare / ambiguous codes
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def _build_vocab() -> dict[str, int]:
    """Create the token → ID mapping.  Order: special tokens, standard AAs, ambiguous AAs."""
    tokens = SPECIAL_TOKENS + STANDARD_AA + AMBIGUOUS_AA
    return {tok: idx for idx, tok in enumerate(tokens)}


class AminoAcidTokenizer:
    """Character-level amino acid tokenizer (30 tokens).

    Each amino acid letter maps to exactly one token ID — no sub-word merging.
    This is the standard approach for protein language models because amino
    acids are already the fundamental "words" of protein sequences.
    """

    def __init__(self, vocab_size: int = 30, save_path: str | None = None, **kwargs) -> None:
        vocab = _build_vocab()
        tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
        # Split on every character so each amino acid becomes its own token.
        tokenizer.pre_tokenizer = Split(pattern="", behavior="isolated")
        tokenizer.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", vocab["[CLS]"]),
                ("[SEP]", vocab["[SEP]"]),
            ],
        )

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
        """Tokenize a protein sequence string into token IDs.

        Each amino acid character is tokenized directly:
        "ACDEF" → [CLS] A C D E F [SEP].
        """
        return self._tokenizer(sequence, **kwargs)

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    @classmethod
    def from_pretrained(cls, path: str) -> "AminoAcidTokenizer":
        obj = cls.__new__(cls)
        obj._tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
        return obj
