"""Tests for amino acid tokenizer encode/decode correctness."""

from __future__ import annotations

import pytest
from plant_bert.tokenizer import AminoAcidTokenizer


@pytest.fixture
def tokenizer():
    return AminoAcidTokenizer(vocab_size=30)


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 30


def test_encode_returns_correct_length(tokenizer):
    # Tokenizer spaces AAs: "ACDEF" → "A C D E F" (9 chars)
    # Split on every char → 5 AAs + 4 spaces = 2*N-1 tokens (no CLS/SEP appended)
    seq = "ACDEF"
    encoding = tokenizer(seq, padding=False, return_tensors=None)
    assert len(encoding["input_ids"]) == len(seq) * 2 - 1


def test_special_token_ids(tokenizer):
    pad_id  = tokenizer._tokenizer.pad_token_id
    mask_id = tokenizer._tokenizer.mask_token_id
    cls_id  = tokenizer._tokenizer.cls_token_id
    assert pad_id  == 0
    assert mask_id == 4
    assert cls_id  == 2


def test_unknown_aa_maps_to_unk(tokenizer):
    # Non-standard character should map to [UNK] (id=1)
    encoding = tokenizer("A#C", padding=False, return_tensors=None)
    assert 1 in encoding["input_ids"]


def test_ambiguous_aa_in_vocab(tokenizer):
    # B, Z, X, U, O are in vocab and must not map to [UNK]
    unk_id = tokenizer._tokenizer.unk_token_id
    for aa in "BZXUO":
        enc = tokenizer(aa, padding=False, return_tensors=None)
        # Single AA → spaced = "aa" → 1 token at index 0
        assert enc["input_ids"][0] != unk_id, f"{aa} mapped to [UNK]"
