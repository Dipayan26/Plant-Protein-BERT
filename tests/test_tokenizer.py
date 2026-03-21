"""Tests for amino acid tokenizer encode/decode correctness."""

from __future__ import annotations

import pytest
from plant_bert.tokenizer import AminoAcidTokenizer


@pytest.fixture
def tokenizer():
    return AminoAcidTokenizer(vocab_size=30)


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 30


def test_encode_standard_sequence(tokenizer):
    encoding = tokenizer("ACDEF", padding=False, return_tensors=None)
    assert len(encoding["input_ids"]) > 0


def test_round_trip(tokenizer):
    seq = "ACDEFGHIKL"
    encoding = tokenizer(seq, padding=False, return_tensors=None)
    decoded = tokenizer._tokenizer.decode(encoding["input_ids"], skip_special_tokens=True)
    # Decoded should contain original amino acids
    assert "A" in decoded
    assert "C" in decoded
