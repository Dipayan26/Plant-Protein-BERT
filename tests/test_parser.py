"""Tests for UniProt DAT parser."""

from __future__ import annotations

from plant_bert.data.uniprot_parser import parse_dat_gz


def test_parse_yields_records(synthetic_dat_gz):
    records = list(parse_dat_gz(synthetic_dat_gz, source="sprot"))
    assert len(records) == 2


def test_parse_record_fields(synthetic_dat_gz):
    records = list(parse_dat_gz(synthetic_dat_gz, source="sprot"))
    r = records[0]
    assert r.entry_name == "SYNTH1_ARATH"
    assert r.accessions[0] == "P00001"
    assert r.sequence == "ACDEFGHIKL"
    assert r.source == "sprot"
    assert r.is_reviewed is True


def test_parse_go_terms(synthetic_dat_gz):
    records = list(parse_dat_gz(synthetic_dat_gz, source="sprot"))
    assert "GO:0005634" in records[0].go_terms


def test_parse_second_record(synthetic_dat_gz):
    records = list(parse_dat_gz(synthetic_dat_gz, source="sprot"))
    assert records[1].sequence == "MNPQRSVW"
    assert records[1].go_terms == []
