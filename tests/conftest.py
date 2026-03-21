"""Shared pytest fixtures. All fixtures use tiny synthetic data — no dependency on real data files."""

from __future__ import annotations

import io
import gzip
import tempfile
from pathlib import Path

import pytest


SYNTHETIC_DAT = """\
ID   SYNTH1_ARATH    Reviewed;         10 AA.
AC   P00001;
DE   RecName: Full=Synthetic test protein 1;
OS   Arabidopsis thaliana (Mouse-ear cress).
OC   Eukaryota; Viridiplantae; Streptophyta; Embryophyta; Tracheophyta;
DR   GO; GO:0005634; C:nucleus; IDA:TAIR.
SQ   SEQUENCE   10 AA;
     ACDEFGHIKL
//
ID   SYNTH2_ARATH    Reviewed;         8 AA.
AC   P00002;
DE   RecName: Full=Synthetic test protein 2;
OS   Arabidopsis thaliana (Mouse-ear cress).
OC   Eukaryota; Viridiplantae; Streptophyta; Embryophyta; Tracheophyta;
SQ   SEQUENCE   8 AA;
     MNPQRSVW
//
"""


@pytest.fixture
def synthetic_dat_gz(tmp_path: Path) -> Path:
    """Write a tiny synthetic UniProt DAT.gz file."""
    path = tmp_path / "synthetic.dat.gz"
    with gzip.open(path, "wt") as f:
        f.write(SYNTHETIC_DAT)
    return path


@pytest.fixture
def synthetic_sequences() -> list[str]:
    return ["ACDEFGHIKL", "MNPQRSVW", "LLLLLLLLLLL"]
