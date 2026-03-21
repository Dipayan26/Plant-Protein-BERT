"""Streaming parser for UniProt flat-file (.dat.gz) format.

Yields UniProtRecord instances one at a time — never loads the full file into memory.
This is required for the 18 GB TrEMBL file.

UniProt DAT format reference:
  https://web.expasy.org/docs/userman.html
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

_OX_RE = re.compile(r"NCBI_TaxID=(\d+)")


@dataclass
class UniProtRecord:
    entry_name: str = ""
    accessions: list[str] = field(default_factory=list)
    description: str = ""
    organism: str = ""
    taxonomy: list[str] = field(default_factory=list)
    tax_id: int = 0       # NCBI TaxID from OX line — used for Viridiplantae filtering
    sequence: str = ""
    go_terms: list[str] = field(default_factory=list)
    source: str = ""  # "sprot" or "trembl"

    @property
    def is_reviewed(self) -> bool:
        """SwissProt (reviewed) entries have manually curated annotations."""
        return self.source == "sprot"


def parse_dat_gz(filepath: str | Path, source: str = "") -> Generator[UniProtRecord, None, None]:
    """Lazily parse a gzipped UniProt DAT file, yielding one record at a time.

    Args:
        filepath: Path to the .dat.gz file.
        source: Label for the source database, e.g. "sprot" or "trembl".

    Yields:
        UniProtRecord for each protein entry.
    """
    filepath = Path(filepath)
    record = UniProtRecord(source=source)
    in_sequence = False
    seq_lines: list[str] = []

    with gzip.open(filepath, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line_type = line[:2]

            if line_type == "ID":
                # ID   ENTRY_NAME STATUS; LENGTH AA.
                parts = line[5:].split()
                record.entry_name = parts[0]

            elif line_type == "AC":
                # AC   P12345; Q67890;
                acs = [ac.rstrip(";") for ac in line[5:].split()]
                record.accessions.extend(acs)

            elif line_type == "DE":
                desc = line[5:].strip()
                if record.description:
                    record.description += " " + desc
                else:
                    record.description = desc

            elif line_type == "OS":
                record.organism += line[5:].strip().rstrip(".")

            elif line_type == "OX":
                # OX   NCBI_TaxID=3702 {ECO:0000313|...}
                m = _OX_RE.search(line)
                if m:
                    record.tax_id = int(m.group(1))

            elif line_type == "OC":
                taxa = [t.strip().rstrip(";.") for t in line[5:].split(";") if t.strip()]
                record.taxonomy.extend(taxa)

            elif line_type == "DR":
                # DR   GO; GO:0005634; C:nucleus; IDA:TAIR.
                if line[5:9] == "GO; ":
                    go_id = line[9:].split(";")[0].strip()
                    record.go_terms.append(go_id)

            elif line_type == "SQ":
                # SQ   SEQUENCE  NNN AA; ...
                in_sequence = True
                seq_lines = []

            elif in_sequence and line_type == "  ":
                # Sequence continuation lines have no tag (start with spaces)
                seq_lines.append(line.strip().replace(" ", ""))

            elif line_type == "//":
                # End of record
                record.sequence = "".join(seq_lines)
                in_sequence = False

                if record.sequence:
                    yield record

                record = UniProtRecord(source=source)
                seq_lines = []
