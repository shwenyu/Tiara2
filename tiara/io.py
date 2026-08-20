"""Small FASTA reader with plain-text and gzip support."""
from __future__ import annotations

import gzip
from pathlib import Path


def fasta(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"input FASTA not found: {path}")
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    header = None
    parts = []
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(parts).upper()
                header = line[1:].strip()
                if not header:
                    raise ValueError("FASTA record has an empty header")
                parts = []
            else:
                parts.append("".join(line.split()))
    if header is not None:
        yield header, "".join(parts).upper()
