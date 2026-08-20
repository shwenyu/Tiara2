"""Minimal TF-IDF feature extraction used by Tiara2 inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:  # Keeps configuration and lightweight tests usable before Conda setup.
    def njit(*args, **kwargs):
        if args and callable(args[0]) and not kwargs:
            return args[0]

        def decorate(function):
            return function

        return decorate


@dataclass(frozen=True)
class TfidfModel:
    k: int
    idfs: np.ndarray

    @classmethod
    def load(cls, directory) -> "TfidfModel":
        directory = Path(directory)
        params = {}
        for line in (directory / "params.txt").read_text().splitlines():
            if line.strip():
                key, value = line.split(":", 1)
                params[key] = value
        idfs = np.load(directory / "model.npy").astype(np.float32)
        k = int(params["k"])
        expected = 4**k
        if idfs.shape != (expected,):
            raise ValueError(f"TF-IDF dimension mismatch: expected {expected}, got {idfs.shape}")
        return cls(k=k, idfs=idfs)


@njit(cache=True)
def _count_kmers(seq: np.ndarray, k: int, out: np.ndarray) -> None:
    mask = (1 << (2 * k)) - 1
    code = 0
    valid = 0
    for base in seq:
        if base == 65:
            value = 0
        elif base == 67:
            value = 1
        elif base == 71:
            value = 2
        elif base == 84:
            value = 3
        else:
            code = 0
            valid = 0
            continue
        code = ((code << 2) | value) & mask
        valid += 1
        if valid >= k:
            out[code] += 1.0


def featurize_block(sequences, k: int, idfs: np.ndarray) -> np.ndarray:
    dim = 4**k
    matrix = np.zeros((len(sequences), dim), dtype=np.float32)
    for index, sequence in enumerate(sequences):
        raw = np.frombuffer(sequence.encode("ascii", errors="ignore"), dtype=np.uint8)
        _count_kmers(raw, k, matrix[index])
    matrix *= idfs
    norms = np.linalg.norm(matrix, axis=1)
    nonzero = norms > 0
    matrix[nonzero] /= norms[nonzero, None]
    return matrix
