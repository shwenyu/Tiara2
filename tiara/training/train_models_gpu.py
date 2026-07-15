#!/usr/bin/env python3
"""Train Tiara's seven fixed-hyperparameter networks in parallel on multiple GPUs.

Recommended on the current 8x RTX 4090 host:
    python -m tiara.training.train_models_gpu DATA_DIR OUTPUT_DIR 2 \
        --gpus 1,2,3,4,5,6,7 --batch-size 4096

The third positional argument is CPU threads PER worker. Each architecture is
assigned to one GPU; GPU 0 can be excluded because it is currently in use.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import random
import time
from pathlib import Path
from typing import Any

# Prevent seven workers from each spawning a large BLAS/OpenMP thread pool.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pkg_resources
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from numba import njit
from skorch import NeuralNetClassifier
from torch import nn

import tiara
from tiara.src.transformations import TfidfWeighter


FIRST_STAGE_PARAMS = [
    dict(k=4, hidden_1=2048, hidden_2=2048, lr=0.001, dropout=0.2, epochs=41),
    dict(k=5, hidden_1=2048, hidden_2=2048, lr=0.001, dropout=0.2, epochs=28),
    dict(k=6, hidden_1=2048, hidden_2=1024, lr=0.001, dropout=0.2, epochs=41),
]

SECOND_STAGE_PARAMS = [
    dict(k=4, hidden_1=256, hidden_2=128, lr=0.001, dropout=0.2, epochs=45),
    dict(k=5, hidden_1=256, hidden_2=128, lr=0.001, dropout=0.2, epochs=37),
    dict(k=6, hidden_1=256, hidden_2=128, lr=0.001, dropout=0.5, epochs=30),
    dict(k=7, hidden_1=128, hidden_2=64, lr=0.01, dropout=0.2, epochs=47),
]

FILE_NAMES = {
    "mitochondria": "mitochondria_fr.fasta",
    "plastids": "plast_fr.fasta",
    "bacteria": "bacteria_fr.fasta",
    "eukarya": "eukarya_fr.fasta",
    "archaea": "archaea_fr.fasta",
}


class MyNNet2(nn.Sequential):
    """The original Tiara MLP architecture, unchanged."""

    def __init__(self, dim_in: int, hid1: int, hid2: int, dim_out: int, dropout: float):
        super().__init__(
            nn.Linear(dim_in, hid1),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hid1, hid2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hid2, dim_out),
            nn.Softmax(1),
        )


@njit(cache=True)
def count_kmers_into(seq: np.ndarray, k: int, out: np.ndarray) -> None:
    """Count canonical A/C/G/T k-mers using a rolling 2-bit code.

    This replaces the original Python substring + dictionary loop and avoids
    constructing a 4^11 mapping that is never needed.
    """
    mask = (1 << (2 * k)) - 1
    code = 0
    valid = 0
    for base in seq:
        if base == 65:       # A
            value = 0
        elif base == 67:     # C
            value = 1
        elif base == 71:     # G
            value = 2
        elif base == 84:     # T
            value = 3
        else:
            code = 0
            valid = 0
            continue
        code = ((code << 2) | value) & mask
        valid += 1
        if valid >= k:
            out[code] += 1.0


def read_fasta(path: Path, strict_acgt: bool = False) -> list[str]:
    with path.open() as handle:
        seqs = []
        for _, seq in SimpleFastaParser(handle):
            seq = seq.upper()
            # Preserve Tiara's original eukarya filtering semantics.
            if strict_acgt and set(seq) != set("ACGT"):
                continue
            seqs.append(seq)
    return seqs


def load_stage_sequences(data_dir: Path, stage: str) -> tuple[list[str], np.ndarray]:
    mito = read_fasta(data_dir / FILE_NAMES["mitochondria"])
    plast = read_fasta(data_dir / FILE_NAMES["plastids"])

    if stage == "second":
        seqs = plast + mito
        y = np.asarray([0] * len(plast) + [2] * len(mito), dtype=np.int64)
        return seqs, y

    bacteria = read_fasta(data_dir / FILE_NAMES["bacteria"])
    archaea = read_fasta(data_dir / FILE_NAMES["archaea"])
    eukarya = read_fasta(data_dir / FILE_NAMES["eukarya"], strict_acgt=True)
    seqs = plast + mito + bacteria + archaea + eukarya
    y = np.asarray(
        [0] * (len(plast) + len(mito))
        + [1] * len(bacteria)
        + [3] * len(archaea)
        + [4] * len(eukarya),
        dtype=np.int64,
    )
    return seqs, y


def idf_path(stage: str, k: int) -> Path:
    # Robust even when launched with `python -m ...` (__name__ == '__main__').
    return Path(tiara.__file__).resolve().parent / "models" / "tfidf-models" / f"k{k}-{stage}-stage"


def make_tfidf(seqs: list[str], stage: str, k: int) -> np.ndarray:
    path = idf_path(stage, k)
    if not path.exists():
        raise FileNotFoundError(f"Missing TF-IDF model: {path}")
    idf = np.asarray(TfidfWeighter.load_params(str(path)).idfs, dtype=np.float32)
    expected = 4**k
    if idf.shape[0] != expected:
        raise ValueError(f"IDF length {idf.shape[0]} != 4**{k} ({expected}) for {path}")

    print(f"[{stage} k={k}] computing TF-IDF for {len(seqs):,} sequences", flush=True)
    X = np.zeros((len(seqs), expected), dtype=np.float32)
    for i, seq in enumerate(seqs):
        raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
        count_kmers_into(raw, k, X[i])
        if (i + 1) % 50000 == 0 or i + 1 == len(seqs):
            print(f"[{stage} k={k}] features {i + 1:,}/{len(seqs):,}", flush=True)

    X *= idf
    norms = np.linalg.norm(X, axis=1)
    nonzero = norms > 0
    X[nonzero] /= norms[nonzero, None]
    if not np.all(nonzero):
        print(f"[{stage} k={k}] warning: {(~nonzero).sum()} zero-norm rows", flush=True)
    return np.ascontiguousarray(X)


def model_name(arch: dict[str, Any]) -> str:
    return "_".join(f"{key}-{value}" for key, value in arch.items()) + ".pkl"


def train_one(task: tuple[int, str, dict[str, Any], int, str, str, int, int]) -> str:
    task_index, stage, arch, gpu_id, data_dir_s, output_dir_s, cpu_threads, batch_size = task
    data_dir = Path(data_dir_s)
    output_dir = Path(output_dir_s)
    output_path = output_dir / model_name(arch)

    torch.set_num_threads(cpu_threads)
    torch.cuda.set_device(gpu_id)
    seed = 20260714 + task_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    label = f"{stage} k={arch['k']} gpu={gpu_id}"
    print(f"[{label}] START -> {output_path.name}", flush=True)
    started = time.time()
    seqs, y = load_stage_sequences(data_dir, stage)
    X = make_tfidf(seqs, stage, arch["k"])
    del seqs

    dim_out = 5 if stage == "first" else 3
    net = NeuralNetClassifier(
        MyNNet2(4 ** arch["k"], arch["hidden_1"], arch["hidden_2"], dim_out, arch["dropout"]),
        max_epochs=arch["epochs"],
        lr=arch["lr"],
        train_split=None,
        iterator_train__shuffle=True,
        iterator_train__pin_memory=True,
        optimizer=torch.optim.Adam,
        device=f"cuda:{gpu_id}",
        batch_size=batch_size,
        verbose=10,
    )
    net.fit(X, y)
    net.save_params(f_params=str(output_path))
    elapsed = (time.time() - started) / 60
    print(f"[{label}] DONE in {elapsed:.1f} min -> {output_path}", flush=True)
    return str(output_path)


def parse_gpu_ids(value: str) -> list[int]:
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must look like 1,2,3,4") from exc
    if not ids or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("--gpus must contain unique GPU IDs")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("cpu_threads", type=int, help="CPU threads per GPU worker (recommend 1-2)")
    parser.add_argument("--gpus", type=parse_gpu_ids, default=parse_gpu_ids("1,2,3,4,5,6,7"),
                        help="physical GPU IDs (default: 1,2,3,4,5,6,7; GPU 0 excluded)")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="training batch size per GPU (default: 4096)")
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="max concurrent GPU workers; lower to reduce peak RAM (default: 4)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [data_dir / name for name in FILE_NAMES.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing training FASTA files:\n" + "\n".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment")

    device_count = torch.cuda.device_count()
    invalid = [gpu for gpu in args.gpus if gpu < 0 or gpu >= device_count]
    if invalid:
        raise ValueError(f"Invalid GPU IDs {invalid}; torch sees {device_count} GPUs")

    jobs = [("first", arch) for arch in FIRST_STAGE_PARAMS] + [
        ("second", arch) for arch in SECOND_STAGE_PARAMS
    ]
    if len(args.gpus) < len(jobs):
        raise ValueError(f"Need 7 distinct GPUs for parallel mode; got {args.gpus}")

    selected = args.gpus[: len(jobs)]
    print(f"CUDA devices visible: {device_count}")
    print(f"Architecture -> GPU mapping:")
    for (stage, arch), gpu in zip(jobs, selected):
        free, total = torch.cuda.mem_get_info(gpu)
        print(f"  {stage:6s} k={arch['k']} -> GPU {gpu} "
              f"({torch.cuda.get_device_name(gpu)}, free {free / 2**30:.1f}/{total / 2**30:.1f} GiB)")

    tasks = [
        (i, stage, arch, gpu, str(data_dir), str(output_dir), args.cpu_threads, args.batch_size)
        for i, ((stage, arch), gpu) in enumerate(zip(jobs, selected))
    ]

    # CUDA requires spawn; fork after CUDA initialization is unsafe.
    ctx = mp.get_context("spawn")
    n_parallel = min(args.max_parallel, len(tasks))
    print(f"Running {len(tasks)} jobs, up to {n_parallel} concurrently "
          f"(max-parallel={args.max_parallel}); each worker pinned to its GPU.")
    with ctx.Pool(processes=n_parallel, maxtasksperchild=1) as pool:
        outputs = pool.map(train_one, tasks, chunksize=1)

    print("All seven models completed:")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
