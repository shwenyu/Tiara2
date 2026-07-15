#!/usr/bin/env python3
"""GPU-parallel hyperparameter search for Tiara's two-stage classifier.

Drop-in, ~1-order-of-magnitude-faster replacement for
hyperparameter_search_first_stage.py / hyperparameter_search_second_stage.py.

What is IDENTICAL to the originals (so results stay comparable):
  * input FASTA file names + label mapping per stage
  * the exact architecture grid (same loops, same order)
  * TF-IDF k-mer features (2-bit rolling count == the old dict lookup)
  * skorch NeuralNetClassifier defaults (NLLLoss, Adam, 50 epochs, Softmax head)

What changed (the speedups):
  * every candidate trains on GPU (device=cuda:N) instead of CPU
  * candidates are spread across all GPUs and run concurrently
  * features are computed ONCE per k with an njit rolling k-mer counter and
    shared to workers via on-disk memmap (no 8x recompute, low RAM)
  * dropped the per-epoch EpochScoring(mean_f1) (it re-ran predict every epoch)
  * fixed the original val-normalization bug: train and eval both use the
    L2-normalized matrices consistently

Called once per k by 05_train.sh, e.g.:
    python -m tiara.training.hyperparameter_search_gpu \
        --stage first --k 6 --gpus 0,1,2,3,4,5,6,7 --max-parallel 8 \
        /data/shouhanyu/Tiara2/train_ready \
        /data/shouhanyu/Tiara2/log/train_v1.1/hp_first_k6.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

# Keep each worker's BLAS/OpenMP footprint tiny; the GPU does the heavy lifting.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp

import numpy as np
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from numba import njit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from skorch import NeuralNetClassifier
from torch import nn

import tiara
from tiara.src.transformations import TfidfWeighter


# --------------------------------------------------------------------------- #
# Stage definitions (files + labels are copied verbatim from the originals).
# First stage concatenates:  organelle, bacteria, archea, eukarya -> 0,1,3,4
# Second stage concatenates: plastids, mitochondria              -> 0,2
# --------------------------------------------------------------------------- #
STAGE_SPEC = {
    "first": {
        "files": ["organelle", "bacteria", "archea", "eukarya"],
        "labels": [0, 1, 3, 4],
        "dim_out": 5,
        "idf": "first-stage",
    },
    "second": {
        "files": ["plastids", "mitochondria"],
        "labels": [0, 2],
        "dim_out": 3,
        "idf": "second-stage",
    },
}


def build_architectures(stage: str) -> list[dict[str, Any]]:
    """Reproduce the original architecture grids exactly (same order/count)."""
    architectures: list[dict[str, Any]] = []
    if stage == "first":
        for hid in [512, 1024, 2048]:
            for learning_rate in [0.001, 0.0001]:
                for dropout in [0.2]:
                    architectures.append(dict(hid1=hid, learning_rate=learning_rate, dropout=dropout))
                    architectures.append(dict(hid1=hid, hid2=hid, learning_rate=learning_rate, dropout=dropout))
        for learning_rate in [0.001, 0.0001]:
            for dropout in [0.2]:
                architectures.append(dict(hid1=512, hid2=256, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=1024, hid2=512, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=2048, hid2=1024, learning_rate=learning_rate, dropout=dropout))
        for hid in [32, 64, 128, 256]:
            for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
                for dropout in [0.2, 0.5]:
                    architectures.append(dict(hid1=hid, learning_rate=learning_rate, dropout=dropout))
                    architectures.append(dict(hid1=hid, hid2=hid, learning_rate=learning_rate, dropout=dropout))
        for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
            for dropout in [0.2, 0.5]:
                architectures.append(dict(hid1=64, hid2=32, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=128, hid2=64, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=128, hid2=64, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=256, hid2=128, learning_rate=learning_rate, dropout=dropout))
    elif stage == "second":
        for hid in [32, 64, 128, 256]:
            for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
                for dropout in [0.2, 0.5]:
                    architectures.append(dict(hid1=hid, learning_rate=learning_rate, dropout=dropout))
                    architectures.append(dict(hid1=hid, hid2=hid, learning_rate=learning_rate, dropout=dropout))
        for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
            for dropout in [0.2, 0.5]:
                architectures.append(dict(hid1=64, hid2=32, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=128, hid2=64, learning_rate=learning_rate, dropout=dropout))
                architectures.append(dict(hid1=256, hid2=128, learning_rate=learning_rate, dropout=dropout))
    else:
        raise ValueError(f"unknown stage: {stage}")
    return architectures


class MLP(nn.Sequential):
    """Same MLP as the originals: 1- or 2-hidden-layer, Dropout+ReLU, Softmax head."""

    def __init__(self, dim_in: int, hid1: int, hid2: int | None, dim_out: int, dropout: float):
        layers: list[nn.Module] = [nn.Linear(dim_in, hid1), nn.Dropout(dropout), nn.ReLU(inplace=True)]
        last = hid1
        if hid2:
            layers += [nn.Linear(hid1, hid2), nn.Dropout(dropout), nn.ReLU(inplace=True)]
            last = hid2
        layers += [nn.Linear(last, dim_out), nn.Softmax(1)]
        super().__init__(*layers)


@njit(cache=True)
def count_kmers_into(seq: np.ndarray, k: int, out: np.ndarray) -> None:
    """Rolling 2-bit ACGT k-mer counter. Index order == product('ACGT', repeat=k),
    so it matches the original dict-lookup indices and the stored IDF ordering.
    Windows spanning a non-ACGT base are skipped, exactly like the KeyError path.
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


def read_fasta(path: Path) -> list[str]:
    with path.open() as handle:
        return [seq for _, seq in SimpleFastaParser(handle)]


def load_idf(stage: str, k: int) -> np.ndarray:
    idf_dir = Path(tiara.__file__).resolve().parent / "models" / "tfidf-models" / f"k{k}-{STAGE_SPEC[stage]['idf']}"
    if not idf_dir.exists():
        raise FileNotFoundError(f"Missing TF-IDF model: {idf_dir}")
    idf = np.asarray(TfidfWeighter.load_params(str(idf_dir)).idfs, dtype=np.float32)
    expected = 4 ** k
    if idf.shape[0] != expected:
        raise ValueError(f"IDF length {idf.shape[0]} != 4**{k} ({expected}) for {idf_dir}")
    return idf


def featurize(seqs: list[str], k: int, idf: np.ndarray, dim: int) -> np.ndarray:
    X = np.zeros((len(seqs), dim), dtype=np.float32)
    for i, seq in enumerate(seqs):
        raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
        count_kmers_into(raw, k, X[i])
        if (i + 1) % 50000 == 0 or i + 1 == len(seqs):
            print(f"  features {i + 1:,}/{len(seqs):,}", flush=True)
    X *= idf
    norms = np.linalg.norm(X, axis=1)
    nz = norms > 0
    X[nz] /= norms[nz, None]
    return np.ascontiguousarray(X)


def build_split(input_dir: Path, stage: str, split: str, k: int, idf: np.ndarray, dim: int):
    spec = STAGE_SPEC[stage]
    groups = [read_fasta(input_dir / split / f"{name}.fasta") for name in spec["files"]]
    counts = [len(g) for g in groups]
    flat = [s for g in groups for s in g]
    print(f"[{stage} k={k}] {split}: " + ", ".join(f"{n}={c}" for n, c in zip(spec["files"], counts)), flush=True)
    X = featurize(flat, k, idf, dim)
    y = np.concatenate([np.full(c, lbl, dtype=np.int64) for c, lbl in zip(counts, spec["labels"])])
    return X, y


# --------------------------------------------------------------------------- #
# Worker: trains one candidate on one GPU, reading features from memmap.
# --------------------------------------------------------------------------- #
def train_one(task: tuple) -> dict[str, Any]:
    (idx, total, stage, k, arch, gpu_id, scratch, shapes, dim_out, epochs, batch_size, seed) = task

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    n_tr, dim = shapes["train"]
    n_va, _ = shapes["val"]
    # mode="c"（copy-on-write）：数组被标记为可写 → 消除 torch 的
    # "NumPy array is not writable" 警告；因为我们从不写入特征，页面仍按只读
    # 共享，内存不会翻倍。
    train_X = np.memmap(os.path.join(scratch, "train_X.f32"), dtype=np.float32, mode="c", shape=(n_tr, dim))
    val_X = np.memmap(os.path.join(scratch, "val_X.f32"), dtype=np.float32, mode="c", shape=(n_va, dim))
    # 标签很小，直接复制成可写数组即可（同样消除警告）。
    train_y = np.array(np.memmap(os.path.join(scratch, "train_y.i64"), dtype=np.int64, mode="r", shape=(n_tr,)))
    val_y = np.array(np.memmap(os.path.join(scratch, "val_y.i64"), dtype=np.int64, mode="r", shape=(n_va,)))

    hid1 = arch["hid1"]
    hid2 = arch.get("hid2")
    lr = arch["learning_rate"]
    drop = arch["dropout"]
    tag = f"hid1={hid1} hid2={hid2} lr={lr} drop={drop}"
    print(f"[{stage} k={k}] ({idx + 1}/{total}) START gpu={gpu_id} {tag}", flush=True)
    started = time.time()

    net = NeuralNetClassifier(
        MLP(dim, hid1, hid2, dim_out, drop),
        max_epochs=epochs,
        lr=lr,
        train_split=None,                 # train on all of train_X; eval val below
        iterator_train__shuffle=True,
        iterator_train__pin_memory=True,
        optimizer=torch.optim.Adam,
        device=device,
        batch_size=batch_size,
        verbose=0,
    )
    net.fit(np.asarray(train_X), train_y)
    y_pred = net.predict(np.asarray(val_X))

    accuracy = float(accuracy_score(val_y, y_pred))
    precision = precision_score(val_y, y_pred, average=None, zero_division=0).tolist()
    recall = recall_score(val_y, y_pred, average=None, zero_division=0).tolist()
    f1 = f1_score(val_y, y_pred, average=None, zero_division=0).tolist()
    mean_f1 = float(np.mean(f1))
    elapsed = time.time() - started
    print(f"[{stage} k={k}] ({idx + 1}/{total}) DONE  gpu={gpu_id} mean_f1={mean_f1:.4f} in {elapsed/60:.1f} min", flush=True)

    return {
        "index": idx,
        "stage": stage,
        "k": k,
        "hid1": hid1,
        "hid2": hid2,
        "learning_rate": lr,
        "dropout": drop,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_f1": mean_f1,
        "gpu": gpu_id,
        "seconds": elapsed,
    }


def parse_gpu_ids(value: str) -> list[int]:
    ids = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("--gpus must be unique ids like 0,1,2,3")
    return ids


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input_dir", type=Path, help="train_ready dir (has train/ and validation/)")
    p.add_argument("output_file", type=Path, help="results json (a .txt summary is written next to it)")
    p.add_argument("--stage", required=True, choices=["first", "second"])
    p.add_argument("--k", required=True, type=int)
    p.add_argument("--gpus", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3,4,5,6,7"))
    p.add_argument("--max-parallel", type=int, default=0, help="concurrent candidates (default: #gpus)")
    p.add_argument("--batch-size", type=int, default=512, help="per-candidate GPU batch size (default 512)")
    p.add_argument("--epochs", type=int, default=50, help="training epochs per candidate (default 50)")
    p.add_argument("--scratch", type=Path, default=None, help="dir for feature memmaps (default: next to output)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    stage, k = args.stage, args.k
    dim = 4 ** k

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment")
    device_count = torch.cuda.device_count()
    invalid = [g for g in args.gpus if g < 0 or g >= device_count]
    if invalid:
        raise ValueError(f"Invalid GPU IDs {invalid}; torch sees {device_count} GPUs")
    max_parallel = args.max_parallel or len(args.gpus)
    max_parallel = min(max_parallel, len(args.gpus))

    # ---- features once per k ----
    print(f"== [{stage} k={k}] computing features (dim={dim}) ==", flush=True)
    idf = load_idf(stage, k)
    train_X, train_y = build_split(input_dir, stage, "train", k, idf, dim)
    val_X, val_y = build_split(input_dir, stage, "validation", k, idf, dim)

    scratch_parent = (args.scratch or output_file.parent).expanduser().resolve()
    scratch_parent.mkdir(parents=True, exist_ok=True)
    scratch = tempfile.mkdtemp(prefix=f"hpfeat_{stage}_k{k}_", dir=str(scratch_parent))
    for name, arr in (("train_X.f32", train_X), ("val_X.f32", val_X),
                      ("train_y.i64", train_y), ("val_y.i64", val_y)):
        mm = np.memmap(os.path.join(scratch, name), dtype=arr.dtype, mode="w+", shape=arr.shape)
        mm[:] = arr[:]
        mm.flush()
        del mm
    shapes = {"train": train_X.shape, "val": val_X.shape}
    del train_X, val_X  # free RAM; workers read via memmap

    architectures = build_architectures(stage)
    total = len(architectures)
    spec = STAGE_SPEC[stage]
    print(f"== [{stage} k={k}] {total} candidates over GPUs {args.gpus}, "
          f"{max_parallel} at a time, batch={args.batch_size}, epochs={args.epochs} ==", flush=True)

    tasks = [
        (i, total, stage, k, arch, args.gpus[i % len(args.gpus)], scratch, shapes,
         spec["dim_out"], args.epochs, args.batch_size, 20260715 + i)
        for i, arch in enumerate(architectures)
    ]

    started = time.time()
    results = []
    ctx = mp.get_context("spawn")
    try:
        with ctx.Pool(processes=max_parallel, maxtasksperchild=1) as pool:
            # imap_unordered：哪个候选先跑完就先返回 → 可以实时刷新总进度。
            for r in pool.imap_unordered(train_one, tasks):
                results.append(r)
                done = len(results)
                elapsed = time.time() - started
                rate = elapsed / done                     # 已含并行加速的“每候选墙钟”
                eta = rate * (total - done)
                best_so_far = max(x["mean_f1"] for x in results)
                pct = 100.0 * done / total
                bar_len = 30
                filled = int(round(bar_len * done / total))
                bar = "#" * filled + "-" * (bar_len - filled)
                print(f"[{stage} k={k}] PROGRESS [{bar}] {done}/{total} ({pct:5.1f}%) "
                      f"| best_f1={best_so_far:.4f} | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m",
                      flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    results.sort(key=lambda r: r["mean_f1"], reverse=True)
    with output_file.open("w") as h:
        json.dump({"stage": stage, "k": k, "labels": spec["labels"], "results": results}, h, indent=2)

    txt = output_file.with_suffix(".txt")
    with txt.open("w") as h:
        h.write(f"stage={stage} k={k} labels={spec['labels']}  ({total} candidates)\n")
        h.write("rank  mean_f1   acc     hid1  hid2  lr        drop  gpu  min\n")
        for rank, r in enumerate(results, 1):
            h.write(f"{rank:>4}  {r['mean_f1']:.4f}  {r['accuracy']:.4f}  "
                    f"{str(r['hid1']):>4}  {str(r['hid2']):>4}  {r['learning_rate']:<8}  "
                    f"{r['dropout']:<4}  {r['gpu']:>3}  {r['seconds']/60:.1f}\n")

    best = results[0]
    print(f"== [{stage} k={k}] BEST mean_f1={best['mean_f1']:.4f} "
          f"hid1={best['hid1']} hid2={best['hid2']} lr={best['learning_rate']} drop={best['dropout']} ==", flush=True)
    print(f"== total wall time: {(time.time() - started)/60:.1f} min -> {output_file} ==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
