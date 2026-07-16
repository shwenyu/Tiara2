#!/usr/bin/env python3
"""Train Tiara NNet models on GPUs with optimized HP JSON and model-level resume.

Examples
--------
Optimized v2.0 models from completed GPU searches::

    python -m tiara.training.train_models_gpu DATA_DIR OUTPUT_DIR 2 \
        --hp-dir /data/shouhanyu/Tiara2/log/training_v1.1 \
        --gpus 0,1,2,3,4,5,6,7 --max-parallel 4 \
        --min-free-mib 18000 --max-gpu-util 30 --poll-seconds 15 \
        --batch-size 4096 --resume

Before every model launch, the scheduler re-queries nvidia-smi, filters the
allowed GPU whitelist, then ranks eligible cards by free VRAM descending and
GPU utilization ascending. A selected GPU stays reserved until that model exits.

If --hp-dir is omitted, the original fixed Tiara parameter tables are used.
The HP-search JSON does not contain per-epoch validation histories, so final
training uses --epochs (default: 50), matching the completed search duration.
"""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import queue
import random
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch
from Bio.SeqIO.FastaIO import SimpleFastaParser
from numba import njit
from skorch import NeuralNetClassifier
from torch import nn

import tiara
from tiara.src.transformations import TfidfWeighter

FIXED_FIRST_STAGE_PARAMS = [
    dict(k=4, hidden_1=2048, hidden_2=2048, lr=0.001, dropout=0.2, epochs=41),
    dict(k=5, hidden_1=2048, hidden_2=2048, lr=0.001, dropout=0.2, epochs=28),
    dict(k=6, hidden_1=2048, hidden_2=1024, lr=0.001, dropout=0.2, epochs=41),
]
FIXED_SECOND_STAGE_PARAMS = [
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
_WORKER_GPU: int | None = None


class TiaraMLP(nn.Sequential):
    """Tiara MLP supporting both one- and two-hidden-layer search winners."""

    def __init__(self, dim_in: int, hid1: int, hid2: int | None,
                 dim_out: int, dropout: float):
        layers: list[nn.Module] = [
            nn.Linear(dim_in, hid1), nn.Dropout(dropout), nn.ReLU(inplace=True)
        ]
        last = hid1
        if hid2 is not None:
            layers.extend([
                nn.Linear(hid1, hid2), nn.Dropout(dropout), nn.ReLU(inplace=True)
            ])
            last = hid2
        layers.extend([nn.Linear(last, dim_out), nn.Softmax(1)])
        super().__init__(*layers)


@njit(cache=True)
def count_kmers_into(seq: np.ndarray, k: int, out: np.ndarray) -> None:
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


def read_fasta(path: Path, strict_acgt: bool = False) -> list[str]:
    seqs: list[str] = []
    with path.open() as handle:
        for _, seq in SimpleFastaParser(handle):
            seq = seq.upper()
            # Preserve the existing trainer's eukarya filtering behavior.
            if strict_acgt and not set(seq).issubset({"A", "C", "G", "T"}):
                continue
            seqs.append(seq)
    return seqs


def load_stage_sequences(data_dir: Path, stage: str) -> tuple[list[str], np.ndarray]:
    mito = read_fasta(data_dir / FILE_NAMES["mitochondria"])
    plast = read_fasta(data_dir / FILE_NAMES["plastids"])
    if stage == "second":
        return plast + mito, np.asarray([0] * len(plast) + [2] * len(mito), dtype=np.int64)

    bacteria = read_fasta(data_dir / FILE_NAMES["bacteria"])
    archaea = read_fasta(data_dir / FILE_NAMES["archaea"])
    eukarya = read_fasta(data_dir / FILE_NAMES["eukarya"], strict_acgt=True)
    seqs = plast + mito + bacteria + archaea + eukarya
    labels = (
        [0] * (len(plast) + len(mito))
        + [1] * len(bacteria)
        + [3] * len(archaea)
        + [4] * len(eukarya)
    )
    return seqs, np.asarray(labels, dtype=np.int64)


def idf_path(stage: str, k: int) -> Path:
    return (Path(tiara.__file__).resolve().parent / "models" / "tfidf-models"
            / f"k{k}-{stage}-stage")


def make_tfidf(seqs: list[str], stage: str, k: int) -> np.ndarray:
    path = idf_path(stage, k)
    if not path.exists():
        raise FileNotFoundError(f"Missing TF-IDF model: {path}")
    idf = np.asarray(TfidfWeighter.load_params(str(path)).idfs, dtype=np.float32)
    expected = 4 ** k
    if idf.shape[0] != expected:
        raise ValueError(f"IDF length {idf.shape[0]} != {expected} for {path}")

    print(f"[{stage} k={k}] computing TF-IDF for {len(seqs):,} sequences", flush=True)
    X = np.zeros((len(seqs), expected), dtype=np.float32)
    for i, seq in enumerate(seqs):
        raw = np.frombuffer(seq.encode("ascii", errors="ignore"), dtype=np.uint8)
        count_kmers_into(raw, k, X[i])
        if (i + 1) % 50000 == 0 or i + 1 == len(seqs):
            print(f"[{stage} k={k}] features {i + 1:,}/{len(seqs):,}", flush=True)
    X *= idf
    norms = np.linalg.norm(X, axis=1)
    nz = norms > 0
    X[nz] /= norms[nz, None]
    if not np.all(nz):
        print(f"[{stage} k={k}] warning: {(~nz).sum()} zero-norm rows", flush=True)
    return np.ascontiguousarray(X)


def hp_file(hp_dir: Path, stage: str, k: int) -> Path:
    return hp_dir / f"hp_{stage}_k{k}.json"


def load_hp_payload(path: Path, stage: str, k: int) -> dict[str, Any]:
    """Load the last valid HP object, tolerating concatenated/trailing data.

    Some resumed historical runs left more than one JSON document (or a small
    trailing fragment) in one .json file. json.load() rejects that with
    JSONDecodeError: Extra data. Decode documents incrementally and select the
    last complete object matching stage/k instead.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    pos = 0
    candidates: list[dict[str, Any]] = []
    while pos < len(text):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            break
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            if candidates:
                print(f"[warning] ignoring trailing non-JSON data in {path} at char {pos}: {exc}", flush=True)
                break
            raise ValueError(f"Cannot parse HP JSON {path}: {exc}") from exc
        if isinstance(obj, dict):
            try:
                matches = obj.get("stage") == stage and int(obj.get("k", -1)) == k
            except (TypeError, ValueError):
                matches = False
            if matches and isinstance(obj.get("results"), list) and obj["results"]:
                candidates.append(obj)
        pos = end
    if not candidates:
        raise ValueError(f"No complete stage={stage} k={k} result object in {path}")
    if len(candidates) > 1:
        print(f"[warning] {path} contains {len(candidates)} valid JSON documents; using the last one", flush=True)
    return candidates[-1]


def load_best_hp(hp_dir: Path, stage: str, k: int, epochs: int) -> dict[str, Any]:
    path = hp_file(hp_dir, stage, k)
    if not path.is_file():
        raise FileNotFoundError(f"Missing HP result: {path}")
    payload = load_hp_payload(path, stage, k)
    results = payload["results"]
    best = max(results, key=lambda row: float(row["mean_f1"]))
    return {
        "k": k,
        "hidden_1": int(best["hid1"]),
        "hidden_2": None if best.get("hid2") is None else int(best["hid2"]),
        "lr": float(best["learning_rate"]),
        "dropout": float(best["dropout"]),
        "epochs": epochs,
        "validation_mean_f1": float(best["mean_f1"]),
        "hp_source": str(path),
    }


def build_jobs(hp_dir: Path | None, epochs: int) -> list[tuple[str, dict[str, Any]]]:
    if hp_dir is None:
        return ([('first', dict(x)) for x in FIXED_FIRST_STAGE_PARAMS]
                + [('second', dict(x)) for x in FIXED_SECOND_STAGE_PARAMS])
    jobs: list[tuple[str, dict[str, Any]]] = []
    for k in (4, 5, 6):
        jobs.append(("first", load_best_hp(hp_dir, "first", k, epochs)))
    for k in (4, 5, 6, 7):
        jobs.append(("second", load_best_hp(hp_dir, "second", k, epochs)))
    return jobs


def fmt_value(value: Any) -> str:
    if value is None:
        return "none"
    return str(value)


def model_name(stage: str, arch: dict[str, Any]) -> str:
    # Stage prefix prevents accidental first/second-stage filename collisions.
    fields = [
        ("k", arch["k"]), ("hidden_1", arch["hidden_1"]),
        ("hidden_2", arch.get("hidden_2")), ("lr", arch["lr"]),
        ("dropout", arch["dropout"]), ("epochs", arch["epochs"]),
    ]
    return stage + "_" + "_".join(f"{k}-{fmt_value(v)}" for k, v in fields) + ".pkl"


def is_complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def init_worker(gpu_queue: Any, cpu_threads: int) -> None:
    global _WORKER_GPU
    _WORKER_GPU = int(gpu_queue.get())
    torch.set_num_threads(cpu_threads)
    torch.cuda.set_device(_WORKER_GPU)
    print(f"[worker pid={os.getpid()}] assigned GPU {_WORKER_GPU}", flush=True)


def train_one(task: tuple[int, str, dict[str, Any], str, str, int]) -> dict[str, Any]:
    task_index, stage, arch, data_dir_s, output_dir_s, batch_size = task
    assert _WORKER_GPU is not None
    gpu_id = _WORKER_GPU
    data_dir = Path(data_dir_s)
    output_dir = Path(output_dir_s)
    output_path = output_dir / model_name(stage, arch)
    tmp_path = output_dir / f".{output_path.name}.pid{os.getpid()}.tmp"

    seed = 20260716 + task_index
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    label = f"{stage} k={arch['k']} gpu={gpu_id}"
    print(f"[{label}] START -> {output_path.name}", flush=True)
    started = time.time()
    try:
        seqs, y = load_stage_sequences(data_dir, stage)
        X = make_tfidf(seqs, stage, int(arch["k"]))
        del seqs
        dim_out = 5 if stage == "first" else 3
        net = NeuralNetClassifier(
            TiaraMLP(4 ** int(arch["k"]), int(arch["hidden_1"]),
                     arch.get("hidden_2"), dim_out, float(arch["dropout"])),
            max_epochs=int(arch["epochs"]),
            lr=float(arch["lr"]),
            train_split=None,
            iterator_train__shuffle=True,
            iterator_train__pin_memory=True,
            optimizer=torch.optim.Adam,
            device=f"cuda:{gpu_id}",
            batch_size=batch_size,
            verbose=10,
        )
        net.fit(X, y)
        net.save_params(f_params=str(tmp_path))
        if not is_complete(tmp_path):
            raise RuntimeError(f"Temporary model was not written correctly: {tmp_path}")
        os.replace(tmp_path, output_path)  # atomic completion boundary for resume
        elapsed = (time.time() - started) / 60
        print(f"[{label}] DONE in {elapsed:.1f} min -> {output_path}", flush=True)
        return {"stage": stage, "k": arch["k"], "gpu": gpu_id,
                "output": str(output_path), "minutes": elapsed, "status": "trained"}
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        try:
            del X
        except UnboundLocalError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def query_gpu_metrics(allowed_gpus: list[int]) -> list[dict[str, int]]:
    """Return allowed GPUs ranked later by free memory and utilization."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    allowed = set(allowed_gpus)
    metrics: list[dict[str, int]] = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        gpu, free_mib, util = map(int, fields)
        if gpu in allowed:
            metrics.append({"gpu": gpu, "free_mib": free_mib, "util": util})
    return metrics


def dynamic_worker_entry(task: tuple[int, str, dict[str, Any], str, str, int],
                         gpu_id: int, cpu_threads: int, result_queue: Any) -> None:
    """Run one model on an explicitly reserved GPU and report the result."""
    global _WORKER_GPU
    _WORKER_GPU = gpu_id
    torch.set_num_threads(cpu_threads)
    torch.cuda.set_device(gpu_id)
    try:
        result_queue.put({"ok": True, "result": train_one(task)})
    except BaseException as exc:
        result_queue.put({
            "ok": False,
            "gpu": gpu_id,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        })
        raise


def parse_gpu_ids(value: str) -> list[int]:
    try:
        ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must look like 1,3,4,5") from exc
    if not ids or len(ids) != len(set(ids)):
        raise argparse.ArgumentTypeError("--gpus must contain unique GPU IDs")
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="flat directory with five *_fr.fasta files")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("cpu_threads", type=int, help="CPU threads per GPU worker (recommend 1-2)")
    parser.add_argument("--hp-dir", type=Path, default=None,
                        help="directory containing hp_first_k{4,5,6}.json and hp_second_k{4,5,6,7}.json")
    parser.add_argument("--epochs", type=int, default=50,
                        help="final epochs for optimized HP models (default: 50)")
    parser.add_argument("--gpus", type=parse_gpu_ids, default=parse_gpu_ids("0,1,2,3,4,5,6,7"),
                        help="allowed GPU whitelist; scheduler chooses among these dynamically")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-parallel", type=int, default=4,
                        help="maximum concurrent model processes")
    parser.add_argument("--min-free-mib", type=int, default=18000,
                        help="launch only when free VRAM is at least this many MiB (default: 18000 of 24564)")
    parser.add_argument("--max-gpu-util", type=int, default=30,
                        help="launch only when GPU utilization is at most this percent (default: 30)")
    parser.add_argument("--poll-seconds", type=float, default=15.0,
                        help="seconds between GPU availability checks (default: 15)")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                        help="skip non-empty completed model files (default: enabled)")
    return parser.parse_args()


def write_manifest(path: Path, source: str, jobs: list[tuple[str, dict[str, Any]]],
                   statuses: list[dict[str, Any]] | None = None) -> None:
    payload = {
        "parameter_source": source,
        "models": [{"stage": stage, **arch, "filename": model_name(stage, arch)}
                   for stage, arch in jobs],
        "statuses": statuses or [],
    }
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    hp_dir = args.hp_dir.expanduser().resolve() if args.hp_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    required = [data_dir / name for name in FILE_NAMES.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing training FASTA files:\n" + "\n".join(missing))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this PyTorch environment")
    invalid = [gpu for gpu in args.gpus if gpu < 0 or gpu >= torch.cuda.device_count()]
    if invalid:
        raise ValueError(f"Invalid GPU IDs {invalid}; torch sees {torch.cuda.device_count()} GPUs")
    if args.max_parallel < 1:
        raise ValueError("--max-parallel must be >= 1")

    jobs = build_jobs(hp_dir, args.epochs)
    source = str(hp_dir) if hp_dir else "built-in fixed Tiara parameters"
    manifest_path = output_dir / "training_manifest.json"
    write_manifest(manifest_path, source, jobs)

    statuses: list[dict[str, Any]] = []
    pending: list[tuple[int, str, dict[str, Any], str, str, int]] = []
    for idx, (stage, arch) in enumerate(jobs):
        target = output_dir / model_name(stage, arch)
        if args.resume and is_complete(target):
            print(f"[resume] SKIP {stage} k={arch['k']}: {target.name}", flush=True)
            statuses.append({"stage": stage, "k": arch["k"], "output": str(target), "status": "skipped_existing"})
        else:
            pending.append((idx, stage, arch, str(data_dir), str(output_dir), args.batch_size))

    if not pending:
        print("[resume] All seven optimized models are already complete.")
        write_manifest(manifest_path, source, jobs, statuses)
        return 0

    max_parallel = min(args.max_parallel, len(args.gpus), len(pending))
    print(f"Parameter source: {source}")
    for stage, arch in jobs:
        print(f"  {stage:6s} k={arch['k']} hid1={arch['hidden_1']} hid2={arch.get('hidden_2')} "
              f"lr={arch['lr']} drop={arch['dropout']} epochs={arch['epochs']} "
              f"valid_f1={arch.get('validation_mean_f1', 'n/a')}")
    print(f"Dynamic scheduler: pending={len(pending)}/7 max_parallel={max_parallel} "
          f"allowed={args.gpus} min_free={args.min_free_mib}MiB "
          f"max_util={args.max_gpu_util}% poll={args.poll_seconds}s")

    # Dynamic scheduling: before every launch, re-query all allowed GPUs and
    # rank eligible cards by free VRAM descending, then utilization ascending.
    # A GPU is reserved in `active` until its child process exits, so this
    # scheduler never places two of its own models on the same card.
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    active: dict[int, dict[str, Any]] = {}
    last_wait_log = 0.0
    failure: str | None = None

    try:
        while pending or active:
            # Reap completed children and release their GPU reservations.
            for pid, record in list(active.items()):
                proc = record["process"]
                if not proc.is_alive():
                    proc.join()
                    print(f"[scheduler] RELEASE GPU {record['gpu']} from pid={pid} "
                          f"exit={proc.exitcode}", flush=True)
                    if proc.exitcode != 0 and failure is None:
                        failure = (f"model process failed: pid={pid} gpu={record['gpu']} "
                                   f"exit={proc.exitcode}")
                    del active[pid]

            # Collect worker reports and update the manifest incrementally.
            while True:
                try:
                    message = result_queue.get_nowait()
                except queue.Empty:
                    break
                if message.get("ok"):
                    statuses.append(message["result"])
                    write_manifest(manifest_path, source, jobs, statuses)
                elif failure is None:
                    failure = message.get("traceback") or message.get("error", "unknown worker error")

            if failure is not None:
                raise RuntimeError(failure)

            launched = False
            while pending and len(active) < max_parallel:
                try:
                    metrics = query_gpu_metrics(args.gpus)
                except (OSError, subprocess.SubprocessError, ValueError) as exc:
                    now = time.time()
                    if now - last_wait_log >= 60:
                        print(f"[scheduler] nvidia-smi query failed: {exc}; retrying", flush=True)
                        last_wait_log = now
                    break

                reserved = {record["gpu"] for record in active.values()}
                eligible = [
                    row for row in metrics
                    if row["gpu"] not in reserved
                    and row["free_mib"] >= args.min_free_mib
                    and row["util"] <= args.max_gpu_util
                ]
                eligible.sort(key=lambda row: (-row["free_mib"], row["util"], row["gpu"]))
                if not eligible:
                    now = time.time()
                    if now - last_wait_log >= 60:
                        ranked = sorted(metrics, key=lambda row: (-row["free_mib"], row["util"]))
                        summary = ", ".join(
                            f"gpu{x['gpu']} free={x['free_mib']}MiB util={x['util']}%"
                            for x in ranked
                        )
                        print(f"[scheduler] WAIT: no eligible GPU ({summary})", flush=True)
                        last_wait_log = now
                    break

                chosen = eligible[0]
                task = pending.pop(0)
                _, stage, arch, *_ = task
                proc = ctx.Process(
                    target=dynamic_worker_entry,
                    args=(task, chosen["gpu"], args.cpu_threads, result_queue),
                )
                proc.start()
                active[proc.pid] = {
                    "process": proc,
                    "gpu": chosen["gpu"],
                    "stage": stage,
                    "k": arch["k"],
                }
                print(f"[scheduler] LAUNCH {stage} k={arch['k']} -> GPU {chosen['gpu']} "
                      f"(free={chosen['free_mib']}MiB util={chosen['util']}%) pid={proc.pid}",
                      flush=True)
                launched = True

            if pending or active:
                time.sleep(args.poll_seconds if not launched else min(2.0, args.poll_seconds))
    finally:
        # On Ctrl-C or a worker failure, stop remaining children. Completed
        # atomic .pkl files stay resumable; partial temporary files are ignored.
        for record in active.values():
            proc = record["process"]
            if proc.is_alive():
                proc.terminate()
        for record in active.values():
            record["process"].join(timeout=10)

    incomplete = [output_dir / model_name(stage, arch) for stage, arch in jobs
                  if not is_complete(output_dir / model_name(stage, arch))]
    if incomplete:
        raise RuntimeError("Training ended but models are missing:\n" + "\n".join(map(str, incomplete)))
    write_manifest(manifest_path, source, jobs, statuses)
    print(f"All seven models complete -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
