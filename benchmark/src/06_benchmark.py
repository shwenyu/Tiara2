#!/usr/bin/env python3
"""
06_benchmark.py

Streaming, bounded-memory Tiara benchmark runner.

For every model x test-set pair:
1. Load the Tiara two-stage classifier once.
2. Stream FASTA records in bounded chunks.
3. Classify one temporary chunk at a time.
4. Update confusion matrices and grouped metrics online.
5. Optionally append per-sequence predictions to gzip TSV.
6. Write aggregate benchmark tables and summary.md.

This implementation avoids retaining:
- the complete FASTA in memory;
- a global sequence-ID -> gold-label dictionary;
- all predictions;
- all evaluation rows.

The inprocess backend supports CUDA.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import gzip
import json
import math
import os
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO

import numpy as np


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def _log(message: str) -> None:
    print(f"      [{dt.datetime.now():%H:%M:%S}] {message}", flush=True)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

GOLD_CLASSES = [
    "archaea",
    "bacteria",
    "eukarya",
    "mitochondrion",
    "plastid",
]

EXTRA_PRED_LABELS = [
    "prokarya",
    "organelle",
    "unknown",
]

ALL_LABELS = GOLD_CLASSES + EXTRA_PRED_LABELS

LABEL_ALIASES = {
    "arc": "archaea",
    "archaea": "archaea",
    "archea": "archaea",

    "bac": "bacteria",
    "bacteria": "bacteria",

    "euk": "eukarya",
    "eukarya": "eukarya",
    "eukaryota": "eukarya",

    "mit": "mitochondrion",
    "mito": "mitochondrion",
    "mitochondria": "mitochondrion",
    "mitochondrion": "mitochondrion",

    "pla": "plastid",
    "plast": "plastid",
    "plastid": "plastid",
    "plastids": "plastid",
    "chloroplast": "plastid",

    "pro": "prokarya",
    "prokarya": "prokarya",

    "org": "organelle",
    "organelle": "organelle",

    "unk": "unknown",
    "unknown": "unknown",
}


def normalise_label(value: str | None) -> str:
    if value is None:
        return "unknown"
    key = str(value).strip().lower()
    return LABEL_ALIASES.get(key, key)


def final_label(stage1: str, stage2: str) -> str:
    stage1 = normalise_label(stage1)
    if stage1 == "organelle":
        return normalise_label(stage2)
    return stage1


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")

    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                "PyYAML is not installed. Run `pip install pyyaml` "
                "or use a JSON config."
            ) from exc

        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)

    if not isinstance(loaded, dict):
        raise ValueError("The benchmark config must contain a mapping/object.")

    return loaded


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


# --------------------------------------------------------------------------- #
# FASTA streaming
# --------------------------------------------------------------------------- #

def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Yield one (header, sequence) tuple at a time."""

    opener = gzip.open if path.suffix.lower() == ".gz" else open

    header: str | None = None
    chunks: list[str] = []

    with opener(path, "rt") as handle:
        for line in handle:
            line = line.rstrip("\r\n")

            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)

                header = line[1:]
                chunks = []
            else:
                chunks.append(line.strip())

    if header is not None:
        yield header, "".join(chunks)


def iter_chunks(
    records: Iterable[tuple[str, str]],
    chunk_size: int,
) -> Iterator[list[tuple[str, str]]]:
    """Collect a bounded number of FASTA records at a time."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    chunk: list[tuple[str, str]] = []

    for record in records:
        chunk.append(record)

        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []

    if chunk:
        yield chunk


def seq_id(header: str) -> str:
    return header.split()[0] if header else header


def gold_from_header(header: str) -> str | None:
    for token in header.replace("\t", " ").split():
        for key in ("label=", "sg=", "class="):
            if token.lower().startswith(key):
                return normalise_label(token.split("=", 1)[1])
    return None


def count_labelled_records(
    path: Path,
    default_gold: str | None,
) -> tuple[int, int]:
    """Streaming dry-run counter; does not retain sequence IDs."""

    total = 0
    labelled = 0

    for header, _ in iter_fasta(path):
        total += 1
        gold = gold_from_header(header) or default_gold
        if gold is not None:
            labelled += 1

    return total, labelled


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #

def per_class_prf(
    matrix: np.ndarray,
    labels: list[str],
) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}

    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)

    for i, label in enumerate(labels):
        tp = int(matrix[i, i])
        support = int(row_sums[i])

        precision = tp / col_sums[i] if col_sums[i] else 0.0
        recall = tp / support if support else 0.0

        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        output[label] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

    return output


def macro_f1(
    prf: dict[str, dict[str, float | int]],
    classes: list[str],
) -> float:
    values = [
        float(prf[label]["f1"])
        for label in classes
        if label in prf and int(prf[label]["support"]) > 0
    ]

    return float(np.mean(values)) if values else 0.0


def accuracy(matrix: np.ndarray) -> float:
    total = int(matrix.sum())
    return float(matrix.trace() / total) if total else 0.0


def multiclass_mcc(matrix: np.ndarray) -> float:
    """
    Gorodkin multiclass MCC.

    Extra predicted classes have zero gold support but are retained in the
    square matrix, so unknown/prokarya/organelle predictions count as errors.
    """

    c = matrix.astype(np.float64)
    total = c.sum()

    if total == 0:
        return 0.0

    row = c.sum(axis=1)
    col = c.sum(axis=0)

    cov_xy = c.trace() * total - float(row @ col)
    cov_xx = total * total - float(row @ row)
    cov_yy = total * total - float(col @ col)

    denominator = math.sqrt(max(cov_xx * cov_yy, 0.0))

    return float(cov_xy / denominator) if denominator else 0.0


def length_bucket(length: int, edges: list[int]) -> str:
    for index, edge in enumerate(edges):
        if length < edge:
            if index == 0:
                return f"<{edge}"
            return f"{edges[index - 1]}-{edge - 1}"

    return f">={edges[-1]}"


# --------------------------------------------------------------------------- #
# Online metric accumulator
# --------------------------------------------------------------------------- #

class OnlineMetrics:
    """Constant-memory aggregate benchmark statistics."""

    def __init__(self, edges: list[int]) -> None:
        self.labels = list(ALL_LABELS)
        self.label_index = {
            label: index
            for index, label in enumerate(self.labels)
        }

        shape = (len(self.labels), len(self.labels))

        self.confusion = np.zeros(shape, dtype=np.int64)
        self.by_length: dict[str, np.ndarray] = {}
        self.by_group: dict[str, list[int]] = {}

        self.edges = edges
        self.predicted_records = 0
        self.skipped_unknown_gold = 0

    def add(
        self,
        gold: str,
        predicted: str,
        sequence_length: int,
        group: str,
    ) -> None:
        gold = normalise_label(gold)
        predicted = normalise_label(predicted)
        group = normalise_label(group)

        if gold not in self.label_index:
            self.skipped_unknown_gold += 1
            return

        if predicted not in self.label_index:
            predicted = "unknown"

        gold_index = self.label_index[gold]
        pred_index = self.label_index[predicted]

        self.confusion[gold_index, pred_index] += 1
        self.predicted_records += 1

        bucket = length_bucket(sequence_length, self.edges)

        if bucket not in self.by_length:
            self.by_length[bucket] = np.zeros_like(self.confusion)

        self.by_length[bucket][gold_index, pred_index] += 1

        if group not in self.by_group:
            # [total, correct]
            self.by_group[group] = [0, 0]

        self.by_group[group][0] += 1

        if gold == predicted:
            self.by_group[group][1] += 1

    def summary(self) -> dict[str, Any]:
        prf = per_class_prf(self.confusion, self.labels)

        by_length_summary: dict[str, dict[str, float | int]] = {}

        for bucket, matrix in self.by_length.items():
            bucket_prf = per_class_prf(matrix, self.labels)

            by_length_summary[bucket] = {
                "n": int(matrix.sum()),
                "accuracy": accuracy(matrix),
                "macro_f1": macro_f1(bucket_prf, GOLD_CLASSES),
            }

        by_group_summary: dict[str, dict[str, float | int]] = {}

        for group, (total, correct) in self.by_group.items():
            by_group_summary[group] = {
                "n": int(total),
                "recall": float(correct / total) if total else 0.0,
            }

        return {
            "n": int(self.confusion.sum()),
            "accuracy": accuracy(self.confusion),
            "macro_f1": macro_f1(prf, GOLD_CLASSES),
            "mcc": multiclass_mcc(self.confusion),
            "per_class": prf,
            "labels": self.labels,
            "confusion": self.confusion.copy(),
            "by_length": by_length_summary,
            "by_group": by_group_summary,
            "skipped_unknown_gold": self.skipped_unknown_gold,
        }


# --------------------------------------------------------------------------- #
# Tiara model construction
# --------------------------------------------------------------------------- #

def _stage_params(
    model: dict[str, Any],
    stage: str,
    k: int,
) -> dict[str, Any]:
    first = {
        4: (2048, 2048, 0.2),
        5: (2048, 2048, 0.2),
        6: (2048, 1024, 0.2),
    }

    second = {
        4: (256, 128, 0.2),
        5: (256, 128, 0.2),
        6: (256, 128, 0.5),
        7: (128, 64, 0.2),
    }

    table = first if stage == "first" else second

    if k not in table:
        raise ValueError(f"Unsupported {stage}-stage k={k}")

    hidden_1, hidden_2, dropout = table[k]

    return {
        "k": k,
        "hidden_1": hidden_1,
        "hidden_2": hidden_2,
        "dropout": dropout,
    }


def build_classifier(
    model: dict[str, Any],
    cfg: dict[str, Any],
    device: str,
):
    import torch
    from tiara.src.classification import Classification

    threads = int(cfg.get("threads", 4))
    torch.set_num_threads(threads)

    prob_cutoff = cfg.get("prob_cutoff", [0.65, 0.65])

    if isinstance(prob_cutoff, (int, float)):
        prob_cutoff = [float(prob_cutoff), float(prob_cutoff)]

    if len(prob_cutoff) != 2:
        raise ValueError("prob_cutoff must contain exactly two values")

    k1 = int(model["k1"])
    k2 = int(model["k2"])

    first = _stage_params(model, "first", k1)
    second = _stage_params(model, "second", k2)

    first.update(
        prob_cutoff=float(prob_cutoff[0]),
        fragment_len=5000,
        dim_out=5,
    )

    second.update(
        prob_cutoff=float(prob_cutoff[1]),
        fragment_len=5000,
        dim_out=3,
    )

    _log(
        f"loading model '{model['name']}' "
        f"(k1={k1}, k2={k2}, threads={threads}) ..."
    )

    classifier = Classification(
        min_len=int(cfg.get("min_len", 3000)),
        nnet_weights=[
            str(Path(model["nnet_first"]).expanduser()),
            str(Path(model["nnet_second"]).expanduser()),
        ],
        params=[first, second],
        tfidf=[
            str(Path(model["tfidf_first"]).expanduser()),
            str(Path(model["tfidf_second"]).expanduser()),
        ],
        threads=threads,
    )

    _log("model loaded")

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device={device} requested, but CUDA is unavailable"
            )

        for network in classifier.nnets:
            network.module_.to(device)
            network.device = device

        _log(f"moved neural networks to {device}")

    return classifier


# --------------------------------------------------------------------------- #
# Streaming classification
# --------------------------------------------------------------------------- #

def process_fasta_streaming(
    classifier,
    fasta: Path,
    default_gold: str | None,
    fixed_group: str | None,
    metrics: OnlineMetrics,
    cfg: dict[str, Any],
    raw_handle: TextIO | None = None,
) -> dict[str, int]:
    """
    Stream one FASTA through Tiara.

    Only one input chunk and its chunk-local metadata are held in memory.
    """

    chunk_size = int(cfg.get("chunk_size", 10_000))
    min_len = int(cfg.get("min_len", 3000))
    verbose = bool(cfg.get("verbose", False))
    log_every = max(1, int(cfg.get("log_every_chunks", 1)))

    if chunk_size <= 0:
        raise ValueError(
            "Streaming benchmark requires chunk_size > 0"
        )

    temporary_directory = cfg.get("tmp_dir")

    if temporary_directory:
        temp_dir = Path(temporary_directory).expanduser()
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dir_string: str | None = str(temp_dir)
    else:
        temp_dir_string = None

    total_seen = 0
    total_eligible = 0
    total_predicted = 0
    total_unlabelled = 0

    _log(
        f"streaming {fasta.name}: "
        f"chunk_size={chunk_size}, min_len={min_len}"
    )

    chunks = iter_chunks(iter_fasta(fasta), chunk_size)

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk_start = time.time()

        # Maps synthetic chunk-local ID to original metadata.
        metadata: dict[str, tuple[str, str, int, str]] = {}

        tmp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".fasta",
                prefix="tiara_benchmark_",
                dir=temp_dir_string,
                delete=False,
                encoding="utf-8",
            ) as temporary_fasta:
                tmp_path = temporary_fasta.name

                for record_number, (header, sequence) in enumerate(chunk):
                    total_seen += 1

                    gold = gold_from_header(header) or default_gold

                    if gold is None:
                        total_unlabelled += 1
                        continue

                    gold = normalise_label(gold)
                    group = normalise_label(fixed_group or gold)
                    original_id = seq_id(header)
                    sequence_length = len(sequence)

                    # Tiara will not predict sequences shorter than min_len.
                    # We omit them from the temporary file, matching the old
                    # benchmark's behaviour of evaluating predicted rows only.
                    if sequence_length < min_len:
                        continue

                    total_eligible += 1

                    internal_id = (
                        f"bench_{chunk_number}_{record_number}"
                    )

                    metadata[internal_id] = (
                        original_id,
                        gold,
                        sequence_length,
                        group,
                    )

                    temporary_fasta.write(
                        f">{internal_id}\n{sequence}\n"
                    )

            if not metadata:
                if chunk_number % log_every == 0:
                    _log(
                        f"{fasta.name}: chunk {chunk_number} contained "
                        "no eligible labelled records"
                    )
                continue

            predicted_in_chunk = 0

            result_iterator = classifier.classify(
                tmp_path,
                verbose=verbose,
            )

            for record in result_iterator:
                internal_id = seq_id(record.desc)
                item = metadata.get(internal_id)

                if item is None:
                    continue

                original_id, gold, sequence_length, group = item

                stage1 = (
                    record.cls[0]
                    if len(record.cls) >= 1
                    else "unknown"
                )

                stage2 = (
                    record.cls[1]
                    if len(record.cls) >= 2
                    else "unknown"
                )

                predicted = final_label(stage1, stage2)

                metrics.add(
                    gold=gold,
                    predicted=predicted,
                    sequence_length=sequence_length,
                    group=group,
                )

                predicted_in_chunk += 1
                total_predicted += 1

                if raw_handle is not None:
                    raw_handle.write(
                        "\t".join(
                            [
                                original_id,
                                gold,
                                normalise_label(stage1),
                                normalise_label(stage2),
                                predicted,
                                str(sequence_length),
                                group,
                                fasta.name,
                            ]
                        )
                        + "\n"
                    )

            elapsed = time.time() - chunk_start
            missing = len(metadata) - predicted_in_chunk

            if (
                chunk_number == 1
                or chunk_number % log_every == 0
                or missing > 0
            ):
                _log(
                    f"{fasta.name}: chunk {chunk_number} done; "
                    f"input={len(chunk):,}, "
                    f"eligible={len(metadata):,}, "
                    f"predicted={predicted_in_chunk:,}, "
                    f"missing={missing:,}, "
                    f"time={elapsed:.1f}s, "
                    f"total_predicted={total_predicted:,}"
                )

        finally:
            if tmp_path is not None:
                try:
                    os.remove(tmp_path)
                except FileNotFoundError:
                    pass

            # Release chunk-local strings, sequences and Tiara records.
            del metadata
            del chunk
            gc.collect()

    _log(
        f"completed {fasta.name}: "
        f"seen={total_seen:,}, "
        f"eligible={total_eligible:,}, "
        f"predicted={total_predicted:,}, "
        f"unlabelled={total_unlabelled:,}"
    )

    return {
        "seen": total_seen,
        "eligible": total_eligible,
        "predicted": total_predicted,
        "unlabelled": total_unlabelled,
    }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_tsv(
    path: Path,
    header: list[str],
    rows: list[list[Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")

        for row in rows:
            handle.write(
                "\t".join(str(value) for value in row) + "\n"
            )


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_outputs(
    results: dict[str, dict[str, dict[str, Any]]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_rows: list[list[Any]] = []
    per_class_rows: list[list[Any]] = []
    per_length_rows: list[list[Any]] = []
    per_group_rows: list[list[Any]] = []

    for testset, models in results.items():
        for model_name, summary in models.items():
            overall_rows.append(
                [
                    testset,
                    model_name,
                    summary["n"],
                    fmt(summary["accuracy"]),
                    fmt(summary["macro_f1"]),
                    fmt(summary["mcc"]),
                ]
            )

            for label, prf in summary["per_class"].items():
                if (
                    int(prf["support"]) == 0
                    and label in EXTRA_PRED_LABELS
                ):
                    continue

                per_class_rows.append(
                    [
                        testset,
                        model_name,
                        label,
                        fmt(float(prf["precision"])),
                        fmt(float(prf["recall"])),
                        fmt(float(prf["f1"])),
                        prf["support"],
                    ]
                )

            for bucket, stats in sorted(
                summary["by_length"].items()
            ):
                per_length_rows.append(
                    [
                        testset,
                        model_name,
                        bucket,
                        stats["n"],
                        fmt(float(stats["accuracy"])),
                        fmt(float(stats["macro_f1"])),
                    ]
                )

            for group, stats in sorted(
                summary["by_group"].items()
            ):
                per_group_rows.append(
                    [
                        testset,
                        model_name,
                        group,
                        stats["n"],
                        fmt(float(stats["recall"])),
                    ]
                )

            labels = summary["labels"]
            confusion = summary["confusion"]

            confusion_rows = [
                [labels[row_index]]
                + [int(value) for value in confusion[row_index]]
                for row_index in range(len(labels))
            ]

            write_tsv(
                out_dir
                / f"confusion_{testset}_{model_name}.tsv",
                ["gold\\pred"] + labels,
                confusion_rows,
            )

    write_tsv(
        out_dir / "metrics_overall.tsv",
        [
            "testset",
            "model",
            "n",
            "accuracy",
            "macro_f1",
            "mcc",
        ],
        overall_rows,
    )

    write_tsv(
        out_dir / "per_class.tsv",
        [
            "testset",
            "model",
            "class",
            "precision",
            "recall",
            "f1",
            "support",
        ],
        per_class_rows,
    )

    write_tsv(
        out_dir / "per_length.tsv",
        [
            "testset",
            "model",
            "length_bucket",
            "n",
            "accuracy",
            "macro_f1",
        ],
        per_length_rows,
    )

    write_tsv(
        out_dir / "per_group.tsv",
        [
            "testset",
            "model",
            "group",
            "n",
            "recall",
        ],
        per_group_rows,
    )

    lines = [
        "# Tiara benchmark summary",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for testset, models in results.items():
        lines.extend(
            [
                f"## Test set: {testset}",
                "",
                "| Model | N | Accuracy | Macro-F1 | MCC |",
                "|---|---:|---:|---:|---:|",
            ]
        )

        for model_name, summary in models.items():
            lines.append(
                f"| {model_name} "
                f"| {summary['n']} "
                f"| {fmt(summary['accuracy'])} "
                f"| {fmt(summary['macro_f1'])} "
                f"| {fmt(summary['mcc'])} |"
            )

        lines.extend(
            [
                "",
                "### Per-class F1",
                "",
                "| Class | "
                + " | ".join(models.keys())
                + " |",
                "|---|"
                + "|".join(["---:"] * len(models))
                + "|",
            ]
        )

        for class_name in GOLD_CLASSES:
            cells = [
                fmt(
                    float(
                        models[model_name]["per_class"]
                        [class_name]["f1"]
                    )
                )
                for model_name in models
            ]

            lines.append(
                f"| {class_name} | "
                + " | ".join(cells)
                + " |"
            )

        lines.append("")

    (out_dir / "summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--backend",
        choices=["inprocess"],
        default=None,
        help="Streaming version currently supports inprocess only.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Examples: cpu, cuda:2",
    )

    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model subset.",
    )

    parser.add_argument(
        "--testsets",
        default=None,
        help="Comma-separated test-set subset.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = load_config(args.config)
    base = args.config.resolve().parent

    backend = args.backend or cfg.get("backend", "inprocess")

    if backend != "inprocess":
        raise SystemExit(
            "This streaming benchmark currently supports "
            "backend=inprocess only."
        )

    device = args.device or cfg.get("device", "cpu")

    chunk_size = int(cfg.get("chunk_size", 10_000))

    if chunk_size <= 0:
        raise SystemExit(
            "chunk_size must be greater than zero in streaming mode"
        )

    edges = sorted(
        int(value)
        for value in cfg.get(
            "length_buckets",
            [500, 1000, 2000, 3000, 5000, 10000, 20000],
        )
    )

    models = list(cfg["models"])
    testsets = list(cfg["testsets"])

    if args.models:
        requested_models = {
            value.strip()
            for value in args.models.split(",")
            if value.strip()
        }

        models = [
            model
            for model in models
            if model["name"] in requested_models
        ]

    if args.testsets:
        requested_testsets = {
            value.strip()
            for value in args.testsets.split(",")
            if value.strip()
        }

        testsets = [
            testset
            for testset in testsets
            if testset["name"] in requested_testsets
        ]

    if not models:
        raise SystemExit("No matching models were selected.")

    if not testsets:
        raise SystemExit("No matching test sets were selected.")

    results_dir = resolve(
        base,
        cfg.get("results_dir", "results"),
    )

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / timestamp

    print(f"backend    = {backend}")
    print(f"device     = {device}")
    print(f"chunk_size = {chunk_size:,}")
    print(f"threads    = {int(cfg.get('threads', 4))}")
    print(f"models     = {[m['name'] for m in models]}")
    print(f"testsets   = {[t['name'] for t in testsets]}")
    print(f"output     -> {out_dir}")

    if args.dry_run:
        print("\nStreaming dry-run counts:")

        grand_total = 0
        grand_labelled = 0

        for testset in testsets:
            testset_total = 0
            testset_labelled = 0

            print(f"\n  [{testset['name']}]")

            for entry in testset["files"]:
                fasta = resolve(base, entry["path"])

                if not fasta.exists():
                    raise FileNotFoundError(fasta)

                default_gold = (
                    normalise_label(entry["gold"])
                    if entry.get("gold")
                    else None
                )

                total, labelled = count_labelled_records(
                    fasta,
                    default_gold,
                )

                testset_total += total
                testset_labelled += labelled

                print(
                    f"    {fasta.name}: "
                    f"total={total:,}, labelled={labelled:,}"
                )

            grand_total += testset_total
            grand_labelled += testset_labelled

            print(
                f"    subtotal: "
                f"total={testset_total:,}, "
                f"labelled={testset_labelled:,}"
            )

        print(
            f"\ndry-run OK: "
            f"total={grand_total:,}, "
            f"labelled={grand_labelled:,}"
        )

        return 0

    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, dict[str, Any]]] = {}

    save_predictions = bool(
        cfg.get("save_predictions", False)
    )

    for testset in testsets:
        testset_name = testset["name"]

        print(
            f"\n== test set: {testset_name} ==",
            flush=True,
        )

        results[testset_name] = {}

        for model in models:
            model_name = model["name"]

            print(
                f"\n  -- model: {model_name}",
                flush=True,
            )

            classifier = None
            raw_handle: TextIO | None = None

            try:
                classifier = build_classifier(
                    model=model,
                    cfg=cfg,
                    device=device,
                )

                metrics = OnlineMetrics(edges)

                if save_predictions:
                    raw_dir = out_dir / "raw"
                    raw_dir.mkdir(parents=True, exist_ok=True)

                    raw_path = (
                        raw_dir
                        / f"{testset_name}__{model_name}.predictions.tsv.gz"
                    )

                    raw_handle = gzip.open(
                        raw_path,
                        "wt",
                        encoding="utf-8",
                    )

                    raw_handle.write(
                        "\t".join(
                            [
                                "sequence_id",
                                "gold",
                                "stage1",
                                "stage2",
                                "prediction",
                                "length",
                                "group",
                                "source_fasta",
                            ]
                        )
                        + "\n"
                    )

                for entry in testset["files"]:
                    fasta = resolve(base, entry["path"])

                    if not fasta.exists():
                        raise FileNotFoundError(fasta)

                    default_gold = (
                        normalise_label(entry["gold"])
                        if entry.get("gold")
                        else None
                    )

                    fixed_group = (
                        normalise_label(entry["group"])
                        if entry.get("group")
                        else default_gold
                    )

                    process_fasta_streaming(
                        classifier=classifier,
                        fasta=fasta,
                        default_gold=default_gold,
                        fixed_group=fixed_group,
                        metrics=metrics,
                        cfg=cfg,
                        raw_handle=raw_handle,
                    )

                summary = metrics.summary()

                results[testset_name][model_name] = summary

                print(
                    f"     n={summary['n']:,}  "
                    f"acc={summary['accuracy']:.4f}  "
                    f"macroF1={summary['macro_f1']:.4f}  "
                    f"mcc={summary['mcc']:.4f}",
                    flush=True,
                )

            finally:
                if raw_handle is not None:
                    raw_handle.close()

                if classifier is not None:
                    del classifier

                gc.collect()

                if device.startswith("cuda"):
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    write_outputs(results, out_dir)

    print(
        f"\nWrote tables + summary.md to {out_dir}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())