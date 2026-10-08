"""Optional eukaryote-threshold calibration for labeled validation data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


EUK_LABELS = {"euk", "eukarya", "eukaryote", "euk_nuclear", "nuclear_eukaryote"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError(f"TSV has no header: {path}")
        return [dict(row) for row in reader]


def calibrate(predictions: Path, truth: Path) -> dict[str, object]:
    prediction_rows = read_tsv(predictions)
    truth_rows = read_tsv(truth)
    if not prediction_rows:
        raise ValueError("predictions TSV contains no records")
    if not truth_rows:
        raise ValueError("truth TSV contains no records")
    if not {"record_id", "p_eukaryote"}.issubset(prediction_rows[0]):
        raise ValueError("predictions must contain record_id and p_eukaryote")
    if not {"record_id", "truth_class"}.issubset(truth_rows[0]):
        raise ValueError("truth must contain record_id and truth_class")

    scores = {}
    for row in prediction_rows:
        record_id = row["record_id"]
        if record_id in scores:
            raise ValueError(f"duplicate prediction record_id: {record_id}")
        try:
            score = float(row["p_eukaryote"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid p_eukaryote for record_id {record_id}") from error
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"p_eukaryote must be finite and in [0, 1]: {record_id}")
        scores[record_id] = score

    labels = {}
    for row in truth_rows:
        record_id = row["record_id"]
        if record_id in labels:
            raise ValueError(f"duplicate truth record_id: {record_id}")
        truth_class = row["truth_class"].strip().lower()
        if not truth_class:
            raise ValueError(f"blank truth_class for record_id {record_id}")
        labels[record_id] = truth_class in EUK_LABELS
    if set(scores) != set(labels):
        raise ValueError("prediction and truth record_id sets must be identical")

    positives = sum(labels.values())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("validation data must contain both eukaryote and non-eukaryote records")

    best = None
    for threshold in sorted(set(scores.values())):
        tp = fp = fn = tn = 0
        for record_id, score in scores.items():
            observed = labels[record_id]
            predicted = score >= threshold
            tp += int(observed and predicted)
            fp += int(not observed and predicted)
            fn += int(observed and not predicted)
            tn += int(not observed and not predicted)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        row = {
            "threshold": threshold,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "fpr": fp / (fp + tn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        if best is None or (row["f1"], row["threshold"]) > (best["f1"], best["threshold"]):
            best = row

    return {
        "status": "validation_calibration",
        "objective": "maximum_observed_f1",
        "recommended_euk_threshold": best["threshold"],
        "metrics_at_recommended_threshold": best,
        "records": len(labels),
        "positive_records": positives,
        "negative_records": negatives,
        "observed_eukaryote_prevalence": positives / len(labels),
        "predictions_sha256": sha256(predictions),
        "truth_sha256": sha256(truth),
        "warning": "Use an independent labeled validation set. Do not calibrate on the final test set.",
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="tiara2-calibrate-euk",
        description="Choose an eukaryote probability threshold by maximum F1 on optional labeled validation data.",
    )
    parser.add_argument("predictions", type=Path, help="Tiara2 TSV containing record_id and p_eukaryote")
    parser.add_argument("truth", type=Path, help="TSV containing record_id and truth_class")
    args = parser.parse_args(argv)
    print(json.dumps(calibrate(args.predictions, args.truth), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
