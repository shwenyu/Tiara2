"""Tiara2 base classifier with a length-conditioned BioSignal residual gate."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from tiara.features import TfidfModel, featurize_block
from tiara.io import fasta

STOPS = {"TAA", "TAG", "TGA"}
STARTS = {"ATG", "GTG", "TTG"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _entropy(values) -> float:
    total = float(sum(values))
    return 0.0 if not total else -sum((x / total) * math.log2(x / total) for x in values if x)


def _reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def _frame_features(seq: str, offset: int) -> list[float]:
    valid = [
        seq[i : i + 3]
        for i in range(offset, len(seq) - 2, 3)
        if set(seq[i : i + 3]) <= set("ACGT")
    ]
    n = max(len(valid), 1)
    stops = [i for i, codon in enumerate(valid) if codon in STOPS]
    segments, previous = [], -1
    for position in stops + [len(valid)]:
        segments.append(position - previous - 1)
        previous = position
    longest = max(segments, default=0)
    mean_orf = sum(segments) / max(len(segments), 1)
    p90 = float(np.percentile(segments, 90)) if segments else 0.0
    return [
        len(stops) / n,
        sum(codon in STARTS for codon in valid) / n,
        longest * 3 / max(len(seq), 1),
        mean_orf * 3 / max(len(seq), 1),
        p90 * 3 / max(len(seq), 1),
        sum(x >= 100 for x in segments) / max(n / 100.0, 1.0),
    ]


def _match_fraction(seq: str, lag: int) -> float:
    valid = matches = 0
    for a, b in zip(seq[:-lag], seq[lag:]):
        if a in "ACGT" and b in "ACGT":
            valid += 1
            matches += a == b
    return matches / max(valid, 1)


def biosignal_features(seq: str) -> np.ndarray:
    length = len(seq)
    counts = [seq.count(base) for base in "ACGTN"]
    acgt = max(sum(counts[:4]), 1)
    max_run = run = 0
    previous = ""
    for char in seq:
        run = run + 1 if char == previous else 1
        previous = char
        max_run = max(max_run, run)
    tri = Counter(seq[i : i + 3] for i in range(max(length - 2, 0)))
    vector = [
        math.log1p(length),
        (counts[1] + counts[2]) / acgt,
        counts[4] / max(length, 1),
        _entropy(counts[:4]) / 2.0,
        _entropy(tri.values()) / 6.0,
        max_run / max(length, 1),
    ]
    rows = []
    for strand in (seq, _reverse_complement(seq)):
        for offset in range(3):
            row = _frame_features(strand, offset)
            rows.append(row)
            vector.extend(row[:4])
    matrix = np.asarray([row[:4] for row in rows], dtype=np.float32)
    for column in range(4):
        values = matrix[:, column]
        vector.extend([float(values.min()), float(values.max()), float(values.std())])
    for row in rows:
        vector.extend(row[4:])
    phases = []
    for offset in range(3):
        phase = seq[offset::3]
        valid = sum(phase.count(base) for base in "ACGT")
        phases.append((phase.count("G") + phase.count("C")) / max(valid, 1))
    vector.extend(phases)
    vector.extend([float(np.std(phases)), float(max(phases) - min(phases))])
    vector.append(_match_fraction(seq, 3) - 0.5 * (_match_fraction(seq, 1) + _match_fraction(seq, 2)))
    result = np.asarray(vector, dtype=np.float32)
    if result.shape != (60,):
        raise RuntimeError(f"unexpected BioSignal feature shape: {result.shape}")
    return result


def apply_eukaryote_threshold(
    root_probability: np.ndarray,
    root_classes: list[str],
    threshold: float | None,
) -> np.ndarray:
    """Return root choices, optionally replacing argmax with an euk-vs-rest threshold."""
    values = np.asarray(root_probability)
    if values.ndim != 2 or values.shape[1] != len(root_classes):
        raise ValueError("root_probability shape does not match root_classes")
    try:
        euk_index = root_classes.index("euk_nuclear")
    except ValueError as error:
        raise ValueError("model root schema does not contain euk_nuclear") from error
    if threshold is None:
        return values.argmax(1)
    threshold = float(threshold)
    if not 0 <= threshold <= 1:
        raise ValueError("euk_threshold must be in [0, 1]")
    non_euk_indices = np.asarray(
        [index for index in range(len(root_classes)) if index != euk_index],
        dtype=np.int64,
    )
    if not len(non_euk_indices):
        raise ValueError("model root schema must contain at least one non-eukaryote class")
    fallback = non_euk_indices[values[:, non_euk_indices].argmax(1)]
    return np.where(values[:, euk_index] >= threshold, euk_index, fallback)


def classify(
    bundle,
    input_fasta,
    output,
    batch=512,
    device=None,
    min_len=1000,
    max_records=None,
    router_overrides=None,
    threshold_overrides=None,
    euk_threshold=None,
):
    """Classify FASTA records through one stable interface and automatic router."""
    import joblib
    import torch

    from tiara.hierarchical.model import HierarchicalClassifier, probabilities
    from tiara.hierarchical.schema import HierarchySchema

    bundle_path = Path(bundle).resolve()
    manifest = json.loads(bundle_path.read_text())
    if manifest.get("format") != "tiara2-biosignal-residual-v1":
        raise ValueError("invalid Tiara2 model manifest")

    def resolve(value):
        path = Path(value)
        return path if path.is_absolute() else (bundle_path.parent / path).resolve()

    base_path = resolve(manifest["base"]["checkpoint"])
    expert_path = resolve(manifest["expert"]["model"])
    for name, path in (("base", base_path), ("expert", expert_path)):
        if _sha256(path) != manifest[name]["sha256"]:
            raise ValueError(f"{name} model hash mismatch")

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(base_path, map_location=dev, weights_only=False)
    schema = HierarchySchema.from_dict(checkpoint["schema"])
    model = HierarchicalClassifier(**checkpoint["model"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(dev).eval()
    expert = joblib.load(expert_path)
    if int(getattr(expert, "n_features_in_", -1)) != 60:
        raise ValueError("BioSignal expert feature dimension mismatch")
    tfidf = TfidfModel.load(resolve(manifest["tfidf"]))

    router = {**manifest["router"], **(router_overrides or {})}
    thresholds = {**manifest.get("thresholds", {}), **(threshold_overrides or {})}
    temperatures = manifest.get("temperatures")
    lower, upper = (int(x) for x in router["length_bins_bp"])
    promoted_total = 0
    accepted_total = 0

    root_classes = list(schema.profile.root)
    try:
        euk_index = root_classes.index("euk_nuclear")
        prok_index = root_classes.index("prok")
        organelle_index = root_classes.index("organelle")
    except ValueError as error:
        raise ValueError("model root schema must contain euk_nuclear, prok and organelle") from error
    if euk_threshold is not None:
        euk_threshold = float(euk_threshold)
        if not 0 <= euk_threshold <= 1:
            raise ValueError("euk_threshold must be in [0, 1]")

    def flush(records, writer):
        nonlocal promoted_total
        if not records:
            return
        sequences = [seq for _, seq in records]
        matrix = featurize_block(sequences, tfidf.k, tfidf.idfs)
        with torch.inference_mode():
            logits = model(torch.from_numpy(matrix).to(dev))
            base_probs = probabilities(logits, temperatures)
        root_probability = base_probs["root"].detach().cpu().numpy()
        expert_probability = expert.predict_proba(
            np.stack([biosignal_features(seq) for seq in sequences])
        )[:, 1]
        p = np.clip(expert_probability, 1e-7, 1 - 1e-7)
        evidence = np.maximum(
            0.0,
            np.log(p / (1 - p)) - np.log(router["tau"] / (1 - router["tau"])),
        )
        lengths = np.asarray([len(seq) for seq in sequences])
        weights = np.ones(len(lengths), dtype=np.float32)
        weights[(lengths >= lower) & (lengths < upper)] = router["middle_weight"]
        weights[lengths >= upper] = router["long_weight"]
        boost = np.minimum(router["alpha"] * weights * evidence, router["delta_max"])
        eligible = root_probability[:, euk_index] >= router["base_euk_min"]
        adjusted_logits = np.log(np.clip(root_probability, 1e-12, 1.0))
        adjusted_logits[:, euk_index] += boost * eligible
        adjusted_root = np.exp(adjusted_logits - adjusted_logits.max(1, keepdims=True))
        adjusted_root /= adjusted_root.sum(1, keepdims=True)
        base_choice = root_probability.argmax(1)
        router_choice = adjusted_root.argmax(1)
        promoted = (base_choice != euk_index) & (router_choice == euk_index) & eligible
        adjusted_choice = apply_eukaryote_threshold(adjusted_root, root_classes, euk_threshold)
        promoted_effective = promoted & (adjusted_choice == euk_index)
        promoted_total += int(promoted_effective.sum())

        for i, (header, seq) in enumerate(records):
            root_index = int(adjusted_choice[i])
            root = schema.profile.root[root_index]
            root_p = float(adjusted_root[i, root_index])
            leaf, leaf_p = root, root_p
            branch = {"euk_nuclear": "euk", "prok": "prok", "organelle": "organelle"}.get(root)
            if branch:
                branch_probs = base_probs[branch][i]
                leaf_index = int(branch_probs.argmax())
                leaf = schema.classes(branch)[leaf_index]
                leaf_p = float(branch_probs[leaf_index])
            if root_p < float(thresholds.get("root", 0)) or (
                branch and leaf_p < float(thresholds.get(branch, 0))
            ):
                leaf = "unknown"
            writer.writerow(
                [
                    header,
                    len(seq),
                    "base+biosignal" if promoted_effective[i] else "base",
                    root,
                    leaf,
                    f"{root_p:.8f}",
                    f"{leaf_p:.8f}",
                    f"{adjusted_root[i, euk_index]:.8f}",
                    f"{adjusted_root[i, prok_index]:.8f}",
                    f"{adjusted_root[i, organelle_index]:.8f}",
                ]
            )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "record_id",
                "length_bp",
                "expert",
                "root",
                "leaf",
                "root_probability",
                "leaf_probability",
                "p_eukaryote",
                "p_prokaryote",
                "p_organelle",
            ]
        )
        records = []
        for header, sequence in fasta(Path(input_fasta)):
            if len(sequence) < min_len:
                continue
            records.append((header, sequence))
            accepted_total += 1
            if len(records) >= batch:
                flush(records, writer)
                records = []
            if max_records is not None and accepted_total >= max_records:
                break
        flush(records, writer)
    return {
        "input": str(Path(input_fasta).resolve()),
        "output": str(output_path.resolve()),
        "accepted_records": accepted_total,
        "promoted_records": promoted_total,
        "device": str(dev),
        "eukaryote_threshold": euk_threshold,
    }
