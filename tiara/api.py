"""Stable public Python API for Tiara2."""
from __future__ import annotations

from tiara.bundle import manifest_path, verify_bundle
from tiara.config import load_config


def classify(input_fasta, output, *, config=None, bundle=None, batch=None, device=None, min_length=None, max_records=None):
    settings = load_config(config)
    runtime = settings["runtime"]
    chosen_batch = runtime["batch"] if batch is None else batch
    chosen_min_length = runtime["min_length"] if min_length is None else min_length
    chosen_max_records = runtime["max_records"] if max_records is None else max_records
    if not isinstance(chosen_batch, int) or chosen_batch < 1:
        raise ValueError("batch must be a positive integer")
    if not isinstance(chosen_min_length, int) or chosen_min_length < 1:
        raise ValueError("min_length must be a positive integer")
    if chosen_max_records is not None and (not isinstance(chosen_max_records, int) or chosen_max_records < 1):
        raise ValueError("max_records must be null or a positive integer")
    chosen_bundle = bundle or settings["model_bundle"]
    manifest = manifest_path(chosen_bundle)
    verify_bundle(manifest)
    from tiara.hierarchical.biosignal_multi_expert import classify as classify_biosignal

    return classify_biosignal(
        manifest,
        input_fasta,
        output,
        batch=chosen_batch,
        device=runtime["device"] if device is None else device,
        min_len=chosen_min_length,
        max_records=chosen_max_records,
        router_overrides=settings["router"],
        threshold_overrides=settings["thresholds"],
    )
