"""Load and validate the documented Tiara2 JSON configuration."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

DEFAULT_CONFIG = {
    "model_bundle": None,
    "runtime": {"batch": 512, "device": None, "min_length": 1000, "max_records": None},
    "router": {},
    "thresholds": {},
}
ROUTER_KEYS = {"tau", "alpha", "delta_max", "base_euk_min", "middle_weight", "long_weight", "length_bins_bp", "demotion_allowed"}
THRESHOLD_KEYS = {"root", "euk", "prok", "organelle"}


def _merge(base, update):
    result = deepcopy(base)
    for key, value in update.items():
        if key not in result:
            raise ValueError(f"unknown configuration key: {key}")
        if isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        else:
            result[key] = value
    return result


def load_config(path=None):
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        config_path = Path(path).expanduser().resolve()
        config = _merge(config, json.loads(config_path.read_text()))
        if config["model_bundle"]:
            bundle = Path(config["model_bundle"]).expanduser()
            if not bundle.is_absolute():
                bundle = config_path.parent / bundle
            config["model_bundle"] = str(bundle.resolve())
    validate_config(config)
    return config


def validate_config(config):
    runtime = config["runtime"]
    if set(runtime) != {"batch", "device", "min_length", "max_records"}:
        raise ValueError("runtime must contain batch, device, min_length and max_records")
    if not isinstance(runtime["batch"], int) or runtime["batch"] < 1:
        raise ValueError("runtime.batch must be a positive integer")
    if not isinstance(runtime["min_length"], int) or runtime["min_length"] < 1:
        raise ValueError("runtime.min_length must be a positive integer")
    if runtime["max_records"] is not None and (not isinstance(runtime["max_records"], int) or runtime["max_records"] < 1):
        raise ValueError("runtime.max_records must be null or a positive integer")
    unknown_router = set(config["router"]) - ROUTER_KEYS
    if unknown_router:
        raise ValueError(f"unknown router keys: {sorted(unknown_router)}")
    router = config["router"]
    if "tau" in router and not 0 < float(router["tau"]) < 1:
        raise ValueError("router.tau must be in (0, 1)")
    for key in ("alpha", "delta_max", "middle_weight", "long_weight"):
        if key in router and float(router[key]) < 0:
            raise ValueError(f"router.{key} must be non-negative")
    if "base_euk_min" in router and not 0 <= float(router["base_euk_min"]) <= 1:
        raise ValueError("router.base_euk_min must be in [0, 1]")
    if "length_bins_bp" in router:
        bins = router["length_bins_bp"]
        if len(bins) != 2 or not all(isinstance(x, int) for x in bins) or not 0 < bins[0] < bins[1]:
            raise ValueError("router.length_bins_bp must be two increasing positive integers")
    if router.get("demotion_allowed") not in (None, False):
        raise ValueError("Tiara2 does not support BioSignal demotion")
    unknown_thresholds = set(config["thresholds"]) - THRESHOLD_KEYS
    if unknown_thresholds:
        raise ValueError(f"unknown threshold keys: {sorted(unknown_thresholds)}")
    for key, value in config["thresholds"].items():
        if not 0 <= float(value) <= 1:
            raise ValueError(f"thresholds.{key} must be in [0, 1]")
