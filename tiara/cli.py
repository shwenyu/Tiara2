"""Command-line interface for Tiara2."""
from __future__ import annotations

import argparse
import json

from tiara.api import classify
from tiara.bundle import verify_bundle
from tiara.config import load_config


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parser():
    command = argparse.ArgumentParser(prog="tiara2-classify", description="Classify DNA contigs with Tiara2.")
    command.add_argument("-i", "--input", help="input FASTA or FASTA.gz")
    command.add_argument("-o", "--output", help="output TSV")
    command.add_argument("--config", help="JSON configuration; see docs/CONFIGURATION.md")
    command.add_argument("--bundle", help="model directory or model_manifest.json")
    command.add_argument("--batch", type=positive_int, help="records per inference batch")
    command.add_argument("--device", help="torch device, for example cpu or cuda")
    command.add_argument("--min-len", type=positive_int, help="skip shorter sequences")
    command.add_argument("--max-records", type=positive_int, help="optional record limit")
    command.add_argument("--verify", action="store_true", help="verify model hashes and exit")
    command.add_argument("--show-config", action="store_true", help="print the resolved configuration and exit")
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    if args.show_config:
        print(json.dumps(load_config(args.config), indent=2, sort_keys=True))
        return
    if args.verify:
        settings = load_config(args.config)
        print(json.dumps(verify_bundle(args.bundle or settings["model_bundle"]), indent=2, sort_keys=True))
        return
    if not args.input or not args.output:
        command.error("--input and --output are required unless --verify or --show-config is used")
    summary = classify(args.input, args.output, config=args.config, bundle=args.bundle, batch=args.batch, device=args.device, min_length=args.min_len, max_records=args.max_records)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
