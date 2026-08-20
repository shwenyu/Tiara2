# Tiara2

Tiara2 classifies assembled DNA contigs into eukaryotic, prokaryotic and organelle groups, with detailed eukaryotic labels including fungi. A lightweight BioSignal expert is routed automatically; users call one CLI or Python API.

## Install

Git LFS is required because the frozen model is distributed with this repository.

```bash
git clone https://github.com/shwenyu/Tiara2.git
cd Tiara2
git lfs pull
bash scripts/create_env.sh
conda activate tiara2
tiara2-classify --verify
```

`create_env.sh` uses the same pinned dependencies as `environment.yml` and forces only the declared channels, so a stale user-level mirror cannot change or break the installation. On a standard Conda setup, `conda env create -f environment.yml` is equivalent.

## Classify contigs

```bash
tiara2-classify --input contigs.fasta --output predictions.tsv
```

The output contains the record ID, sequence length, selected expert path, root and leaf labels, and probabilities. Sequences shorter than 1,000 bp are skipped by default.

Python usage:

```python
from tiara import classify

classify("contigs.fasta", "predictions.tsv")
```

## Configure

Copy [`config/default.json`](config/default.json), edit it, and run:

```bash
tiara2-classify --config my_config.json -i contigs.fasta -o predictions.tsv
```

All supported runtime, router and threshold fields are documented in [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md). Model scope, validation and artifact hashes are summarized in [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Reproducibility check

```bash
bash scripts/smoke_test.sh
```

Tiara2 is released under the MIT License. Please cite Tiara when using Tiara2 in academic work; project-specific Tiara2 citation information will be added when available.
