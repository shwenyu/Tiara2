# Tiara2

Tiara2 builds on the original Tiara tool and retains its efficient k-mer/TF-IDF approach to DNA contig classification. Compared with Tiara, it uses a larger, leakage-audited training corpus, hierarchical eukaryotic/prokaryotic/organelle labels, and a multi-objective base model combined with a length-aware BioSignal residual expert. These changes preserve a lightweight architecture and a single automatic-routing interface while improving eukaryotic and fungal recovery, particularly for short contigs.

## Install

Install the lightweight runtime from conda-forge, then fetch the frozen model once:

```bash
conda create -n tiara2 -c conda-forge tiara2
conda activate tiara2
tiara2-download-models
tiara2-classify --verify
```

The model download is approximately 1.2 GB. It is version-pinned, SHA-256 and
size verified, idempotent when rerun, and stored outside the Conda package in
the user cache. Set `TIARA2_MODEL_HOME` or pass `--model-dir` to select another
location. Tiara2 never downloads model files merely because an environment was
activated.

For a source installation, Git LFS is required:

```bash
git lfs install
git clone https://github.com/shwenyu/Tiara2.git
cd Tiara2
bash scripts/create_env.sh
conda activate tiara2
tiara2-classify --verify
```

`create_env.sh` initializes the clone's LFS checkout, downloads the frozen model,
and then uses the same pinned dependencies as `environment.yml`. It forces only
the declared Conda channels, so a stale user-level mirror cannot change or break
the installation. On a standard Conda setup, `conda env create -f environment.yml`
is equivalent after `git lfs pull`.

## Classify contigs

```bash
tiara2-classify --input contigs.fasta --output predictions.tsv
```

Sequences shorter than 1,000 bp are skipped by default and therefore do not appear in the output.

## Output format

Tiara2 writes one tab-separated row per accepted FASTA record:

```text
record_id  length_bp  expert  root  leaf  root_probability  leaf_probability
contig_1   1820       base+biosignal  euk_nuclear  fungi  0.91240000  0.86410000
```

| Column | Interpretation |
|---|---|
| `record_id` | FASTA header without the leading `>`. |
| `length_bp` | Contig length in base pairs. |
| `expert` | `base+biosignal` means the BioSignal residual changed the root decision to eukaryotic; `base` means the base decision was retained. Routing is automatic. |
| `root` | Broad class: `euk_nuclear`, `prok`, or `organelle`. |
| `leaf` | Detailed class within the selected root, such as `fungi`, `bacteria`, `archaea`, `mitochondria`, or `plastid`. Other eukaryotic groups are also reported. |
| `root_probability` | Probability of the selected root after automatic residual routing. |
| `leaf_probability` | Probability assigned by the corresponding branch head to the selected leaf. |

If custom thresholds are configured and a prediction does not pass them, `leaf` is written as `unknown` while the probabilities are retained for downstream filtering. Root and leaf probabilities describe different hierarchy levels and should not be compared as if they were one flat probability distribution; new cohorts should calibrate thresholds on independent validation data.

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
