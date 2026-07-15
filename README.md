# Tiara v2
[Tiara v2](https://github.com/shwenyu/tiara) is a research fork of the original [Tiara](https://github.com/ibe-uw/tiara), a deep-learning system for identifying eukaryotic sequences in metagenomic data.
This fork modernizes the training and evaluation workflow, adds GPU-capable training utilities, includes a retrained **v1.1 fixed-hyperparameter baseline**, and provides a streaming benchmark runner for reproducible comparison with the original Tiara models.
> **Project status:** active research and development. The v1.1 models are a controlled retraining baseline, not a final Tiara v2 release.
## Overview
Tiara uses a two-stage classification pipeline:
1. **First stage:** classifies sequences into high-level groups, including eukaryotic, bacterial, archaeal, organelle and unknown/prokaryotic predictions.
2. **Second stage:** further classifies sequences predicted as organelle into mitochondria, plastid or unknown.
The core representation remains compatible with the original Tiara design:
~~~plain text
FASTA contig
    ↓
sequence chopping
    ↓
k-mer bag-of-words
    ↓
TF-IDF weighting + L2 normalization
    ↓
Stage 1 neural network
    ↓ organelle only
Stage 2 neural network
    ↓
final sequence-level prediction
~~~
Tiara v2 currently focuses on a controlled question: **can updated training data and a reproducible modern training pipeline improve generalization while preserving the original lightweight inference design?**
## What is included in this fork
- Updated Python packaging and dependency configuration.
- GPU-capable model training through `train_models_gpu.py`.
- Separate hyperparameter-search scripts for the first and second stages.
- Versioned TF-IDF and neural-network model assets.
- Tiara v1.1 NNet parameters trained with the original fixed hyperparameters.
- A unified, bounded-memory benchmark runner.
- CPU and GPU benchmark backends.
- Per-class, per-group and contig-length-stratified evaluation.
- Explicit comparison against the original Tiara model.
## Current model status
The repository contains two main groups of model assets:
~~~plain text
tiara/models/
├── tfidf-models/
└── nnet-models-v1.1/
~~~
### `tfidf-models/`
Contains the TF-IDF parameters used to transform k-mer count vectors before neural-network inference. The TF-IDF model must match the corresponding k-mer size used by the NNet.
### `nnet-models-v1.1/`
Contains seven retrained `.pkl` neural-network parameter files:
- four first-stage models for `k = 4, 5, 6, 7`;
- three second-stage models for `k = 4, 5, 6` or the combinations produced by the current training pipeline;
- filenames encode the k-mer size, hidden-layer dimensions, learning rate, dropout and selected epoch.
The v1.1 baseline uses **Tiara's original fixed hyperparameters** on the updated training data. A completed grid search was not used to select these released v1.1 weights. This distinction is important when interpreting benchmark results.
## Repository structure
~~~plain text
tiara/
├── tiara/
│   ├── src/                         # classification and feature-extraction code
│   ├── training/                    # TF-IDF, search and NNet training scripts
│   └── models/
│       ├── tfidf-models/            # versioned TF-IDF parameters
│       └── nnet-models-v1.1/        # retrained v1.1 NNet parameters
├── benchmark/
│   └── src/
│       ├── 06_benchmark.py          # streaming benchmark runner
│       └── benchmark_config.yaml    # benchmark configuration
├── docs/                            # project documentation
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
~~~
Runtime logs, benchmark result directories, caches, local backups and biological source datasets are intentionally excluded from version control.
## Requirements
The current development workflow has been tested with:
- Python 3.10
- PyTorch 2.x
- NumPy
- Biopython
- scikit-learn
- skorch
- joblib
- numba
- tqdm
- PyYAML
A CUDA-capable GPU is optional. Feature extraction remains CPU-based; CUDA accelerates neural-network inference and GPU-enabled training paths.
The original Tiara release targeted Python 3.7–3.9 and PyTorch 1.x. This fork updates the environment for modern Python/PyTorch workflows, but compatibility can vary by platform.
## Installation
Clone this fork:
~~~bash
git clone https://github.com/shwenyu/tiara.git
cd tiara
~~~
Create an isolated environment:
~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~
Install the dependencies and the package in editable mode:
~~~bash
pip install -r requirements.txt
pip install -e .
~~~
Verify the installation:
~~~bash
python -c "import tiara; print('Tiara import OK')"
tiara --help
~~~
For the tested GPU environment, verify CUDA separately:
~~~bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY
~~~
## Basic classification
The original Tiara command-line interface remains available:
~~~bash
tiara -i sample_input.fasta -o predictions.tsv
~~~
Common options:
~~~bash
tiara \
  -i sample_input.fasta \
  -o predictions.tsv \
  -t 8 \
  -p 0.65 0.65 \
  --probabilities
~~~
Tiara creates a tab-separated prediction file and a corresponding log file. Sequences should generally be at least 1,000 bp; approximately 3,000 bp or longer is recommended for more reliable classification.
> The versioned `nnet-models-v1.1/` weights are currently selected explicitly by the benchmark configuration. Do not assume that the default `tiara` CLI automatically switches from the original bundled model to the v1.1 model.
## Training data layout
The training scripts expect separate `train`, `validation` and `test` sets:
~~~plain text
train_ready/
├── train/
│   ├── archaea.fasta
│   ├── bacteria.fasta
│   ├── eukarya.fasta
│   ├── mitochondria.fasta
│   └── plastids.fasta
├── validation/
│   └── ...
└── test/
    └── ...
~~~
For Stage 1, mitochondria and plastids are merged into the organelle class. Stage 2 learns the mitochondria-versus-plastid distinction.
The current data-preparation workflow uses genome-level partitioning to reduce leakage between training and validation/test sets. Large intermediate FASTA files and source genomes are not distributed in this repository.
## Training TF-IDF parameters
Train TF-IDF parameters from the prepared training directory:
~~~bash
python -m tiara.training.train_tfidf \
  /path/to/train_ready/train \
  /path/to/output_tfidf_models
~~~
The generated TF-IDF parameters must remain paired with the NNet k-mer configuration used for training and inference.
## Hyperparameter search
First-stage search:
~~~bash
python -m tiara.training.hyperparameter_search_first_stage \
  /path/to/train_ready \
  hp_first_k6.json \
  6 \
  16
~~~
Second-stage search:
~~~bash
python -m tiara.training.hyperparameter_search_second_stage \
  /path/to/train_ready \
  hp_second_k7.json \
  7 \
  16
~~~
Arguments are:
~~~plain text
<training-directory> <output-json> <k-mer-size> <CPU-workers>
~~~
Hyperparameter-search outputs record candidate performance; they are not themselves deployable model weights. A selected configuration must still be used to train and save the final NNet.
## Training the neural networks
Run the GPU-capable training driver:
~~~bash
python -m tiara.training.train_models_gpu \
  /path/to/train_ready_flat \
  /path/to/output_models \
  16
~~~
The training driver produces the first-stage and second-stage `.pkl` model files. Use a stable output directory and record the TF-IDF checksums, software environment and training logs for reproducibility.
On shared systems, limit CPU math-library threads before starting a multi-process search or training run:
~~~bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
~~~
## Benchmarking
The repository provides a unified benchmark runner:
~~~plain text
benchmark/src/06_benchmark.py
~~~
It evaluates every configured model × test-set pair and reports:
- confusion matrices;
- overall accuracy, macro-F1 and MCC;
- per-class precision, recall and F1;
- taxonomic-group-stratified performance;
- contig-length-stratified performance;
- a Markdown summary comparing model variants.
### Validate the configuration
~~~bash
python benchmark/src/06_benchmark.py \
  --config benchmark/src/benchmark_config.yaml \
  --dry-run
~~~
### Run on GPU
~~~bash
python benchmark/src/06_benchmark.py \
  --config benchmark/src/benchmark_config.yaml \
  --device cuda:0
~~~
### Run on CPU
Set the configuration backend/device to CPU, or run:
~~~bash
python benchmark/src/06_benchmark.py \
  --config benchmark/src/benchmark_config.yaml \
  --device cpu
~~~
### Benchmark backends
- `inprocess`: loads explicit TF-IDF and NNet paths; supports GPU inference.
- `cli`: invokes the installed `tiara` command; useful for validating the package-default model on CPU.
Feature extraction—sequence chopping, k-mer counting and TF-IDF transformation—runs on CPU. GPU acceleration applies to the neural-network stages.
### Bounded-memory execution
The benchmark processes FASTA records in configurable chunks rather than loading the complete test set into memory. Relevant configuration fields include:
~~~yaml
backend: inprocess
device: cuda:0
threads: 16
min_len: 3000
prob_cutoff: [0.65, 0.65]
chunk_size: 2000
length_buckets: [500, 1000, 2000, 3000, 5000, 10000, 20000]
~~~
Use `chunk_size` to control peak memory consumption. Temporary files and benchmark outputs should be written to storage with sufficient free space.
### Benchmark outputs
A result directory may contain:
~~~plain text
confusion_holdout_original.tsv
metrics_overall.tsv
per_class.tsv
per_group.tsv
per_length.tsv
summary.md
~~~
Generated result directories and logs are ignored by default because they can be large and may still be changing while an experiment is running.
## Reproducibility checklist
For each released model, record:
- Git commit hash;
- training-data version and split strategy;
- TF-IDF file checksums;
- k-mer sizes for both stages;
- hidden-layer dimensions;
- learning rate and dropout;
- selected epoch;
- Python, PyTorch and skorch versions;
- random seed;
- benchmark configuration and test-set identifier.
Inspect the current revision with:
~~~bash
git rev-parse HEAD
python --version
python -c "import torch, skorch; print(torch.__version__, skorch.__version__)"
~~~
## Relationship to the original Tiara
This project is a fork and experimental extension of the original Tiara. It retains the core two-stage k-mer/TF-IDF/MLP architecture while developing:
- updated training datasets;
- modern Python and PyTorch compatibility;
- reproducible GPU training;
- explicit versioned model parameters;
- bounded-memory evaluation;
- future work on class imbalance, loss functions, calibration and hierarchical classification.
Results produced by this repository should be labeled with the exact model version, for example:
~~~plain text
Original Tiara
Tiara v1.1 fixedHP
Tiara v2 experimental
~~~
Do not describe the current v1.1 fixedHP weights as the final Tiara v2 model.
## Known limitations
- The project is under active development.
- The v1.1 weights use fixed original Tiara hyperparameters rather than a completed global hyperparameter search.
- Pickle-based `.pkl` models remain coupled to compatible Python/skorch/PyTorch code.
- Benchmark quality depends on correct TF-IDF/NNet pairing.
- GPU acceleration does not eliminate the CPU cost of k-mer feature extraction.
- Classification reliability decreases on short or low-complexity contigs.
- Model performance must be validated on independent, leakage-controlled test sets before biological interpretation.
## Security note
Python pickle files can execute code during deserialization. Only load model files from this repository or another trusted source. Never load untrusted `.pkl` files.
## Data availability
Source genomes, generated FASTA fragments, training matrices, logs and intermediate benchmark files are not included in the Git repository. Their provenance and preparation workflow should be documented separately for each experiment.
## Citation
If you use this fork, cite the original Tiara publication:
> Michał Karlicki, Stanisław Antonowicz, Anna Karnkowska. **Tiara: deep learning-based classification system for eukaryotic sequences.** *Bioinformatics*, Volume 38, Issue 2, 15 January 2022, Pages 344–350. [https://doi.org/10.1093/bioinformatics/btab672](https://doi.org/10.1093/bioinformatics/btab672)
Also report the exact Git commit and model version used in your analysis.
## License
Tiara is distributed under the MIT License. This fork retains the original copyright and license notices. See [`LICENSE`](LICENSE) for details.
## Acknowledgements
This work builds on the original Tiara software and publication by Karlicki, Antonowicz and Karnkowska. The v1.1/v2 workflow is an independent research fork and is not an official release of the original Tiara project.
## Issues
For reproducible bug reports, open an issue at:
\<[https://github.com/shwenyu/tiara/issues](https://github.com/shwenyu/tiara/issues)\>
Include the command, configuration, Git commit, environment versions, relevant log excerpt and a minimal non-sensitive input example.