# Tiara2 configuration

Tiara2 accepts one JSON file through `--config`. Start by copying `config/default.json`. Command-line values override the matching runtime values in the file.

## Runtime fields

| Field | Meaning |
|---|---|
| `model_bundle` | Optional compatible bundle. Relative paths are resolved from the configuration file. |
| `runtime.batch` | Number of records processed together. Lower it if GPU or RAM is limited. |
| `runtime.device` | `null` selects CUDA when available and CPU otherwise; `cpu`, `cuda` and `cuda:1` are accepted. |
| `runtime.min_length` | Skip shorter sequences. The validated default is 1,000 bp. |
| `runtime.max_records` | Optional positive record limit; `null` processes every accepted record. |

## Automatic soft router

The base model always runs. The BioSignal expert can only add bounded evidence for the eukaryotic root class; it cannot demote a base-model eukaryotic prediction.

| Field | Meaning |
|---|---|
| `router.tau` | BioSignal probability at which positive residual evidence begins. |
| `router.alpha` | Residual evidence scale. |
| `router.delta_max` | Maximum logit boost per contig. |
| `router.base_euk_min` | Minimum base eukaryotic probability eligible for a boost. |
| `router.length_bins_bp` | Two increasing boundaries defining short, middle and long contigs. |
| `router.middle_weight` | Residual multiplier in the middle bin. |
| `router.long_weight` | Residual multiplier in the long bin. |
| `router.demotion_allowed` | Must remain `false`. |

Omitted router fields retain the frozen model defaults. Changing router fields creates a custom operating point and is not covered by the published benchmark unless it is recalibrated on an independent validation set.

## Output thresholds

`thresholds.root`, `thresholds.euk`, `thresholds.prok` and `thresholds.organelle` accept values from 0 to 1. If the selected probability is below a configured threshold, the leaf label is written as `unknown`; probabilities remain in the TSV. Empty thresholds preserve frozen argmax behavior.

Model-location precedence is `--bundle`, configuration `model_bundle`,
`TIARA2_MODEL_BUNDLE`, the verified user cache created by
`tiara2-download-models`, then a source checkout's bundled model. Set
`TIARA2_MODEL_HOME` to choose a persistent cache location. Explicit runtime CLI
arguments override the JSON file.

## Optional eukaryote decision threshold

`--euk-threshold FLOAT` is separate from the output-confidence thresholds
above. It applies a binary eukaryote-versus-rest decision to the full
`p_eukaryote` root probability. Records below the value fall back to the more
probable non-eukaryotic root. Omitting the option retains the original argmax
decision.

The documented general-purpose preset is:

```bash
tiara2-classify -i contigs.fasta -o predictions.tsv --euk-threshold 0.50
```

Do not substitute `thresholds.euk = 0.50`: that configuration field gates the
selected eukaryotic leaf confidence and has a different meaning. Calibration
is optional and requires representative labels; see the README for the
two-command workflow and the benchmark evidence behind the preset.

Validate a configuration without inference:

```bash
tiara2-classify --config my_config.json --show-config
```
