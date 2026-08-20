# Tiara2 model card

## Intended use

Tiara2 classifies assembled DNA contigs of at least 1,000 bp. The root model predicts `euk_nuclear`, `prok` or `organelle`; branch heads refine eukaryotes, prokaryotes and organelles. The eukaryotic head includes fungi, plants, algae, metazoa and major protist groups.

The package is designed for metagenomic contig triage and downstream fungal-contig recovery. It is not a replacement for taxonomic confirmation, gene annotation or MAG quality control.

## Architecture

The frozen k-mer TF-IDF hierarchical classifier is the base model. A 60-feature ExtraTrees BioSignal expert measures coding-frame and sequence-composition signals. A length-conditioned, eukaryote-only soft residual gate combines the two. The expert cannot demote base eukaryotic calls, limiting regression risk.

## Validation summary

The selected model passed independent strict and sens-like-unseen non-regression gates. Relative to the frozen base model, internal root macro-F1 increased by 0.00073 on the strict panel and 0.00102 on sens-like-unseen; eukaryotic and fungal recall also increased on both panels. The calibrated external evaluation improved all eight reported aggregates, with the largest gains on 1,000–2,500 bp contigs.

External aggregate scores used 1% prevalence and length-specific calibrated decision thresholds. The package returns probabilities and frozen argmax labels; threshold selection should be recalibrated for a new study if prevalence or error costs differ. Machine-readable values are in `tiara/models/default/benchmark_summary.json`.

## Frozen artifacts

| Artifact | SHA-256 |
|---|---|
| Base model | `1b227cb5acd192fda07febcdd4a6eafa8cd6894eb695a0ca5ca34de45e98a887` |
| BioSignal expert | `5d8c163f12d00b89004610df780fe14e8ce40f0bddc608197a6c3cffb140c372` |
| TF-IDF model | `2beeb73b339dab892cf68e43afb55194a7a7a4ede5fca6594139af396efa4715` |
| TF-IDF parameters | `7d09059ccbfafbc87364e82216c23c79e960a12c83b1028b795329dd3606ce76` |

`tiara2-classify --verify` recomputes every hash before use.

## Limitations

- Validation does not cover every sequencing platform, assembly method or environment.
- Short and low-complexity contigs remain harder than long contigs.
- Probabilities are not guaranteed to be calibrated for a new prevalence.
- Combine predictions with contamination checks and biological evidence before final interpretation.
