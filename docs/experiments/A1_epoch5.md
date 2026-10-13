# Experiment A1 — Epoch 5 Acoustic Model

## Status

Historical experiment documented from the Manipuri ASR Research
Progress Report dated 10 October 2026.

**Reproduction status:** Not yet independently reproduced.
**Checkpoint provenance:** Needs verification.
**Decoder configuration:** Needs verification.

## Objective

Record the historical A1 acoustic-model result as part of the
low-resource Manipuri/Meiteilon automatic speech recognition
research pipeline.

The recorded acoustic backbone for the project is W2V-BERT 2.0
with a CTC output head. The exact A1 training configuration must
be recovered from the original scripts and logs before attributing
all project-level architecture details to this particular run.

## Reported results

| Variant | Character Error Rate (CER) | Word Error Rate (WER) |
|---|---:|---:|
| A1, epoch 4 | 15.8695% | 59.7270% |
| A1, epoch 5 | 15.9139% | 59.0963% |

Source: Manipuri ASR Research Progress Report, 10 October 2026,
section 5, page 6.

These are historical reported measurements. The available report
does not establish that the epoch-4 and epoch-5 results were
computed using identical checkpoints, decoding settings,
normalization, and evaluation scripts.

## Interpretation

From epoch 4 to epoch 5:

- CER increased by 0.0444 percentage points.
- WER decreased by 0.6307 percentage points.

The two metrics therefore moved in opposite directions.
Neither epoch should be declared universally better from these
two numbers alone.

## Provenance still to recover

- Original A1 training script and configuration.
- Checkpoint corresponding to epoch 5.
- Training log and per-epoch metrics.
- Dataset manifest and split used for evaluation.
- Decoder settings and text-normalization procedure.
- Evaluation script and exact output files.
- Software versions and random seed, if recorded.

## Reproducibility and data policy

Do not fabricate missing logs or reconstruct a checkpoint from
the reported metrics. Preserve original artifacts unchanged.

Keep raw audio, restricted datasets, model checkpoints, cached
features, and large generated outputs out of Git unless their
redistribution is explicitly appropriate. Commit source code,
verified configuration, concise result summaries, and provenance
documentation.

Do not use the official held-out test set for tuning.

## Source limitation

The cited progress report explicitly says the exact A1 checkpoint
and decoder provenance require consolidation. This document
preserves that limitation instead of presenting the experiment
as fully reproducible.
