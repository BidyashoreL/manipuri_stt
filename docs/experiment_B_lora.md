# Experiment B: LoRA Fine-Tuning for Manipuri ASR

## Status

Completed: 5 training epochs.

## Objective

Evaluate parameter-efficient fine-tuning of the existing B3 W2V-BERT 2.0 + CTC acoustic model using LoRA.

## Configuration

- Base checkpoint: B3 best checkpoint, epoch 11
- Training manifest: `train_16k.csv`
- Training clips: 23,234
- Validation manifest: `validation_16k.csv`
- Validation clips: 1,693
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Target modules: `linear_q`, `linear_k`, `linear_v`, `linear_out`
- Encoder attention layers: 24
- Trainable parameters: 3,736,892
- Trainable parameter fraction: approximately 0.64%
- Optimizer: bitsandbytes AdamW8bit
- Mixed precision: BF16
- Gradient checkpointing: enabled with `use_reentrant=False`
- GPU: NVIDIA GeForce RTX 5050 Laptop GPU

## Validation Results

| Epoch | Training loss | Validation loss | CER (%) | WER (%) |
|---:|---:|---:|---:|---:|
| 1 | 0.2943 | 0.4241 | 16.80 | 51.40 |
| 2 | 0.2128 | 0.4022 | 15.97 | 48.47 |
| 3 | 0.1960 | 0.3850 | 15.98 | 47.34 |
| 4 | 0.1839 | 0.3834 | 15.89 | 46.44 |
| 5 | 0.1751 | 0.3728 | 15.615 | 44.95 |

The best observed validation CER was 15.615% at epoch 5.

## Interpretation and Limitations

Validation loss decreased over the five epochs. Validation CER improved overall, with a small regression at epoch 3.

These results describe this experiment's 1,693-clip validation set. They do not establish an improvement over the original B3 checkpoint because B3 has not yet been evaluated on this exact validation set with the same decoder and normalization.

LoRA gradient-flow verification and a controlled no-LoRA continuation ablation remain outstanding.

The official test set was not used in this experiment. It must remain untouched during model selection.

## Reproducibility

Training script:
`ASR/experiments/experiment_B_lora/train_experiment_B_lora.py`

Diagnostic scripts:
- `ASR/experiments/experiment_B_lora/inspect_lora_targets.py`
- `ASR/experiments/experiment_B_lora/smoke_test_lora.py`

Training history:
`ASR/experiments/experiment_B_lora/history.json`

The original B3 checkpoint, LoRA checkpoints, datasets, and generated validation prediction CSVs are intentionally excluded from this repository commit. They must be preserved locally and handled according to dataset licensing and storage requirements.
