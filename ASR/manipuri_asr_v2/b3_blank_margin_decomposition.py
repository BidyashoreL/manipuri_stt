
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio
import wave
import io
import os
import re
import unicodedata
from functools import lru_cache

# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()
ASR = ROOT.parent

TRAINER_PATH = ROOT / "train_b3_narrow_E11_E12.py"
FORENSICS_SCRIPT = ROOT / "b3_ctc_logit_forensics.py"
VAL_CSV = ASR / "combined_dataset" / "val.csv"
CHECKPOINT = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"
ERROR_CSV = (
    ASR / "checkpoints" / "v3_ctc" / "b3_narrow"
    / "diagnostics" / "sample_diagnostics.csv"
)

OUT_DIR = ROOT / "diagnostics" / "blank_margin_decomposition"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
BLANK_ID = 0
SEVERE_INDICES = {187, 349, 491, 568, 803, 814}

print("[ROOT]", ROOT)
print("[DEVICE]", DEVICE)

for p in [TRAINER_PATH, FORENSICS_SCRIPT, VAL_CSV, CHECKPOINT, ERROR_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")

# ============================================================
# Load only audio helpers from the working forensic script.
# This does not execute its complete analysis.
# ============================================================

source = FORENSICS_SCRIPT.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(FORENSICS_SCRIPT))

helper_names = {
    "decode_pcm",
    "read_parquet_audio_table",
    "load_parquet_audio",
    "normalize_audio",
    "resolve_audio",
    "make_features",
}

helper_nodes = [
    node for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in helper_names
]

found = {node.name for node in helper_nodes}
missing_helpers = helper_names - found

if missing_helpers:
    raise RuntimeError(
        "Missing helper functions in forensic script: "
        + ", ".join(sorted(missing_helpers))
    )

helper_module = ast.Module(body=helper_nodes, type_ignores=[])
exec(compile(helper_module, str(FORENSICS_SCRIPT), "exec"), globals())

print("[AUDIO] Existing forensic helpers loaded.")

# ============================================================
# Load exact model architecture and checkpoint
# ============================================================

spec = ast  # avoid accidental trainer execution before import
import importlib.util

trainer_spec = importlib.util.spec_from_file_location(
    "b3_trainer",
    TRAINER_PATH
)
b3 = importlib.util.module_from_spec(trainer_spec)
trainer_spec.loader.exec_module(b3)

vocab = b3.load_vocab()
model = b3.B2Model(len(vocab))
b3.load_b2_checkpoint(model)
b3.configure_b3(model)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

if isinstance(checkpoint, dict):
    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
else:
    state = checkpoint

missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    raise RuntimeError("Missing checkpoint keys: " + ", ".join(missing))
if unexpected:
    print("[WARNING] Unexpected keys:", unexpected)

model.to(DEVICE)
model.eval()

head = model.ctc_head
weight = head.weight.detach().float()
bias = (
    head.bias.detach().float()
    if head.bias is not None
    else torch.zeros(weight.shape[0], device=weight.device)
)

print("[MODEL] Checkpoint loaded.")

# ============================================================
# Load data and choose normal comparison samples
# ============================================================

val_df = pd.read_csv(VAL_CSV, encoding="utf-8-sig").reset_index(drop=True)
errors = pd.read_csv(ERROR_CSV).drop_duplicates("sample_index").copy()
errors["sample_index"] = errors["sample_index"].astype(int)

if len(val_df) != 864:
    raise RuntimeError(f"Expected 864 validation rows, got {len(val_df)}")

if "beam4g_cer" not in errors.columns:
    raise RuntimeError("beam4g_cer missing from sample diagnostics.")

normal_pool = errors[
    (~errors["sample_index"].isin(SEVERE_INDICES))
    & (errors["beam4g_cer"] <= 0.05)
].sort_values(["beam4g_cer", "sample_index"])

normal_indices = normal_pool["sample_index"].head(12).tolist()
if len(normal_indices) < 5:
    raise RuntimeError("Could not find at least five normal comparison samples.")

selected = sorted(SEVERE_INDICES.union(normal_indices))
print("[SEVERE]", sorted(SEVERE_INDICES))
print("[NORMAL]", normal_indices)

# ============================================================
# Extract logits and decompose blank margin
# ============================================================

records = []

with torch.no_grad():
    for idx in selected:
        row = val_df.iloc[idx]

        audio = resolve_audio(row)
        if not isinstance(audio, torch.Tensor):
            audio = torch.as_tensor(audio, dtype=torch.float32)

        audio = audio.squeeze()
        if audio.ndim != 1 or audio.numel() == 0:
            raise ValueError(
                f"Sample {idx}: invalid waveform shape={tuple(audio.shape)}, "
                f"numel={audio.numel()}"
            )

        features = make_features(model.processor, audio).to(DEVICE)

        hidden = model.encoder(
            input_features=features
        ).last_hidden_state

        adapted = model.adapter(hidden)
        logits = model.ctc_head(adapted)[0].float()
        h = adapted[0].float()

        blank_logits = logits[:, BLANK_ID]
        nonblank_logits = logits[:, 1:]
        best_char_logits, best_local_ids = nonblank_logits.max(dim=-1)
        best_char_ids = best_local_ids + 1

        total_margin = blank_logits - best_char_logits

        weight_contribution = (
            h * (
                weight[BLANK_ID].unsqueeze(0)
                - weight[best_char_ids]
            )
        ).sum(dim=-1)

        bias_contribution = bias[BLANK_ID] - bias[best_char_ids]
        reconstructed = weight_contribution + bias_contribution
        reconstruction_error = (total_margin - reconstructed).abs()

        if reconstruction_error.max().item() > 1e-3:
            raise RuntimeError(
                f"Margin reconstruction failed for sample {idx}: "
                f"{reconstruction_error.max().item()}"
            )

        record = {
            "sample_index": idx,
            "group": "SEVERE" if idx in SEVERE_INDICES else "NORMAL_COMPARISON",
            "speaker_id": str(row.get("speaker_id", "")),
            "source": str(row.get("source", "")),
            "reference_length": len(str(row.get("text", "")).strip()),
            "frames_T": int(logits.shape[0]),
            "blank_argmax_fraction": float((total_margin > 0).float().mean().item()),
            "mean_total_blank_margin": float(total_margin.mean().item()),
            "median_total_blank_margin": float(total_margin.median().item()),
            "mean_weight_contribution": float(weight_contribution.mean().item()),
            "mean_bias_contribution": float(bias_contribution.mean().item()),
            "weight_positive_fraction": float(
                (weight_contribution > 0).float().mean().item()
            ),
            "bias_positive_fraction": float(
                (bias_contribution > 0).float().mean().item()
            ),
            "max_reconstruction_abs_error": float(
                reconstruction_error.max().item()
            ),
        }

        records.append(record)

        print(
            f"[{idx:03d}] {record['group']:<17} "
            f"blank={record['blank_argmax_fraction']:.3f} "
            f"total={record['mean_total_blank_margin']:.3f} "
            f"weight={record['mean_weight_contribution']:.3f} "
            f"bias={record['mean_bias_contribution']:.3f}"
        )

result = pd.DataFrame(records)
merge_cols = [
    c for c in [
        "sample_index", "beam4g_cer", "beam4g_wer",
        "greedy_cer", "greedy_wer", "category"
    ] if c in errors.columns
]
result = result.merge(errors[merge_cols], on="sample_index", how="left")

sample_path = OUT_DIR / "sample_margin_decomposition.csv"
result.to_csv(sample_path, index=False)

metric_cols = [
    c for c in [
        "blank_argmax_fraction", "mean_total_blank_margin",
        "mean_weight_contribution", "mean_bias_contribution",
        "weight_positive_fraction", "bias_positive_fraction",
        "beam4g_cer", "beam4g_wer"
    ] if c in result.columns
]

summary = result.groupby("group")[metric_cols].agg(["count", "mean", "median", "std"])
summary_path = OUT_DIR / "group_margin_decomposition_summary.csv"
summary.to_csv(summary_path)

head_summary = {
    "blank_id": BLANK_ID,
    "vocab_size": int(weight.shape[0]),
    "head_input_dimension": int(weight.shape[1]),
    "blank_bias": float(bias[BLANK_ID].item()),
    "mean_nonblank_bias": float(bias[1:].mean().item()),
    "blank_weight_norm": float(weight[BLANK_ID].norm().item()),
    "mean_nonblank_weight_norm": float(weight[1:].norm(dim=1).mean().item()),
    "severe_indices": sorted(SEVERE_INDICES),
    "normal_comparison_indices": normal_indices,
    "note": (
        "Positive bias contribution alone does not prove causation. "
        "The best nonblank class can vary frame by frame."
    ),
}

with open(OUT_DIR / "head_parameter_summary.json", "w", encoding="utf-8") as f:
    json.dump(head_summary, f, indent=2)

print()
print("=" * 70)
print("DECOMPOSITION COMPLETE")
print("[SAMPLE CSV]", sample_path)
print("[GROUP SUMMARY]", summary_path)
print("[HEAD SUMMARY]", OUT_DIR / "head_parameter_summary.json")
print()
print(result.groupby("group")[metric_cols].mean(numeric_only=True).to_string())
