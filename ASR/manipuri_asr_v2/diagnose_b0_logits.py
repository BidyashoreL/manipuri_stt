import sys
from pathlib import Path
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader

ROOT = Path.cwd()
SCRIPT = ROOT / "ASR" / "manipuri_asr_v2" / "train_ctc_v2.py"

sys.path.insert(0, str(SCRIPT.parent))

# Import the existing implementation.
import train_ctc_v2 as tc

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN = ROOT / "ASR" / "checkpoints" / "v2_ctc" / "b0_5epoch"
BEST = RUN / "best"

print("=" * 70)
print("B0 CTC LOGIT / PREDICTION DIAGNOSTIC")
print("=" * 70)
print("DEVICE:", DEVICE)

# ------------------------------------------------------------------
# Load data exactly as training does
# ------------------------------------------------------------------

all_df = pd.read_csv(ROOT / "ASR" / "combined_dataset" / "all.csv")
train_df = pd.read_csv(ROOT / "ASR" / "combined_dataset" / "train.csv")
val_df = pd.read_csv(ROOT / "ASR" / "combined_dataset" / "val.csv")

train_df, val_df = tc.attach_cache_indices(
    train_df,
    val_df,
    all_df,
)

vocab_path = BEST / "vocab.json"

if not vocab_path.exists():
    vocab_path = RUN / "vocab.json"

print("VOCAB:", vocab_path)

vocab = json.loads(vocab_path.read_text(encoding="utf-8"))

# ------------------------------------------------------------------
# Recreate cache index exactly like training
# ------------------------------------------------------------------

cache_root = ROOT / "ASR" / "feature_cache"
cache_index = {}

for source_dir in cache_root.iterdir():
    if not source_dir.is_dir():
        continue

    source = source_dir.name

    for p in source_dir.glob("*.pt"):
        try:
            manifest_index = int(p.stem)
        except ValueError:
            continue

        cache_index[(source, manifest_index)] = p

print("CACHE ENTRIES:", len(cache_index))

# ------------------------------------------------------------------
# Dataset / loader
# ------------------------------------------------------------------

dataset = tc.CachedCTCDataset(
    val_df,
    cache_index,
    vocab,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=tc.ctc_collate,
)

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

model = tc.CTCASR(
    vocab_size=len(vocab),
    mode="b0",
    bottleneck=256,
).to(DEVICE)

model_path = BEST / "model.pt"

if not model_path.exists():
    # Some runs may use epoch_001 etc.
    candidates = sorted(RUN.glob("epoch_*/model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No model.pt found under {RUN}"
        )
    model_path = candidates[0]

print("MODEL:", model_path)

state = torch.load(
    model_path,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(state)
model.eval()

inv_vocab = {int(v): k for k, v in vocab.items()}
blank_id = int(vocab[tc.SPECIAL_BLANK])
unk_id = int(vocab[tc.SPECIAL_UNK])

print("VOCAB SIZE:", len(vocab))
print("BLANK ID:", blank_id)
print("UNK ID:", unk_id)

# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

pred_counter = torch.zeros(len(vocab), dtype=torch.long)

total_frames = 0
blank_frames = 0
nonblank_frames = 0

max_probs = []
entropy_values = []

sample_count = 0

print("\nRunning validation...")
print()

with torch.no_grad():

    for batch_idx, batch in enumerate(loader):

        features = batch["features"].to(
            DEVICE,
            non_blocking=True,
        )

        logits = model(features)

        probs = torch.softmax(logits.float(), dim=-1)

        max_p, pred = probs.max(dim=-1)

        # Frame-level counts
        flat_pred = pred.detach().cpu().reshape(-1)

        pred_counter += torch.bincount(
            flat_pred,
            minlength=len(vocab),
        )

        total_frames += pred.numel()
        blank_frames += int((pred == blank_id).sum().item())
        nonblank_frames += int((pred != blank_id).sum().item())

        max_probs.extend(
            max_p.detach().cpu().reshape(-1).tolist()
        )

        # Entropy
        entropy = -(probs * torch.log(
            probs.clamp_min(1e-9)
        )).sum(dim=-1)

        entropy_values.extend(
            entropy.detach().cpu().reshape(-1).tolist()
        )

        # First few utterances
        if sample_count < 10:

            for i in range(len(batch["texts"])):

                ids = pred[i].detach().cpu().tolist()

                raw_top = []

                for token_id in ids:
                    raw_top.append(
                        inv_vocab.get(
                            int(token_id),
                            "<unknown-id>"
                        )
                    )

                decoded = tc.decode_ids(
                    ids,
                    vocab,
                )

                print("-" * 70)
                print("REF :", batch["texts"][i])
                print("HYP :", decoded)

                print(
                    "TARGET LEN:",
                    int(batch["target_lengths"][i]),
                )

                print(
                    "INPUT LEN :",
                    int(batch["input_lengths"][i]),
                )

                print(
                    "RAW FRAME TOKENS:",
                    "".join(raw_top[:120]),
                )

                sample_count += 1

                if sample_count >= 10:
                    break

        if sample_count >= 10 and batch_idx >= 20:
            break

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

import numpy as np

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("FRAMES ANALYZED:", total_frames)

print(
    "BLANK FRAMES:",
    blank_frames,
    f"({blank_frames / max(total_frames,1) * 100:.2f}%)",
)

print(
    "NON-BLANK FRAMES:",
    nonblank_frames,
    f"({nonblank_frames / max(total_frames,1) * 100:.2f}%)",
)

print(
    "MEAN MAX PROB:",
    float(np.mean(max_probs)),
)

print(
    "MEDIAN MAX PROB:",
    float(np.median(max_probs)),
)

print(
    "MEAN ENTROPY:",
    float(np.mean(entropy_values)),
)

print("\nTOP FRAME-LEVEL PREDICTIONS:")

top = torch.topk(
    pred_counter,
    min(15, len(vocab)),
)

for count, token_id in zip(
    top.values.tolist(),
    top.indices.tolist(),
):

    token = inv_vocab.get(
        int(token_id),
        "<unknown-id>",
    )

    pct = count / max(total_frames, 1) * 100

    print(
        f"{token!r:12s} "
        f"id={token_id:3d} "
        f"frames={count:8d} "
        f"pct={pct:7.3f}%"
    )

print("\nUNIQUE PREDICTED TOKENS:")
print(
    int((pred_counter > 0).sum().item()),
    "/",
    len(vocab),
)

print("\n" + "=" * 70)
