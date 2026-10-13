from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# ============================================================
# PROJECT PATHS
# ============================================================

# Current working directory:
# ASR\
ROOT = Path.cwd()

# Parent:
# ...\manipuri_stt\
ASR = ROOT.parent

# Exact location of B3 script:
# ASR\manipuri_asr_v2\
B3_SCRIPT_DIR = ROOT / "manipuri_asr_v2"

if not B3_SCRIPT_DIR.exists():
    raise FileNotFoundError(
        f"B3 script directory not found:\n{B3_SCRIPT_DIR}"
    )


# ============================================================
# IMPORT B3 SCRIPT
# ============================================================

# This is the critical fix.
# Python will now search inside:
# ASR\manipuri_asr_v2\
sys.path.insert(
    0,
    str(B3_SCRIPT_DIR)
)

import train_b3_narrow_E11_E12 as b3


# ============================================================
# FILE PATHS
# ============================================================

VAL_CSV = (
    ASR
    / "combined_dataset"
    / "val.csv"
)

B3_CHECKPOINT = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "best.pt"
)

CACHE_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "val_logit_cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOGITS_PATH = CACHE_DIR / "logits.npy"
OFFSETS_PATH = CACHE_DIR / "offsets.npy"
TEXTS_PATH = CACHE_DIR / "texts.json"
META_PATH = CACHE_DIR / "metadata.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("B3 VALIDATION LOGIT CACHE")
print("=" * 70)

print(f"[ROOT]       {ROOT}")
print(f"[B3 SCRIPT]  {B3_SCRIPT_DIR}")
print(f"[VAL CSV]    {VAL_CSV}")
print(f"[CHECKPOINT] {B3_CHECKPOINT}")
print(f"[CACHE DIR]  {CACHE_DIR}")

print()
print(f"[DEVICE] {DEVICE}")


if DEVICE.type == "cuda":

    print(
        f"[GPU] "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"[VRAM] "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

    print(
        f"[BF16] "
        f"{torch.cuda.is_bf16_supported()}"
    )


# ============================================================
# VERIFY FILES
# ============================================================

print("\n[CHECK] Verifying required files...")

if not VAL_CSV.exists():

    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )

if not B3_CHECKPOINT.exists():

    raise FileNotFoundError(
        f"B3 checkpoint not found:\n{B3_CHECKPOINT}"
    )

print("[CHECK] Validation CSV: PASS")
print("[CHECK] B3 checkpoint: PASS")


# ============================================================
# LOAD VALIDATION CSV
# ============================================================

print("\n[DATA] Loading validation CSV...")

val_df = pd.read_csv(
    VAL_CSV
)

print(
    f"[DATA] Validation rows: "
    f"{len(val_df):,}"
)

if len(val_df) != 864:

    raise RuntimeError(
        f"Expected 864 validation rows, "
        f"found {len(val_df)}"
    )


# ============================================================
# LOAD VOCAB
# ============================================================

print("\n[VOCAB] Loading vocabulary...")

vocab = b3.load_vocab()

print(
    f"[VOCAB] Size: "
    f"{len(vocab)}"
)

print(
    f"[VOCAB] Blank ID: "
    f"{vocab[b3.SPECIAL_BLANK]}"
)

print(
    f"[VOCAB] Unknown ID: "
    f"{vocab[b3.SPECIAL_UNK]}"
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\n[MODEL] Creating B3 model...")

model = b3.B2Model(
    vocab_size=len(vocab)
)

print("[MODEL] Model created")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\n[MODEL] Loading B3 E11 best checkpoint...")

checkpoint = torch.load(
    B3_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

if not isinstance(checkpoint, dict):

    raise RuntimeError(
        "Unexpected checkpoint format."
    )

if "model_state_dict" not in checkpoint:

    raise RuntimeError(
        "B3 checkpoint does not contain "
        "'model_state_dict'."
    )


missing, unexpected = model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=False
)


if missing:

    print("\n[ERROR] Missing checkpoint keys:")

    for key in missing:
        print("   ", key)

    raise RuntimeError(
        "Missing B3 checkpoint keys."
    )


if unexpected:

    print(
        "\n[WARNING] Unexpected checkpoint keys:"
    )

    for key in unexpected:
        print("   ", key)


print(
    "[MODEL] B3 checkpoint loaded successfully"
)

print(
    f"[MODEL] Epoch: "
    f"{checkpoint.get('epoch')}"
)

print(
    f"[MODEL] Stored Beam CER: "
    f"{checkpoint.get('val_beam500_cer')}"
)

print(
    f"[MODEL] Stored Beam WER: "
    f"{checkpoint.get('val_beam500_wer')}"
)


# ============================================================
# CONFIGURE B3
# ============================================================

print("\n[MODEL] Configuring B3...")

b3.configure_b3(
    model
)

model.eval()
model.to(DEVICE)

print(
    "[MODEL] B3 configuration applied"
)


# ============================================================
# CREATE DATASET
# ============================================================

print("\n[DATASET] Creating validation dataset...")

dataset = b3.AudioCTCDataset(
    val_df,
    vocab
)

print(
    f"[DATASET] Samples: "
    f"{len(dataset):,}"
)


# ============================================================
# INFERENCE
# ============================================================

all_logits = []

offsets = [0]

texts = []

total = len(dataset)


print("\n" + "=" * 70)
print("STARTING VALIDATION LOGIT EXTRACTION")
print("=" * 70)

print(
    f"Processing {total} validation samples..."
)

print(
    "W2V-BERT inference will run ONCE."
)

print(
    "The cached logits will later be reused "
    "for all KenLM alpha/beta experiments."
)

print()


with torch.inference_mode():

    for idx in range(total):

        # ----------------------------------------------------
        # Load sample
        # ----------------------------------------------------

        item = dataset[idx]

        audio = item["audio"]

        text = item["text"]


        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        features = b3.make_features(
            model.processor,
            [audio]
        )


        features = features.to(
            DEVICE,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        use_bf16 = (
            DEVICE.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bf16
        ):

            logits = model(
                features
            )


        # ----------------------------------------------------
        # Log probabilities
        # ----------------------------------------------------

        log_probs = F.log_softmax(
            logits.float(),
            dim=-1
        )


        # ----------------------------------------------------
        # Remove batch dimension
        #
        # [1, T, 60]
        #
        # -> [T, 60]
        # ----------------------------------------------------

        lp = (
            log_probs[0]
            .cpu()
            .numpy()
            .astype(
                np.float32,
                copy=False
            )
        )


        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        all_logits.append(
            lp
        )

        offsets.append(
            offsets[-1]
            + lp.shape[0]
        )

        texts.append(
            text
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (idx + 1) % 10 == 0
            or idx == 0
            or idx + 1 == total
        ):

            print(
                f"[{idx + 1:4d}/{total}] "
                f"frames={lp.shape[0]:4d} "
                f"text_chars={len(text):3d}"
            )


        # ----------------------------------------------------
        # Cleanup
        # ----------------------------------------------------

        del features
        del logits
        del log_probs


        if (
            DEVICE.type == "cuda"
            and (idx + 1) % 50 == 0
        ):

            torch.cuda.empty_cache()


# ============================================================
# CONCATENATE
# ============================================================

print("\n" + "=" * 70)
print("PREPARING CACHE")
print("=" * 70)

print(
    "[WRITE] Concatenating logits..."
)

concatenated = np.concatenate(
    all_logits,
    axis=0
)

offsets = np.asarray(
    offsets,
    dtype=np.int64
)


# ============================================================
# VERIFY
# ============================================================

print(
    f"[WRITE] Total frames: "
    f"{concatenated.shape[0]:,}"
)

print(
    f"[WRITE] Vocabulary dimension: "
    f"{concatenated.shape[1]}"
)

print(
    f"[WRITE] Samples: "
    f"{len(texts):,}"
)

print(
    f"[WRITE] Memory: "
    f"{concatenated.nbytes / 1024**2:.2f} MB"
)


if concatenated.shape[1] != len(vocab):

    raise RuntimeError(
        f"Vocabulary mismatch: "
        f"{concatenated.shape[1]} vs "
        f"{len(vocab)}"
    )


if len(offsets) != len(texts) + 1:

    raise RuntimeError(
        "Offset count mismatch."
    )


# ============================================================
# SAVE LOGITS
# ============================================================

print("\n[WRITE] Saving logits...")

np.save(
    LOGITS_PATH,
    concatenated
)


# ============================================================
# SAVE OFFSETS
# ============================================================

print("[WRITE] Saving offsets...")

np.save(
    OFFSETS_PATH,
    offsets
)


# ============================================================
# SAVE TEXTS
# ============================================================

print("[WRITE] Saving reference texts...")

with open(
    TEXTS_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        texts,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model": "W2V-BERT 2.0",

    "experiment": "B3-NARROW",

    "checkpoint": str(
        B3_CHECKPOINT
    ),

    "epoch": int(
        checkpoint.get(
            "epoch",
            -1
        )
    ),

    "num_samples": len(
        texts
    ),

    "num_frames": int(
        concatenated.shape[0]
    ),

    "vocab_size": int(
        concatenated.shape[1]
    ),

    "normalization":
        "NFC + whitespace collapse",

    "logits_type":
        "log_softmax",

    "dtype":
        "float32",

    "validation_csv":
        str(VAL_CSV),

    "beam_width":
        500,

    "device":
        str(DEVICE),

    "bf16":
        bool(
            DEVICE.type == "cuda"
            and torch.cuda.is_bf16_supported()
        ),

    "checkpoint_beam_cer":
        checkpoint.get(
            "val_beam500_cer"
        ),

    "checkpoint_beam_wer":
        checkpoint.get(
            "val_beam500_wer"
        )
}


print("[WRITE] Saving metadata...")

with open(
    META_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("CACHE COMPLETE")
print("=" * 70)

print(
    f"Logits   : {LOGITS_PATH}"
)

print(
    f"Offsets  : {OFFSETS_PATH}"
)

print(
    f"Texts    : {TEXTS_PATH}"
)

print(
    f"Metadata : {META_PATH}"
)

print()
print(
    f"Samples  : {len(texts):,}"
)

print(
    f"Frames   : {concatenated.shape[0]:,}"
)

print(
    f"Vocab    : {concatenated.shape[1]}"
)

print(
    f"Memory   : "
    f"{concatenated.nbytes / 1024**2:.2f} MB"
)

print()
print(
    "[SUCCESS] B3 validation logits cached."
)

print(
    "[NEXT] Ready for KenLM alpha/beta sweep."
)