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

# This script is located at:
# ASR\manipuri_asr_v2\cache_b3_val_logits.py

SCRIPT_DIR = Path(__file__).resolve().parent

# ASR directory:
# ASR\manipuri_asr_v2 -> parent = ASR
ASR = SCRIPT_DIR.parent

# B3 scripts are in the same directory as this script
B3_SCRIPT_DIR = SCRIPT_DIR

if not B3_SCRIPT_DIR.exists():
    raise FileNotFoundError(
        f"B3 script directory not found:\n{B3_SCRIPT_DIR}"
    )

# Make sure Python can import train_b3_narrow_E11_E12.py
if str(B3_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(B3_SCRIPT_DIR))

import train_b3_narrow_E11_E12 as b3

# ============================================================
# FIX PATHS USED INTERNALLY BY THE ORIGINAL B3 SCRIPT
# ============================================================
#
# The original B3 training script assumes it is launched from:
# ASR\manipuri_asr_v2\
#
# We are importing it from:
# ASR\
#
# Therefore, override its ASR-dependent paths here.
# We DO NOT modify the original B3 training script.
#

b3.ROOT = B3_SCRIPT_DIR
b3.ASR = ASR

b3.TRAIN_CSV = (
    ASR
    / "combined_dataset"
    / "train.csv"
)

b3.VAL_CSV = (
    ASR
    / "combined_dataset"
    / "val.csv"
)

b3.B2_CHECKPOINT = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "regularized"
    / "FINAL_B2_MODEL.pt"
)

b3.VOCAB_PATH = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

b3.UNIGRAM_PATH = (
    ASR
    / "manipuri_asr_v2"
    / "kenlm_experiment"
    / "manipuri_unigrams.txt"
)

b3.SAVE_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
)

print("\n[PATCH] B3 internal paths corrected")
print(f"[PATCH] B3 VOCAB      : {b3.VOCAB_PATH}")
print(f"[PATCH] B3 B2 CHECKPT : {b3.B2_CHECKPOINT}")
print(f"[PATCH] B3 TRAIN CSV  : {b3.TRAIN_CSV}")
print(f"[PATCH] B3 VAL CSV    : {b3.VAL_CSV}")
# ============================================================
# PATHS
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
# START
# ============================================================

print("=" * 70)
print("B3 VALIDATION LOGIT CACHE")
print("=" * 70)

print(f"[SCRIPT]     {SCRIPT_DIR}")
print(f"[ASR]        {ASR}")
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
# VERIFY REQUIRED FILES
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
# LOAD VALIDATION DATA
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
        f"\nExpected exactly 864 validation rows.\n"
        f"Found: {len(val_df)}"
    )

print("[DATA] Expected 864 rows: PASS")


# ============================================================
# LOAD VOCAB
# ============================================================

print("\n[VOCAB] Loading vocabulary...")

vocab = b3.load_vocab()

inv_vocab = {
    int(v): k
    for k, v in vocab.items()
}

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
# LOAD B3 CHECKPOINT
# ============================================================

print("\n[MODEL] Loading B3 E11 best checkpoint...")

checkpoint = torch.load(
    B3_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

if not isinstance(checkpoint, dict):
    raise RuntimeError(
        "Unexpected checkpoint format. "
        "Expected a dictionary."
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
        "B3 checkpoint is missing required model keys."
    )

if unexpected:

    print(
        "\n[WARNING] Unexpected checkpoint keys:"
    )

    for key in unexpected:
        print("   ", key)

print("[MODEL] B3 checkpoint loaded successfully")

print(
    f"[MODEL] Epoch: "
    f"{checkpoint.get('epoch')}"
)

print(
    f"[MODEL] Stored Beam-500 CER: "
    f"{checkpoint.get('val_beam500_cer')}"
)

print(
    f"[MODEL] Stored Beam-500 WER: "
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

print("[MODEL] B3 configuration applied")
print("[MODEL] Evaluation mode enabled")


# ============================================================
# DATASET
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

if len(dataset) != 864:
    raise RuntimeError(
        f"Dataset length mismatch: "
        f"{len(dataset)}"
    )


# ============================================================
# INFERENCE SETUP
# ============================================================

all_logits = []

offsets = [0]

texts = []

total = len(dataset)

print("\n" + "=" * 70)
print("STARTING VALIDATION LOGIT EXTRACTION")
print("=" * 70)

print(
    "\nThis will run W2V-BERT once over all "
    f"{total} validation samples."
)

print(
    "The resulting logits will be reused for "
    "all KenLM experiments."
)

print()


# ============================================================
# INFERENCE
# ============================================================

with torch.inference_mode():

    for idx in range(total):

        # ----------------------------------------------------
        # Load one validation sample
        # ----------------------------------------------------

        item = dataset[idx]

        audio = item["audio"]

        text = item["text"]


        # ----------------------------------------------------
        # Convert audio to W2V-BERT input features
        # ----------------------------------------------------

        features = b3.make_features(
            model.processor,
            [audio]
        )


        # ----------------------------------------------------
        # Move to GPU
        # ----------------------------------------------------

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
        # Convert logits to log probabilities
        # ----------------------------------------------------

        log_probs = F.log_softmax(
            logits.float(),
            dim=-1
        )


        # ----------------------------------------------------
        # Remove batch dimension
        #
        # Shape:
        #     [1, T, 60]
        #
        # becomes:
        #     [T, 60]
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
        # Store logits
        # ----------------------------------------------------

        all_logits.append(
            lp
        )


        # ----------------------------------------------------
        # Store offset
        # ----------------------------------------------------

        offsets.append(
            offsets[-1]
            + lp.shape[0]
        )


        # ----------------------------------------------------
        # Store normalized reference text
        # ----------------------------------------------------

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
        # Free temporary tensors
        # ----------------------------------------------------

        del features
        del logits
        del log_probs


        # ----------------------------------------------------
        # Periodic GPU cleanup
        # ----------------------------------------------------

        if (
            DEVICE.type == "cuda"
            and (idx + 1) % 50 == 0
        ):

            torch.cuda.empty_cache()


# ============================================================
# CONCATENATE LOGITS
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
# VERIFY CACHE SHAPE
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
    f"[WRITE] Number of samples: "
    f"{len(texts):,}"
)

print(
    f"[WRITE] Approx memory: "
    f"{concatenated.nbytes / 1024**2:.2f} MB"
)

if concatenated.shape[1] != len(vocab):

    raise RuntimeError(
        "\nVocabulary dimension mismatch!\n"
        f"Logits vocab dimension: {concatenated.shape[1]}\n"
        f"Expected: {len(vocab)}"
    )

if len(offsets) != len(texts) + 1:

    raise RuntimeError(
        "\nOffset count mismatch!\n"
        f"Offsets: {len(offsets)}\n"
        f"Expected: {len(texts) + 1}"
    )


# ============================================================
# SAVE LOGITS
# ============================================================

print("\n[WRITE] Saving logits...")

np.save(
    LOGITS_PATH,
    concatenated
)

print(
    f"[WRITE] Saved:\n"
    f"        {LOGITS_PATH}"
)


# ============================================================
# SAVE OFFSETS
# ============================================================

print("\n[WRITE] Saving offsets...")

np.save(
    OFFSETS_PATH,
    offsets
)

print(
    f"[WRITE] Saved:\n"
    f"        {OFFSETS_PATH}"
)


# ============================================================
# SAVE TEXTS
# ============================================================

print("\n[WRITE] Saving reference texts...")

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

print(
    f"[WRITE] Saved:\n"
    f"        {TEXTS_PATH}"
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model":
        "W2V-BERT 2.0",

    "experiment":
        "B3-NARROW",

    "checkpoint":
        str(B3_CHECKPOINT),

    "epoch":
        int(
            checkpoint.get(
                "epoch",
                -1
            )
        ),

    "num_samples":
        len(texts),

    "num_frames":
        int(
            concatenated.shape[0]
        ),

    "vocab_size":
        int(
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

print(
    f"[WRITE] Saved:\n"
    f"        {META_PATH}"
)


# ============================================================
# FINAL VERIFICATION
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