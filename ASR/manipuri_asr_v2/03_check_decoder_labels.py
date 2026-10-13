from pathlib import Path
import json

from pyctcdecode import build_ctcdecoder


ASR_DIR = Path(__file__).resolve().parent

VOCAB_PATH = (
    ASR_DIR.parent
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

UNIGRAM_PATH = (
    ASR_DIR
    / "kenlm_experiment"
    / "manipuri_unigrams.txt"
)


# ============================================================
# LOAD VOCAB
# ============================================================

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

if len(vocab) != 60:
    raise RuntimeError(
        f"Expected 60 vocabulary entries, got {len(vocab)}"
    )

id_to_token = {
    idx: token
    for token, idx in vocab.items()
}

raw_labels = [
    id_to_token[i]
    for i in range(len(id_to_token))
]


print("=" * 70)
print("PYCTCDECODE LABEL MAPPING TEST")
print("=" * 70)

print(f"[INFO] Vocabulary size: {len(raw_labels)}")


# ============================================================
# MAP LABELS
# ============================================================
#
# CTC blank:
#   <blank> -> ""
#
# Unknown token:
#   <unk> -> "<unk>"
#
# Everything else stays unchanged.
#
# IMPORTANT:
# Keep all 60 positions unchanged so that
# logits[..., i] corresponds exactly to decoder label i.
# ============================================================

decoder_labels = []

for idx, token in enumerate(raw_labels):

    if token == "<blank>":
        label = ""

    elif token == "<unk>":
        label = "<unk>"

    else:
        label = token

    decoder_labels.append(label)

    print(
        f"{idx:2d}: "
        f"{token!r:12s} -> "
        f"{label!r}"
    )


# ============================================================
# CHECK DUPLICATES
# ============================================================

seen = {}

for idx, label in enumerate(decoder_labels):

    if label in seen:
        raise RuntimeError(
            f"Duplicate decoder label detected:\n"
            f"label={label!r}\n"
            f"IDs={seen[label]} and {idx}"
        )

    seen[label] = idx

print()
print("[PASS] No duplicate decoder labels.")


# ============================================================
# LOAD UNIGRAMS
# ============================================================

with open(UNIGRAM_PATH, "r", encoding="utf-8") as f:
    unigrams = [
        line.rstrip("\n")
        for line in f
        if line.strip()
    ]

print(f"[INFO] Unigram entries: {len(unigrams):,}")


# ============================================================
# BUILD DECODER
# ============================================================

print()
print("[INFO] Building pyctcdecode decoder...")

decoder = build_ctcdecoder(
    labels=decoder_labels,
    unigrams=unigrams,
)


# ============================================================
# SUCCESS
# ============================================================

print()
print("=" * 70)
print("DECODER CONSTRUCTION SUCCESS")
print("=" * 70)

print("[PASS] 60 labels accepted")
print("[PASS] <blank> mapped to empty string")
print("[PASS] <unk> preserved as <unk>")
print(f"[PASS] {len(unigrams):,} Manipuri unigrams loaded")
print("[DONE] Decoder is ready for validation.")