from pathlib import Path
import json

from pyctcdecode import build_ctcdecoder


# ============================================================
# PATHS
# ============================================================

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
# LOAD VOCABULARY
# ============================================================

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

print("=" * 70)
print("CTC DECODER CONSTRUCTION TEST")
print("=" * 70)

print(f"[INFO] Vocabulary size: {len(vocab)}")

if len(vocab) != 60:
    raise RuntimeError(
        f"Expected 60 vocabulary entries, got {len(vocab)}"
    )


# IMPORTANT:
# Sort by CTC ID so decoder label position exactly matches
# the model's output dimension.

id_to_token = {
    idx: token
    for token, idx in vocab.items()
}

labels = [
    id_to_token[i]
    for i in range(len(id_to_token))
]


print()
print("[INFO] First 10 CTC labels:")

for i, label in enumerate(labels[:10]):
    print(f"  {i}: {repr(label)}")

print()
print("[INFO] Last 10 CTC labels:")

for i, label in enumerate(labels[-10:], start=len(labels) - 10):
    print(f"  {i}: {repr(label)}")


# ============================================================
# LOAD UNIGRAMS
# ============================================================

with open(UNIGRAM_PATH, "r", encoding="utf-8") as f:
    unigrams = [
        line.rstrip("\n")
        for line in f
        if line.strip()
    ]

print()
print(f"[INFO] Unigrams: {len(unigrams):,}")


# ============================================================
# BUILD DECODER
# ============================================================

print()
print("[INFO] Building pyctcdecode decoder...")

decoder = build_ctcdecoder(
    labels=labels,
    unigrams=unigrams,
)

print()
print("=" * 70)
print("DECODER CREATED SUCCESSFULLY")
print("=" * 70)

print("[DONE]")