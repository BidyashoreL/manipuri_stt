from pathlib import Path
import json
import re
import unicodedata

import numpy as np
from pyctcdecode import build_ctcdecoder


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

CACHE_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "val_logit_cache"
)

LOGITS_PATH = CACHE_DIR / "logits.npy"
OFFSETS_PATH = CACHE_DIR / "offsets.npy"
TEXTS_PATH = CACHE_DIR / "texts.json"

VOCAB_PATH = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

UNIGRAM_PATH = (
    ASR
    / "manipuri_asr_v2"
    / "kenlm_experiment"
    / "manipuri_unigrams.txt"
)

KENLM_BIN = (
    ASR
    / "manipuri_asr_v2"
    / "kenlm_experiment"
    / "manipuri_4gram.bin"
)

OUTPUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = OUTPUT_DIR / "b3_4gram_nbest50_val.json"


# ============================================================
# SETTINGS
# ============================================================

BEAM_WIDTH = 500
N_BEST = 50

# Best 4-gram configurations discovered earlier
ALPHA = 0.5
BETA = 1.0

# Set this to None to process all 864 samples.
# For the first test, use 3.
TEST_SAMPLES = None


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize(
        "NFC",
        str(text)
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("B3 + 4-GRAM KENLM N-BEST GENERATION")
print("=" * 70)

required_files = [
    LOGITS_PATH,
    OFFSETS_PATH,
    TEXTS_PATH,
    VOCAB_PATH,
    UNIGRAM_PATH,
    KENLM_BIN,
]

print("\n[CHECK] Required files:")

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(f"[OK] {path}")


# ============================================================
# LOAD LOGITS
# ============================================================

print("\n[LOAD] Loading cached B3 validation logits...")

logits = np.load(
    LOGITS_PATH,
    mmap_mode="r"
)

offsets = np.load(
    OFFSETS_PATH
)

with open(
    TEXTS_PATH,
    "r",
    encoding="utf-8"
) as f:
    references = json.load(f)

references = [
    normalize_text(x)
    for x in references
]

print(f"[LOAD] Logits shape : {logits.shape}")
print(f"[LOAD] Samples      : {len(references):,}")

if len(references) != 864:
    raise RuntimeError(
        f"Expected 864 validation samples, "
        f"found {len(references)}"
    )


# ============================================================
# LOAD VOCAB
# ============================================================

print("\n[VOCAB] Loading vocabulary...")

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as f:
    vocab = json.load(f)

labels = [
    token
    for token, idx
    in sorted(
        vocab.items(),
        key=lambda x: x[1]
    )
]

# pyctcdecode expects blank as ""
labels = [
    ""
    if token == "<blank>"
    else token
    for token in labels
]

print(f"[VOCAB] Labels: {len(labels)}")


# ============================================================
# LOAD UNIGRAMS
# ============================================================

print("\n[LM] Loading unigram list...")

with open(
    UNIGRAM_PATH,
    "r",
    encoding="utf-8"
) as f:
    unigrams = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(f"[LM] Unigrams: {len(unigrams):,}")


# ============================================================
# BUILD DECODER
# ============================================================

print("\n[DECODER] Building 4-gram decoder...")

print(f"  Alpha      : {ALPHA}")
print(f"  Beta       : {BETA}")
print(f"  Beam width : {BEAM_WIDTH}")
print(f"  N-best     : {N_BEST}")
print(f"  KenLM      : {KENLM_BIN.name}")

decoder = build_ctcdecoder(
    labels=labels,
    kenlm_model_path=str(KENLM_BIN),
    unigrams=unigrams,
    alpha=ALPHA,
    beta=BETA,
)

print("[DECODER] Ready.")


# ============================================================
# N-BEST DECODING
# ============================================================

def decode_nbest(lp):
    """
    Decode one utterance and return N-best hypotheses.

    pyctcdecode in the current environment returns tuples:

        (
            text,
            kenlm_state,
            timesteps,
            lm_score,
            logit_score
        )
    """

    beams = decoder.decode_beams(
        lp,
        beam_width=BEAM_WIDTH,
        beam_prune_logp=-10.0,
        token_min_logp=-5.0,
        hotwords=None,
        hotword_weight=10.0,
    )

    results = []

    for rank, beam in enumerate(
        beams[:N_BEST],
        start=1
    ):

        # ----------------------------------------------------
        # Current pyctcdecode tuple format
        # ----------------------------------------------------

        text = beam[0]
        lm_score = beam[3]
        logit_score = beam[4]

        text = normalize_text(text)

        # IMPORTANT:
        # Keep the two scores separately.
        # Do not invent a new total score here because
        # pyctcdecode already determines the beam ranking.
        #
        # We will later use these values for neural LM
        # rescoring.

        results.append(
            {
                "rank": rank,
                "text": text,
                "lm_score": float(lm_score),
                "logit_score": float(logit_score),
            }
        )

    return results

# ============================================================
# PROCESS VALIDATION SET
# ============================================================

num_samples = (
    len(references)
    if TEST_SAMPLES is None
    else min(
        TEST_SAMPLES,
        len(references)
    )
)

print("\n" + "=" * 70)
print(
    f"DECODING {num_samples} SAMPLE(S)"
)
print("=" * 70)


all_results = []

for i in range(num_samples):

    start = int(
        offsets[i]
    )

    end = int(
        offsets[i + 1]
    )

    lp = logits[
        start:end
    ]

    print(
        f"\n[{i + 1}/{num_samples}] "
        f"frames={end-start}"
    )

    nbest = decode_nbest(lp)

    sample = {
        "sample_index": i,
        "reference": references[i],
        "hypotheses": nbest,
    }

    all_results.append(
        sample
    )

    print(
        f"REFERENCE:\n"
        f"  {references[i]}"
    )

    print("\nTOP HYPOTHESES:")

    for hyp in nbest[:10]:

        print(
             f"  {hyp['rank']:2d}. "
             f"{hyp['text']}"
        )

        print(
             f"      "
             f"lm={hyp['lm_score']:.4f} "
             f"acoustic={hyp['logit_score']:.4f}"
       )


# ============================================================
# SAVE
# ============================================================

output = {
    "model": "B3-NARROW",
    "decoder": "KenLM-4gram",
    "beam_width": BEAM_WIDTH,
    "n_best": N_BEST,
    "alpha": ALPHA,
    "beta": BETA,
    "num_samples": num_samples,
    "total_validation_samples": len(references),
    "samples": all_results,
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    f"Saved N-best results to:\n"
    f"{OUTPUT_JSON}"
)

print(
    f"\nSamples processed: {num_samples}"
)

print(
    f"N-best per sample: {N_BEST}"
)