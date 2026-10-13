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

RESULTS_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "kenlm_sweep"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_JSON = RESULTS_DIR / "unigram_validation_results.json"


# ============================================================
# SETTINGS
# ============================================================

BEAM_WIDTH = 500

# We evaluate the same LM weighting convention used by
# pyctcdecode. These are deliberately kept explicit so the
# result is reproducible.

ALPHA = 0.5
BETAS = [
    -1.0,
    0.0,
    1.0,
    1.5,
    2.0,
]


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
# EDIT DISTANCE
# ============================================================

def edit_distance(reference, hypothesis):

    m = len(reference)
    n = len(hypothesis)

    previous = list(range(n + 1))

    for i in range(1, m + 1):

        current = [i]

        for j in range(1, n + 1):

            cost = (
                0
                if reference[i - 1] == hypothesis[j - 1]
                else 1
            )

            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost
                )
            )

        previous = current

    return previous[n]


# ============================================================
# CER
# ============================================================

def calculate_cer(references, hypotheses):

    errors = 0
    total = 0

    for ref, hyp in zip(
        references,
        hypotheses
    ):

        errors += edit_distance(
            ref,
            hyp
        )

        total += len(ref)

    if total == 0:
        return 0.0

    return errors / total


# ============================================================
# WER
# ============================================================

def calculate_wer(references, hypotheses):

    errors = 0
    total = 0

    for ref, hyp in zip(
        references,
        hypotheses
    ):

        ref_words = ref.split()
        hyp_words = hyp.split()

        errors += edit_distance(
            ref_words,
            hyp_words
        )

        total += len(ref_words)

    if total == 0:
        return 0.0

    return errors / total


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("MANIPURI UNIGRAM VALIDATION")
print("=" * 70)

print("\n[CHECK] Required files...")

for path in [
    LOGITS_PATH,
    OFFSETS_PATH,
    TEXTS_PATH,
    VOCAB_PATH,
    UNIGRAM_PATH,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(
        f"[OK] {path.name}"
    )


# ============================================================
# LOAD LOGITS
# ============================================================

print("\n[LOAD] Loading cached logits...")

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


print(
    f"[LOAD] Logits shape: {logits.shape}"
)

print(
    f"[LOAD] Samples: {len(references):,}"
)

if len(references) != 864:

    raise RuntimeError(
        f"Expected 864 validation samples, "
        f"found {len(references)}"
    )


# ============================================================
# LOAD VOCAB
# ============================================================

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

labels = [
    ""
    if token == "<blank>"
    else token
    for token in labels
]

print(
    f"[VOCAB] Labels: {len(labels)}"
)


# ============================================================
# LOAD UNIGRAMS
# ============================================================

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

print(
    f"[LM] Unigrams: {len(unigrams):,}"
)


# ============================================================
# NORMALIZE REFERENCES
# ============================================================

references = [
    normalize_text(x)
    for x in references
]


# ============================================================
# GREEDY BASELINE
# ============================================================

print("\n" + "=" * 70)
print("GREEDY BASELINE")
print("=" * 70)


def greedy_decode(lp):

    ids = np.argmax(
        lp,
        axis=-1
    )

    output = []

    previous = None

    for token_id in ids:

        token_id = int(token_id)

        if token_id == 0:

            previous = None
            continue

        if token_id == previous:

            continue

        output.append(
            labels[token_id]
        )

        previous = token_id

    return normalize_text(
        "".join(output)
    )


greedy_predictions = []

for i in range(
    len(references)
):

    start = int(offsets[i])
    end = int(offsets[i + 1])

    lp = logits[start:end]

    greedy_predictions.append(
        greedy_decode(lp)
    )


greedy_cer = calculate_cer(
    references,
    greedy_predictions
)

greedy_wer = calculate_wer(
    references,
    greedy_predictions
)

print(
    f"Greedy CER : {greedy_cer * 100:.4f}%"
)

print(
    f"Greedy WER : {greedy_wer * 100:.4f}%"
)


# ============================================================
# UNIGRAM SWEEP
# ============================================================

results = []

for beta in BETAS:

    print("\n" + "=" * 70)
    print("UNIGRAM CONFIGURATION")
    print("=" * 70)

    print(
        f"alpha = {ALPHA}"
    )

    print(
        f"beta  = {beta}"
    )

    print("=" * 70)

    print(
        "[DECODER] Building unigram decoder..."
    )

    decoder = build_ctcdecoder(
        labels=labels,
        unigrams=unigrams,
        alpha=ALPHA,
        beta=beta,
    )

    predictions = []

    for i in range(
        len(references)
    ):

        start = int(offsets[i])
        end = int(offsets[i + 1])

        lp = logits[start:end]

        prediction = decoder.decode(
            lp,
            beam_width=BEAM_WIDTH
        )

        predictions.append(
            normalize_text(prediction)
        )

        if (
            (i + 1) % 100 == 0
            or i + 1 == len(references)
        ):

            print(
                f"[{i + 1:4d}/{len(references)}]"
            )

    cer = calculate_cer(
        references,
        predictions
    )

    wer = calculate_wer(
        references,
        predictions
    )

    delta_cer = cer - greedy_cer
    delta_wer = wer - greedy_wer

    result = {
        "decoder": "unigram",
        "alpha": ALPHA,
        "beta": beta,
        "beam_width": BEAM_WIDTH,
        "cer": cer,
        "wer": wer,
        "cer_percent": cer * 100,
        "wer_percent": wer * 100,
        "delta_vs_greedy_cer": delta_cer * 100,
        "delta_vs_greedy_wer": delta_wer * 100,
    }

    results.append(result)

    print()
    print(
        f"CER : {cer * 100:.4f}%"
    )

    print(
        f"WER : {wer * 100:.4f}%"
    )

    print(
        f"Δ CER vs greedy : "
        f"{delta_cer * 100:+.4f} pp"
    )

    print(
        f"Δ WER vs greedy : "
        f"{delta_wer * 100:+.4f} pp"
    )


# ============================================================
# FIND BEST
# ============================================================

best_cer = min(
    results,
    key=lambda x: x["cer"]
)

best_wer = min(
    results,
    key=lambda x: x["wer"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "settings": {
        "beam_width": BEAM_WIDTH,
        "alpha": ALPHA,
        "betas": BETAS,
        "num_validation_samples": len(references),
        "unigram_file": str(UNIGRAM_PATH),
    },

    "greedy_baseline": {
        "cer": greedy_cer,
        "wer": greedy_wer,
        "cer_percent": greedy_cer * 100,
        "wer_percent": greedy_wer * 100,
    },

    "best_unigram_by_cer": best_cer,

    "best_unigram_by_wer": best_wer,

    "all_results": results,
}


with open(
    RESULTS_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("UNIGRAM VALIDATION COMPLETE")
print("=" * 70)

print()

print(
    f"GREEDY CER : "
    f"{greedy_cer * 100:.4f}%"
)

print(
    f"GREEDY WER : "
    f"{greedy_wer * 100:.4f}%"
)

print("\nBEST UNIGRAM BY CER")

print(
    f"alpha = {best_cer['alpha']}"
)

print(
    f"beta  = {best_cer['beta']}"
)

print(
    f"CER   = "
    f"{best_cer['cer_percent']:.4f}%"
)

print(
    f"Δ CER = "
    f"{best_cer['delta_vs_greedy_cer']:+.4f} pp"
)

print("\nBEST UNIGRAM BY WER")

print(
    f"alpha = {best_wer['alpha']}"
)

print(
    f"beta  = {best_wer['beta']}"
)

print(
    f"WER   = "
    f"{best_wer['wer_percent']:.4f}%"
)

print(
    f"Δ WER = "
    f"{best_wer['delta_vs_greedy_wer']:+.4f} pp"
)

print()

print(
    f"[SAVED] {RESULTS_JSON}"
)