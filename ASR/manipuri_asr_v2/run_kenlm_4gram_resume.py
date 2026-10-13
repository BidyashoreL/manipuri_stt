from pathlib import Path
import json
import csv
import re
import unicodedata

import numpy as np
from pyctcdecode import build_ctcdecoder


# ============================================================
# MANIPURI 4-GRAM KENLM RESUME SWEEP
# ============================================================
#
# Purpose:
#   Resume the interrupted 4-gram validation sweep from
#   configuration 14/20.
#
# Already completed:
#   Configurations 1-13
#
# Will run:
#   Configurations 14-20
#
# Validation:
#   Same cached B3 validation logits
#   Same 864 validation samples
#   Same vocabulary
#   Same beam width = 500
#
# ============================================================


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


RESULTS_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "kenlm_4gram_sweep"
)


RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


RESULTS_JSON = (
    RESULTS_DIR
    / "kenlm_4gram_resume_results.json"
)


RESULTS_CSV = (
    RESULTS_DIR
    / "kenlm_4gram_resume_results.csv"
)


# ============================================================
# SETTINGS
# ============================================================

BEAM_WIDTH = 500

ALPHAS = [
    0.2,
    0.5,
    0.8,
    1.0,
]

BETAS = [
    -1.0,
    0.0,
    1.0,
    1.5,
    2.0,
]


TOTAL_CONFIGS = len(ALPHAS) * len(BETAS)

# Resume from configuration 14.
RESUME_FROM = 14


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
print("MANIPURI 4-GRAM KENLM RESUME SWEEP")
print("=" * 70)

print("\n[CHECK] Required files...")


required_files = [
    LOGITS_PATH,
    OFFSETS_PATH,
    TEXTS_PATH,
    VOCAB_PATH,
    UNIGRAM_PATH,
    KENLM_BIN,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(
        f"[OK] {path.name}"
    )


print(
    f"\n[LM] Using:\n{KENLM_BIN}"
)


# ============================================================
# LOAD CACHED LOGITS
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


print(
    f"[LOAD] Logits shape: {logits.shape}"
)

print(
    f"[LOAD] Samples: {len(references):,}"
)


if len(references) != 864:

    raise RuntimeError(
        f"Expected exactly 864 validation samples, "
        f"found {len(references)}"
    )


# ============================================================
# NORMALIZE REFERENCES
# ============================================================

references = [
    normalize_text(x)
    for x in references
]


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


# pyctcdecode requires the CTC blank
# to be represented by an empty string.

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


print(
    f"[LM] Unigrams: {len(unigrams):,}"
)


# ============================================================
# BUILD 4-GRAM DECODER
# ============================================================

def build_decoder(alpha, beta):

    print(
        f"[DECODER] Building 4-gram decoder "
        f"(alpha={alpha}, beta={beta})..."
    )

    return build_ctcdecoder(
        labels=labels,

        kenlm_model_path=str(
            KENLM_BIN
        ),

        unigrams=unigrams,

        alpha=alpha,

        beta=beta,
    )


# ============================================================
# GREEDY DECODER
# ============================================================

def greedy_decode(lp):

    ids = np.argmax(
        lp,
        axis=-1
    )

    output = []

    previous = None

    for token_id in ids:

        token_id = int(token_id)

        # CTC blank
        if token_id == 0:

            previous = None

            continue

        # CTC repeated token
        if token_id == previous:

            continue

        output.append(
            labels[token_id]
        )

        previous = token_id

    return normalize_text(
        "".join(output)
    )


# ============================================================
# GREEDY BASELINE
# ============================================================

print("\n" + "=" * 70)
print("GREEDY BASELINE")
print("=" * 70)


greedy_predictions = []


for i in range(
    len(references)
):

    start = int(
        offsets[i]
    )

    end = int(
        offsets[i + 1]
    )

    lp = logits[
        start:end
    ]

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
    f"Greedy CER : "
    f"{greedy_cer * 100:.4f}%"
)

print(
    f"Greedy WER : "
    f"{greedy_wer * 100:.4f}%"
)


# ============================================================
# RESULTS
# ============================================================

results = []


results.append(
    {
        "decoder": "greedy",
        "alpha": None,
        "beta": None,
        "beam_width": 1,
        "cer": greedy_cer,
        "wer": greedy_wer,
        "cer_percent": greedy_cer * 100,
        "wer_percent": greedy_wer * 100,
    }
)


# ============================================================
# PRELOADED COMPLETED 4-GRAM RESULTS
# ============================================================
#
# These are the results from configurations 1-13
# already completed before the KeyboardInterrupt.
#
# We DO NOT rerun them.
#
# ============================================================

completed_results = [

    # --------------------------------------------------------
    # CONFIG 1
    # alpha=0.2 beta=-1.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.2,
        "beta": -1.0,
        "beam_width": 500,
        "cer": 0.135556,
        "wer": 0.439828,
        "cer_percent": 13.5556,
        "wer_percent": 43.9828,
        "delta_vs_greedy_cer":
            (0.135556 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.439828 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 2
    # alpha=0.2 beta=0.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.2,
        "beta": 0.0,
        "beam_width": 500,
        "cer": 0.133511,
        "wer": 0.450141,
        "cer_percent": 13.3511,
        "wer_percent": 45.0141,
        "delta_vs_greedy_cer":
            (0.133511 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.450141 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 3
    # alpha=0.2 beta=1.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.2,
        "beta": 1.0,
        "beam_width": 500,
        "cer": 0.134484,
        "wer": 0.485161,
        "cer_percent": 13.4484,
        "wer_percent": 48.5161,
        "delta_vs_greedy_cer":
            (0.134484 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.485161 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 4
    # alpha=0.2 beta=1.5
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.2,
        "beta": 1.5,
        "beam_width": 500,
        "cer": 0.136000,
        "wer": 0.510981,
        "cer_percent": 13.6000,
        "wer_percent": 51.0981,
        "delta_vs_greedy_cer":
            (0.136000 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.510981 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 5
    # alpha=0.2 beta=2.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.2,
        "beta": 2.0,
        "beam_width": 500,
        "cer": 0.138353,
        "wer": 0.541178,
        "cer_percent": 13.8353,
        "wer_percent": 54.1178,
        "delta_vs_greedy_cer":
            (0.138353 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.541178 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 6
    # alpha=0.5 beta=-1.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.5,
        "beta": -1.0,
        "beam_width": 500,
        "cer": 0.146633,
        "wer": 0.455631,
        "cer_percent": 14.6633,
        "wer_percent": 45.5631,
        "delta_vs_greedy_cer":
            (0.146633 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.455631 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 7
    # alpha=0.5 beta=0.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.5,
        "beta": 0.0,
        "beam_width": 500,
        "cer": 0.140669,
        "wer": 0.441534,
        "cer_percent": 14.0669,
        "wer_percent": 44.1534,
        "delta_vs_greedy_cer":
            (0.140669 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.441534 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 8
    # alpha=0.5 beta=1.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.5,
        "beta": 1.0,
        "beam_width": 500,
        "cer": 0.135667,
        "wer": 0.436489,
        "cer_percent": 13.5667,
        "wer_percent": 43.6489,
        "delta_vs_greedy_cer":
            (0.135667 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.436489 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 9
    # alpha=0.5 beta=1.5
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.5,
        "beta": 1.5,
        "beam_width": 500,
        "cer": 0.133733,
        "wer": 0.439679,
        "cer_percent": 13.3733,
        "wer_percent": 43.9679,
        "delta_vs_greedy_cer":
            (0.133733 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.439679 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 10
    # alpha=0.5 beta=2.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.5,
        "beta": 2.0,
        "beam_width": 500,
        "cer": 0.132919,
        "wer": 0.446580,
        "cer_percent": 13.2919,
        "wer_percent": 44.6580,
        "delta_vs_greedy_cer":
            (0.132919 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.446580 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 11
    # alpha=0.8 beta=-1.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.8,
        "beta": -1.0,
        "beam_width": 500,
        "cer": 0.170733,
        "wer": 0.534130,
        "cer_percent": 17.0733,
        "wer_percent": 53.4130,
        "delta_vs_greedy_cer":
            (0.170733 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.534130 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 12
    # alpha=0.8 beta=0.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.8,
        "beta": 0.0,
        "beam_width": 500,
        "cer": 0.159323,
        "wer": 0.491616,
        "cer_percent": 15.9323,
        "wer_percent": 49.1616,
        "delta_vs_greedy_cer":
            (0.159323 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.491616 - greedy_wer) * 100,
    },

    # --------------------------------------------------------
    # CONFIG 13
    # alpha=0.8 beta=1.0
    # --------------------------------------------------------
    {
        "decoder": "kenlm_4gram",
        "alpha": 0.8,
        "beta": 1.0,
        "beam_width": 500,
        "cer": 0.151426,
        "wer": 0.462161,
        "cer_percent": 15.1426,
        "wer_percent": 46.2161,
        "delta_vs_greedy_cer":
            (0.151426 - greedy_cer) * 100,
        "delta_vs_greedy_wer":
            (0.462161 - greedy_wer) * 100,
    },
]


print(
    "\n[RESUME] Preloaded completed configurations:"
)

print(
    f"[RESUME] {len(completed_results)} / "
    f"{TOTAL_CONFIGS} configurations"
)


results.extend(
    completed_results
)


# ============================================================
# RESUME CONFIGURATIONS 14-20
# ============================================================

print("\n" + "=" * 70)
print("RESUMING 4-GRAM SWEEP")
print("=" * 70)

print(
    f"Starting from configuration "
    f"{RESUME_FROM}/{TOTAL_CONFIGS}"
)


config_number = 0


for alpha in ALPHAS:

    for beta in BETAS:

        config_number += 1

        # Skip already completed configurations.
        if config_number < RESUME_FROM:

            continue


        print("\n" + "=" * 70)

        print(
            f"CONFIG "
            f"{config_number}/{TOTAL_CONFIGS}"
        )

        print(
            f"alpha = {alpha}"
        )

        print(
            f"beta  = {beta}"
        )

        print("=" * 70)


        decoder = build_decoder(
            alpha,
            beta
        )


        predictions = []


        for i in range(
            len(references)
        ):

            start = int(
                offsets[i]
            )

            end = int(
                offsets[i + 1]
            )

            lp = logits[
                start:end
            ]


            prediction = decoder.decode(
                lp,
                beam_width=BEAM_WIDTH
            )


            predictions.append(
                normalize_text(
                    prediction
                )
            )


            if (
                (i + 1) % 100 == 0
                or i + 1 == len(references)
            ):

                print(
                    f"[{i + 1:4d}/"
                    f"{len(references)}]"
                )


        cer = calculate_cer(
            references,
            predictions
        )


        wer = calculate_wer(
            references,
            predictions
        )


        delta_cer = (
            cer - greedy_cer
        )


        delta_wer = (
            wer - greedy_wer
        )


        result = {

            "decoder":
                "kenlm_4gram",

            "alpha":
                alpha,

            "beta":
                beta,

            "beam_width":
                BEAM_WIDTH,

            "cer":
                cer,

            "wer":
                wer,

            "cer_percent":
                cer * 100,

            "wer_percent":
                wer * 100,

            "delta_vs_greedy_cer":
                delta_cer * 100,

            "delta_vs_greedy_wer":
                delta_wer * 100,

        }


        results.append(
            result
        )


        print()

        print(
            f"CER : "
            f"{cer * 100:.4f}%"
        )

        print(
            f"WER : "
            f"{wer * 100:.4f}%"
        )

        print(
            f"Delta CER vs greedy : "
            f"{delta_cer * 100:+.4f} pp"
        )

        print(
            f"Delta WER vs greedy : "
            f"{delta_wer * 100:+.4f} pp"
        )


# ============================================================
# CHECK COMPLETENESS
# ============================================================

kenlm_results = [
    x
    for x in results
    if x["decoder"] == "kenlm_4gram"
]


if len(kenlm_results) != TOTAL_CONFIGS:

    raise RuntimeError(
        f"Expected {TOTAL_CONFIGS} 4-gram "
        f"results, found {len(kenlm_results)}"
    )


print(
    f"\n[CHECK] Completed "
    f"{len(kenlm_results)}/{TOTAL_CONFIGS} "
    f"4-gram configurations."
)


# ============================================================
# FIND BEST RESULTS
# ============================================================

best_cer = min(
    kenlm_results,
    key=lambda x: x["cer"]
)


best_wer = min(
    kenlm_results,
    key=lambda x: x["wer"]
)


# ============================================================
# 3-GRAM REFERENCE
# ============================================================

THREEGRAM_BEST_CER = 13.3190
THREEGRAM_BEST_CER_ALPHA = 0.5
THREEGRAM_BEST_CER_BETA = 2.0

THREEGRAM_BEST_WER = 43.7379
THREEGRAM_BEST_WER_ALPHA = 0.5
THREEGRAM_BEST_WER_BETA = 1.0


cer_difference_vs_3gram = (
    best_cer["cer_percent"]
    - THREEGRAM_BEST_CER
)


wer_difference_vs_3gram = (
    best_wer["wer_percent"]
    - THREEGRAM_BEST_WER
)


# ============================================================
# SAVE JSON
# ============================================================

output = {

    "experiment":
        "B3 validation 4-gram KenLM resume sweep",

    "validation_samples":
        len(references),

    "beam_width":
        BEAM_WIDTH,

    "alphas":
        ALPHAS,

    "betas":
        BETAS,

    "resume_from":
        RESUME_FROM,

    "kenlm_binary":
        str(KENLM_BIN),

    "unigram_file":
        str(UNIGRAM_PATH),

    "greedy_baseline": {

        "cer":
            greedy_cer,

        "wer":
            greedy_wer,

        "cer_percent":
            greedy_cer * 100,

        "wer_percent":
            greedy_wer * 100,
    },

    "threegram_reference": {

        "best_cer_percent":
            THREEGRAM_BEST_CER,

        "best_cer_alpha":
            THREEGRAM_BEST_CER_ALPHA,

        "best_cer_beta":
            THREEGRAM_BEST_CER_BETA,

        "best_wer_percent":
            THREEGRAM_BEST_WER,

        "best_wer_alpha":
            THREEGRAM_BEST_WER_ALPHA,

        "best_wer_beta":
            THREEGRAM_BEST_WER_BETA,
    },

    "best_4gram_by_cer":
        best_cer,

    "best_4gram_by_wer":
        best_wer,

    "difference_vs_3gram": {

        "cer_pp":
            cer_difference_vs_3gram,

        "wer_pp":
            wer_difference_vs_3gram,
    },

    "all_results":
        results,
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
# SAVE CSV
# ============================================================

with open(
    RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    fieldnames = [
        "decoder",
        "alpha",
        "beta",
        "beam_width",
        "cer",
        "wer",
        "cer_percent",
        "wer_percent",
        "delta_vs_greedy_cer",
        "delta_vs_greedy_wer",
    ]


    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )


    writer.writeheader()


    for result in results:

        writer.writerow(
            result
        )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("4-GRAM RESUME SWEEP COMPLETE")
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


print("\n" + "-" * 70)
print("BEST 4-GRAM BY CER")
print("-" * 70)

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
    f"WER   = "
    f"{best_cer['wer_percent']:.4f}%"
)

print(
    f"Delta vs greedy CER = "
    f"{best_cer['delta_vs_greedy_cer']:+.4f} pp"
)

print(
    f"Difference vs 3-gram CER = "
    f"{cer_difference_vs_3gram:+.4f} pp"
)


print("\n" + "-" * 70)
print("BEST 4-GRAM BY WER")
print("-" * 70)

print(
    f"alpha = {best_wer['alpha']}"
)

print(
    f"beta  = {best_wer['beta']}"
)

print(
    f"CER   = "
    f"{best_wer['cer_percent']:.4f}%"
)

print(
    f"WER   = "
    f"{best_wer['wer_percent']:.4f}%"
)

print(
    f"Delta vs greedy WER = "
    f"{best_wer['delta_vs_greedy_wer']:+.4f} pp"
)

print(
    f"Difference vs 3-gram WER = "
    f"{wer_difference_vs_3gram:+.4f} pp"
)


print("\n" + "-" * 70)
print("3-GRAM REFERENCE")
print("-" * 70)

print(
    f"Best CER = "
    f"{THREEGRAM_BEST_CER:.4f}% "
    f"(alpha={THREEGRAM_BEST_CER_ALPHA}, "
    f"beta={THREEGRAM_BEST_CER_BETA})"
)

print(
    f"Best WER = "
    f"{THREEGRAM_BEST_WER:.4f}% "
    f"(alpha={THREEGRAM_BEST_WER_ALPHA}, "
    f"beta={THREEGRAM_BEST_WER_BETA})"
)


print("\n" + "-" * 70)
print("FINAL DECODER DECISION")
print("-" * 70)


if best_cer["cer_percent"] < THREEGRAM_BEST_CER:

    print(
        "4-GRAM WINS ON CER"
    )

elif best_cer["cer_percent"] > THREEGRAM_BEST_CER:

    print(
        "3-GRAM WINS ON CER"
    )

else:

    print(
        "CER TIE"
    )


if best_wer["wer_percent"] < THREEGRAM_BEST_WER:

    print(
        "4-GRAM WINS ON WER"
    )

elif best_wer["wer_percent"] > THREEGRAM_BEST_WER:

    print(
        "3-GRAM WINS ON WER"
    )

else:

    print(
        "WER TIE"
    )


print()

print(
    f"[SAVED] {RESULTS_JSON}"
)

print(
    f"[SAVED] {RESULTS_CSV}"
)

print("=" * 70)