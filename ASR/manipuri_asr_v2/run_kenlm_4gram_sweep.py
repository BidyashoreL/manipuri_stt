from pathlib import Path
import json
import sys
import unicodedata
import re

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
    / "kenlm_4gram_sweep_results.json"
)

RESULTS_CSV = (
    RESULTS_DIR
    / "kenlm_4gram_sweep_results.csv"
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
# CER
# ============================================================

def edit_distance(
    reference,
    hypothesis
):

    m = len(reference)
    n = len(hypothesis)

    previous = list(
        range(n + 1)
    )

    for i in range(1, m + 1):

        current = [i]

        for j in range(1, n + 1):

            cost = (
                0
                if reference[i - 1]
                == hypothesis[j - 1]
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


def calculate_cer(
    references,
    hypotheses
):

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

def calculate_wer(
    references,
    hypotheses
):

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
# LOAD VOCAB
# ============================================================

print("=" * 70)
print("MANIPURI KENLM DECODING SWEEP")
print("=" * 70)

print("\n[CHECK] Required files...")

for path in [
    LOGITS_PATH,
    OFFSETS_PATH,
    TEXTS_PATH,
    VOCAB_PATH,
    UNIGRAM_PATH,
    KENLM_BIN,
]:

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    print(
        f"[OK] {path.name}"
    )


# ============================================================
# LOAD DATA
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
    f"[LOAD] Logits shape: "
    f"{logits.shape}"
)

print(
    f"[LOAD] Samples: "
    f"{len(references):,}"
)

if len(references) != 864:

    raise RuntimeError(
        f"Expected 864 samples, "
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


# CTC blank must be represented
# as an empty string for pyctcdecode.

labels = [
    ""
    if token == "<blank>"
    else token
    for token in labels
]


print(
    f"[VOCAB] Labels: "
    f"{len(labels)}"
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
    f"[LM] Unigrams: "
    f"{len(unigrams):,}"
)


# ============================================================
# DECODER BUILDER
# ============================================================

def build_decoder(
    alpha,
    beta
):

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

def greedy_decode(
    lp
):

    ids = np.argmax(
        lp,
        axis=-1
    )

    output = []

    previous = None

    for token_id in ids:

        token_id = int(
            token_id
        )

        # Blank
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
# PREPARE REFERENCES
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
# SAVE RESULTS
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
# KENLM SWEEP
# ============================================================

total_configs = (
    len(ALPHAS)
    * len(BETAS)
)

config_number = 0


for alpha in ALPHAS:

    for beta in BETAS:

        config_number += 1

        print("\n" + "=" * 70)

        print(
            f"CONFIG "
            f"{config_number}/{total_configs}"
        )

        print(
            f"alpha = {alpha}"
        )

        print(
            f"beta  = {beta}"
        )

        print("=" * 70)


        print(
            "[DECODER] Building 4-gram decoder..."
        )

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
            f"Î” CER vs greedy : "
            f"{delta_cer * 100:+.4f} pp"
        )

        print(
            f"Î” WER vs greedy : "
            f"{delta_wer * 100:+.4f} pp"
        )


# ============================================================
# SORT RESULTS
# ============================================================

kenlm_results = [
    x
    for x in results
    if x["decoder"] == "kenlm_4gram"
]


best_cer = min(
    kenlm_results,
    key=lambda x: x["cer"]
)

best_wer = min(
    kenlm_results,
    key=lambda x: x["wer"]
)


# ============================================================
# SAVE JSON
# ============================================================

output = {

    "settings": {

        "beam_width":
            BEAM_WIDTH,

        "alphas":
            ALPHAS,

        "betas":
            BETAS,

        "num_validation_samples":
            len(references),

        "kenlm_binary":
            str(KENLM_BIN),

        "unigram_file":
            str(UNIGRAM_PATH),
    },

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

    "best_4gram_by_cer":
        best_cer,

    "best_4gram_by_wer":
        best_wer,

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

import csv

with open(
    RESULTS_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
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
print("KENLM SWEEP COMPLETE")
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

print("\nBEST 3-GRAM BY CER")

print(
    f"alpha = "
    f"{best_cer['alpha']}"
)

print(
    f"beta  = "
    f"{best_cer['beta']}"
)

print(
    f"CER   = "
    f"{best_cer['cer_percent']:.4f}%"
)

print(
    f"Î” CER  = "
    f"{best_cer['delta_vs_greedy_cer']:+.4f} pp"
)

print("\nBEST 3-GRAM BY WER")

print(
    f"alpha = "
    f"{best_wer['alpha']}"
)

print(
    f"beta  = "
    f"{best_wer['beta']}"
)

print(
    f"WER   = "
    f"{best_wer['wer_percent']:.4f}%"
)

print(
    f"Î” WER  = "
    f"{best_wer['delta_vs_greedy_wer']:+.4f} pp"
)

print()

print(
    f"[SAVED] {RESULTS_JSON}"
)

print(
    f"[SAVED] {RESULTS_CSV}"
)


