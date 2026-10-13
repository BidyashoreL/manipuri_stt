"""
======================================================================
MBR / CONSENSUS DECODING FOR MANIPURI ASR
======================================================================

Research-clean evaluation of Minimum Bayes Risk (MBR) decoding
on the existing pooled Top-20 N-best candidate set.

Dataset:
    B3 validation set
    864 utterances

Candidate source:
    b3_4gram_pooled100_val.json

Selection:
    Character-level MBR
    Word-level MBR

Posterior-like candidate weights:
    softmax(gamma * decoder_score)

Decoder score:
    logit_score + alpha * lm_score

where:
    alpha = 0.5

IMPORTANT:
    Reference transcripts are NEVER used for MBR selection.
    They are used only for final CER/WER evaluation.

Official 410-sample test set is NOT touched.
======================================================================
"""

from pathlib import Path
import json
import unicodedata
import numpy as np


# ======================================================================
# PATHS
# ======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

NBEST_PATH = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "b3_4gram_pooled100_val.json"
)

OUTPUT_PATH = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "mbr_results.json"
)


# ======================================================================
# CONFIGURATION
# ======================================================================

TOP_K = 20

# 4-gram decoder alpha used when converting
# acoustic + LM scores into a sequence score.
LM_ALPHA = 0.5

# gamma controls sharpness of the posterior-like distribution:
#
# gamma < 1  -> flatter
# gamma = 1  -> normal
# gamma > 1  -> sharper
#
# Equivalent temperature is:
#
# temperature = 1 / gamma
#
GAMMAS = [
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
]


# ======================================================================
# TEXT NORMALIZATION
# ======================================================================

def normalize_text(text):
    """
    NFC normalization + whitespace normalization.
    """
    text = unicodedata.normalize(
        "NFC",
        str(text),
    )

    return " ".join(text.split()).strip()


# ======================================================================
# EDIT DISTANCE
# ======================================================================

def edit_distance(a, b):
    """
    Standard Levenshtein edit distance.

    Works with:
        list(characters)
        list(words)
    """

    if len(a) == 0:
        return len(b)

    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))

    for i, x in enumerate(a, start=1):

        cur = [i]

        for j, y in enumerate(b, start=1):

            substitution = (
                prev[j - 1]
                + (0 if x == y else 1)
            )

            insertion = cur[j - 1] + 1
            deletion = prev[j] + 1

            cur.append(
                min(
                    substitution,
                    insertion,
                    deletion,
                )
            )

        prev = cur

    return prev[-1]


# ======================================================================
# SOFTMAX
# ======================================================================

def stable_softmax(scores, gamma):
    """
    Convert decoder scores to posterior-like weights.

        P_i = exp(gamma * score_i)
              ---------------------
              sum_j exp(gamma * score_j)

    Numerically stable implementation.
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    scaled = gamma * scores

    scaled -= np.max(scaled)

    probs = np.exp(scaled)

    denominator = np.sum(probs)

    if denominator <= 0:
        return np.ones_like(probs) / len(probs)

    return probs / denominator


# ======================================================================
# LOAD DATASET
# ======================================================================

print("=" * 78)
print("MINIMUM BAYES RISK (MBR) DECODING")
print("MANIPURI ASR — B3 4-GRAM POOLED N-BEST")
print("=" * 78)

print()
print(f"Candidate file : {NBEST_PATH}")
print(f"Top-K          : {TOP_K}")
print(f"LM alpha       : {LM_ALPHA}")
print(f"Gammas         : {GAMMAS}")
print()

if not NBEST_PATH.exists():
    raise FileNotFoundError(
        f"\nN-best file not found:\n{NBEST_PATH}"
    )

print("Loading N-best dataset...")

with open(
    NBEST_PATH,
    "r",
    encoding="utf-8",
) as f:

    data = json.load(f)


# Support both:
#
# {"samples": [...]}
#
# and:
#
# [...]
#
if isinstance(data, dict):
    samples = data.get(
        "samples",
        data.get("results", []),
    )
else:
    samples = data


if not isinstance(samples, list):
    raise ValueError(
        "Unexpected N-best JSON structure."
    )


print(
    f"Loaded samples : {len(samples)}"
)


# ======================================================================
# STORAGE
# ======================================================================

processed_samples = []

total_chars = 0
total_words = 0

top1_cer_errors = 0
top1_wer_errors = 0

oracle_cer_errors = 0
oracle_wer_errors = 0

gamma_results = {
    g: {
        "char_cer_errors": 0,
        "char_wer_errors": 0,
        "word_cer_errors": 0,
        "word_wer_errors": 0,

        "char_selected_ranks": [],
        "word_selected_ranks": [],

        "char_changed": 0,
        "word_changed": 0,

        "char_risks": [],
        "word_risks": [],
    }
    for g in GAMMAS
}


# ======================================================================
# PROCESS EACH UTTERANCE
# ======================================================================

print()
print("=" * 78)
print("PROCESSING CANDIDATE POOL")
print("=" * 78)

for sample_idx, sample in enumerate(samples):

    # --------------------------------------------------------------
    # Reference
    # --------------------------------------------------------------

    ref = normalize_text(
        sample.get(
            "reference",
            sample.get("text", ""),
        )
    )

    # --------------------------------------------------------------
    # Candidate extraction
    # --------------------------------------------------------------

    hypotheses = sample.get(
        "hypotheses",
        sample.get(
            "nbest",
            sample.get(
                "candidates",
                [],
            ),
        ),
    )

    hypotheses = hypotheses[:TOP_K]

    if not hypotheses:
        continue

    cand_texts = [
        normalize_text(
            h.get("text", "")
        )
        for h in hypotheses
    ]

    # Remove empty candidates.
    valid_indices = [
        i
        for i, text in enumerate(cand_texts)
        if text != ""
    ]

    if not valid_indices:
        continue

    hypotheses = [
        hypotheses[i]
        for i in valid_indices
    ]

    cand_texts = [
        cand_texts[i]
        for i in valid_indices
    ]

    K = len(cand_texts)

    # --------------------------------------------------------------
    # Reference lengths
    # --------------------------------------------------------------

    ref_chars = len(ref)
    ref_words = len(ref.split())

    total_chars += ref_chars
    total_words += ref_words

    # --------------------------------------------------------------
    # Decoder scores
    # --------------------------------------------------------------
    #
    # IMPORTANT:
    #
    # score = acoustic logit + alpha * LM score
    #
    # alpha = 0.5
    #
    # We intentionally do NOT use the reference.
    #

    decoder_scores = np.array(
        [
            float(
                h.get(
                    "logit_score",
                    0.0,
                )
            )
            +
            LM_ALPHA
            *
            float(
                h.get(
                    "lm_score",
                    0.0,
                )
            )
            for h in hypotheses
        ],
        dtype=np.float64,
    )

    # --------------------------------------------------------------
    # Top-1 baseline
    # --------------------------------------------------------------

    top1_text = cand_texts[0]

    top1_char_error = edit_distance(
        list(ref),
        list(top1_text),
    )

    top1_word_error = edit_distance(
        ref.split(),
        top1_text.split(),
    )

    top1_cer_errors += top1_char_error
    top1_wer_errors += top1_word_error

    # --------------------------------------------------------------
    # Oracle Top-K
    # --------------------------------------------------------------

    candidate_char_errors = [
        edit_distance(
            list(ref),
            list(text),
        )
        for text in cand_texts
    ]

    candidate_word_errors = [
        edit_distance(
            ref.split(),
            text.split(),
        )
        for text in cand_texts
    ]

    oracle_cer_errors += min(
        candidate_char_errors
    )

    oracle_wer_errors += min(
        candidate_word_errors
    )

    # --------------------------------------------------------------
    # Pairwise candidate distances
    # --------------------------------------------------------------

    char_dist_matrix = np.zeros(
        (K, K),
        dtype=np.float64,
    )

    word_dist_matrix = np.zeros(
        (K, K),
        dtype=np.float64,
    )

    for i in range(K):

        char_i = list(
            cand_texts[i]
        )

        word_i = cand_texts[i].split()

        for j in range(i + 1, K):

            char_j = list(
                cand_texts[j]
            )

            word_j = cand_texts[j].split()

            char_distance = edit_distance(
                char_i,
                char_j,
            )

            word_distance = edit_distance(
                word_i,
                word_j,
            )

            char_dist_matrix[i, j] = (
                char_distance
            )

            char_dist_matrix[j, i] = (
                char_distance
            )

            word_dist_matrix[i, j] = (
                word_distance
            )

            word_dist_matrix[j, i] = (
                word_distance
            )

    # --------------------------------------------------------------
    # MBR for every gamma
    # --------------------------------------------------------------

    sample_record = {
        "sample_index": sample_idx,
        "reference": ref,
        "top1": top1_text,
        "num_candidates": K,
        "candidates": [],
        "mbr": {},
    }

    # Keep candidate metadata.
    for rank, h in enumerate(
        hypotheses,
        start=1,
    ):

        sample_record["candidates"].append(
            {
                "rank": rank,
                "text": cand_texts[rank - 1],
                "logit_score": float(
                    h.get(
                        "logit_score",
                        0.0,
                    )
                ),
                "lm_score": float(
                    h.get(
                        "lm_score",
                        0.0,
                    )
                ),
                "decoder_score": float(
                    decoder_scores[rank - 1]
                ),
            }
        )

    for gamma in GAMMAS:

        probs = stable_softmax(
            decoder_scores,
            gamma,
        )

        # ==========================================================
        # CHARACTER MBR
        # ==========================================================

        expected_char_risk = (
            char_dist_matrix @ probs
        )

        best_char_idx = int(
            np.argmin(
                expected_char_risk
            )
        )

        char_text = cand_texts[
            best_char_idx
        ]

        char_error = edit_distance(
            list(ref),
            list(char_text),
        )

        char_word_error = edit_distance(
            ref.split(),
            char_text.split(),
        )

        char_rank = (
            best_char_idx + 1
        )

        gamma_results[gamma][
            "char_cer_errors"
        ] += char_error

        gamma_results[gamma][
            "char_wer_errors"
        ] += char_word_error

        gamma_results[gamma][
            "char_selected_ranks"
        ].append(
            char_rank
        )

        gamma_results[gamma][
            "char_risks"
        ].append(
            float(
                expected_char_risk[
                    best_char_idx
                ]
            )
        )

        if char_rank != 1:
            gamma_results[gamma][
                "char_changed"
            ] += 1

        # ==========================================================
        # WORD MBR
        # ==========================================================

        expected_word_risk = (
            word_dist_matrix @ probs
        )

        best_word_idx = int(
            np.argmin(
                expected_word_risk
            )
        )

        word_text = cand_texts[
            best_word_idx
        ]

        word_char_error = edit_distance(
            list(ref),
            list(word_text),
        )

        word_error = edit_distance(
            ref.split(),
            word_text.split(),
        )

        word_rank = (
            best_word_idx + 1
        )

        gamma_results[gamma][
            "word_cer_errors"
        ] += word_char_error

        gamma_results[gamma][
            "word_wer_errors"
        ] += word_error

        gamma_results[gamma][
            "word_selected_ranks"
        ].append(
            word_rank
        )

        gamma_results[gamma][
            "word_risks"
        ].append(
            float(
                expected_word_risk[
                    best_word_idx
                ]
            )
        )

        if word_rank != 1:
            gamma_results[gamma][
                "word_changed"
            ] += 1

        # ----------------------------------------------------------
        # Save selected candidates
        # ----------------------------------------------------------

        sample_record["mbr"][
            str(gamma)
        ] = {
            "character": {
                "rank": char_rank,
                "text": char_text,
                "risk": float(
                    expected_char_risk[
                        best_char_idx
                    ]
                ),
            },
            "word": {
                "rank": word_rank,
                "text": word_text,
                "risk": float(
                    expected_word_risk[
                        best_word_idx
                    ]
                ),
            },
        }

    processed_samples.append(
        sample_record
    )

    if (
        len(processed_samples) % 100 == 0
    ):

        print(
            f"Processed "
            f"{len(processed_samples)}/"
            f"{len(samples)} samples..."
        )


# ======================================================================
# BASELINE / ORACLE METRICS
# ======================================================================

if total_chars == 0:
    raise RuntimeError(
        "No valid samples were processed."
    )

baseline_cer = (
    100.0
    * top1_cer_errors
    / total_chars
)

baseline_wer = (
    100.0
    * top1_wer_errors
    / total_words
)

oracle_cer = (
    100.0
    * oracle_cer_errors
    / total_chars
)

oracle_wer = (
    100.0
    * oracle_wer_errors
    / total_words
)


# ======================================================================
# PRINT RESULTS
# ======================================================================

print()
print("=" * 78)
print(
    "MINIMUM BAYES RISK (MBR) DECODING RESULTS"
)
print("TOP-20 POOLED CANDIDATE POOL")
print("=" * 78)

print()
print(
    f"Processed samples          : "
    f"{len(processed_samples)}"
)

print(
    f"Reference characters       : "
    f"{total_chars:,}"
)

print(
    f"Reference words            : "
    f"{total_words:,}"
)

print()
print("-" * 78)
print("BASELINE / ORACLE")
print("-" * 78)

print(
    f"Original Top-1 CER         : "
    f"{baseline_cer:.4f}%"
)

print(
    f"Original Top-1 WER         : "
    f"{baseline_wer:.4f}%"
)

print()

print(
    f"Top-20 Oracle CER          : "
    f"{oracle_cer:.4f}%"
)

print(
    f"Top-20 Oracle WER          : "
    f"{oracle_wer:.4f}%"
)

print(
    f"Oracle CER gap             : "
    f"{oracle_cer - baseline_cer:+.4f} pp"
)

print(
    f"Oracle WER gap             : "
    f"{oracle_wer - baseline_wer:+.4f} pp"
)


# ======================================================================
# CHARACTER MBR
# ======================================================================

print()
print("-" * 78)
print("CHARACTER-LEVEL MBR")
print("-" * 78)

best_char = None

for gamma in GAMMAS:

    result = gamma_results[gamma]

    cer = (
        100.0
        * result["char_cer_errors"]
        / total_chars
    )

    wer = (
        100.0
        * result["char_wer_errors"]
        / total_words
    )

    ranks = np.array(
        result["char_selected_ranks"]
    )

    avg_rank = (
        float(np.mean(ranks))
        if len(ranks)
        else 0.0
    )

    median_rank = (
        float(np.median(ranks))
        if len(ranks)
        else 0.0
    )

    changed_pct = (
        100.0
        * result["char_changed"]
        / len(processed_samples)
    )

    print(
        f"Gamma {gamma:>3.1f} -> "
        f"CER: {cer:.4f}% "
        f"({cer - baseline_cer:+.4f} pp) | "
        f"WER: {wer:.4f}% | "
        f"Avg Rank: {avg_rank:.2f} | "
        f"Median Rank: {median_rank:.1f} | "
        f"Changed: {changed_pct:.2f}%"
    )

    current = {
        "gamma": gamma,
        "cer": cer,
        "wer": wer,
        "cer_delta_pp": (
            cer - baseline_cer
        ),
        "wer_delta_pp": (
            wer - baseline_wer
        ),
        "avg_rank": avg_rank,
        "median_rank": median_rank,
        "changed_pct": changed_pct,
    }

    if (
        best_char is None
        or cer < best_char["cer"]
    ):
        best_char = current


# ======================================================================
# WORD MBR
# ======================================================================

print()
print("-" * 78)
print("WORD-LEVEL MBR")
print("-" * 78)

best_word = None

for gamma in GAMMAS:

    result = gamma_results[gamma]

    cer = (
        100.0
        * result["word_cer_errors"]
        / total_chars
    )

    wer = (
        100.0
        * result["word_wer_errors"]
        / total_words
    )

    ranks = np.array(
        result["word_selected_ranks"]
    )

    avg_rank = (
        float(np.mean(ranks))
        if len(ranks)
        else 0.0
    )

    median_rank = (
        float(np.median(ranks))
        if len(ranks)
        else 0.0
    )

    changed_pct = (
        100.0
        * result["word_changed"]
        / len(processed_samples)
    )

    print(
        f"Gamma {gamma:>3.1f} -> "
        f"CER: {cer:.4f}% | "
        f"WER: {wer:.4f}% "
        f"({wer - baseline_wer:+.4f} pp) | "
        f"Avg Rank: {avg_rank:.2f} | "
        f"Median Rank: {median_rank:.1f} | "
        f"Changed: {changed_pct:.2f}%"
    )

    current = {
        "gamma": gamma,
        "cer": cer,
        "wer": wer,
        "cer_delta_pp": (
            cer - baseline_cer
        ),
        "wer_delta_pp": (
            wer - baseline_wer
        ),
        "avg_rank": avg_rank,
        "median_rank": median_rank,
        "changed_pct": changed_pct,
    }

    if (
        best_word is None
        or wer < best_word["wer"]
    ):
        best_word = current


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print()
print("=" * 78)
print("BEST MBR RESULTS")
print("=" * 78)

print()
print("Best Character MBR by CER:")
print(
    f"  Gamma       : "
    f"{best_char['gamma']}"
)

print(
    f"  CER         : "
    f"{best_char['cer']:.4f}%"
)

print(
    f"  WER         : "
    f"{best_char['wer']:.4f}%"
)

print(
    f"  CER delta   : "
    f"{best_char['cer_delta_pp']:+.4f} pp"
)

print(
    f"  WER delta   : "
    f"{best_char['wer_delta_pp']:+.4f} pp"
)

print()
print("Best Word MBR by WER:")
print(
    f"  Gamma       : "
    f"{best_word['gamma']}"
)

print(
    f"  CER         : "
    f"{best_word['cer']:.4f}%"
)

print(
    f"  WER         : "
    f"{best_word['wer']:.4f}%"
)

print(
    f"  CER delta   : "
    f"{best_word['cer_delta_pp']:+.4f} pp"
)

print(
    f"  WER delta   : "
    f"{best_word['wer_delta_pp']:+.4f} pp"
)


# ======================================================================
# SAVE JSON
# ======================================================================

output = {
    "experiment": {
        "name": "Minimum Bayes Risk / Consensus Decoding",
        "candidate_file": str(NBEST_PATH),
        "top_k": TOP_K,
        "lm_alpha": LM_ALPHA,
        "gammas": GAMMAS,
        "num_samples": len(
            processed_samples
        ),
        "score_formula": (
            "logit_score + "
            f"{LM_ALPHA} * lm_score"
        ),
        "selection": (
            "argmin_i sum_j "
            "P(Y_j|X) * "
            "EditDistance(Y_i,Y_j)"
        ),
        "reference_used_for_selection": False,
    },

    "baseline": {
        "cer": baseline_cer,
        "wer": baseline_wer,
        "char_errors": top1_cer_errors,
        "ref_chars": total_chars,
        "word_errors": top1_wer_errors,
        "ref_words": total_words,
    },

    "oracle_top20": {
        "cer": oracle_cer,
        "wer": oracle_wer,
        "char_errors": oracle_cer_errors,
        "word_errors": oracle_wer_errors,
    },

    "best_character_mbr": best_char,

    "best_word_mbr": best_word,

    "per_gamma": {},

    "per_sample": processed_samples,
}


# Add aggregate gamma results.
for gamma in GAMMAS:

    result = gamma_results[gamma]

    char_ranks = np.array(
        result["char_selected_ranks"]
    )

    word_ranks = np.array(
        result["word_selected_ranks"]
    )

    char_cer = (
        100.0
        * result["char_cer_errors"]
        / total_chars
    )

    char_wer = (
        100.0
        * result["char_wer_errors"]
        / total_words
    )

    word_cer = (
        100.0
        * result["word_cer_errors"]
        / total_chars
    )

    word_wer = (
        100.0
        * result["word_wer_errors"]
        / total_words
    )

    output["per_gamma"][str(gamma)] = {

        "character_mbr": {
            "cer": char_cer,
            "wer": char_wer,
            "cer_delta_pp": (
                char_cer
                - baseline_cer
            ),
            "wer_delta_pp": (
                char_wer
                - baseline_wer
            ),
            "avg_rank": float(
                np.mean(char_ranks)
            ),
            "median_rank": float(
                np.median(char_ranks)
            ),
            "changed_pct": (
                100.0
                * result["char_changed"]
                / len(processed_samples)
            ),
        },

        "word_mbr": {
            "cer": word_cer,
            "wer": word_wer,
            "cer_delta_pp": (
                word_cer
                - baseline_cer
            ),
            "wer_delta_pp": (
                word_wer
                - baseline_wer
            ),
            "avg_rank": float(
                np.mean(word_ranks)
            ),
            "median_rank": float(
                np.median(word_ranks)
            ),
            "changed_pct": (
                100.0
                * result["word_changed"]
                / len(processed_samples)
            ),
        },
    }


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ======================================================================
# COMPLETION
# ======================================================================

print()
print("=" * 78)
print("RESULTS SAVED")
print("=" * 78)

print(
    OUTPUT_PATH
)

print()
print("=" * 78)
print("SUCCESS")
print("=" * 78)