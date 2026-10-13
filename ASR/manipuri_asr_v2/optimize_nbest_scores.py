import json
import re
import unicodedata
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
ASR = ROOT.parent

NBEST_JSON = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "b3_4gram_nbest50_val.json"
)

OUTPUT_JSON = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "score_fusion_results.json"
)

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_distance(ref, hyp):

    n = len(ref)
    m = len(hyp)

    prev = list(range(m + 1))

    for i in range(1, n + 1):

        curr = [i] + [0] * m

        for j in range(1, m + 1):

            cost = 0 if ref[i - 1] == hyp[j - 1] else 1

            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost
            )

        prev = curr

    return prev[m]


def cer(ref, hyp):

    ref = list(ref)
    hyp = list(hyp)

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    return edit_distance(ref, hyp) / len(ref)


def wer(ref, hyp):

    ref = ref.split()
    hyp = hyp.split()

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    return edit_distance(ref, hyp) / len(ref)


# ============================================================
# LOAD N-BEST
# ============================================================

print("=" * 70)
print("N-BEST SCORE FUSION")
print("=" * 70)

print("\nLoading:")
print(NBEST_JSON)

with open(NBEST_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"]

print(f"\nValidation samples: {len(samples)}")
print(f"N-best size: {data['n_best']}")


# ============================================================
# PREPARE SAMPLE DATA
# ============================================================

prepared = []

for sample in samples:

    reference = normalize_text(sample["reference"])

    candidates = []

    for h in sample["hypotheses"]:

        text = normalize_text(h["text"])

        candidates.append({
            "rank": int(h["rank"]),
            "text": text,
            "lm_score": float(h["lm_score"]),
            "logit_score": float(h["logit_score"]),
            "length": len(text),
            "words": len(text.split()),
        })

    prepared.append({
        "reference": reference,
        "candidates": candidates,
    })


# ============================================================
# SCORE NORMALIZATION
#
# Raw acoustic and LM scores have different scales.
#
# We normalize each candidate's score within each sample
# using min-max normalization.
#
# Higher normalized score = better.
# ============================================================

def normalize_scores(candidates, key):

    values = [c[key] for c in candidates]

    lo = min(values)
    hi = max(values)

    if abs(hi - lo) < 1e-12:

        for c in candidates:
            c[f"{key}_norm"] = 0.5

    else:

        for c in candidates:

            # Higher raw score is better.
            c[f"{key}_norm"] = (
                (c[key] - lo) / (hi - lo)
            )


# ============================================================
# ADD NORMALIZED FEATURES
# ============================================================

for sample in prepared:

    candidates = sample["candidates"]

    normalize_scores(candidates, "lm_score")
    normalize_scores(candidates, "logit_score")

    # Rank feature:
    # rank 1 = 1.0
    # rank 50 = 0.0
    max_rank = max(c["rank"] for c in candidates)

    for c in candidates:

        c["rank_norm"] = (
            1.0
            - (c["rank"] - 1) / max(max_rank - 1, 1)
        )


# ============================================================
# SCORE FUSION
#
# score =
#
#   acoustic_weight * acoustic
# + lm_weight       * lm
# + rank_weight     * rank
#
# We test many combinations.
# ============================================================

ACOUSTIC_WEIGHTS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

LM_WEIGHTS = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

RANK_WEIGHTS = [
    0.0,
    0.05,
    0.10,
    0.20,
]


# ============================================================
# EVALUATE ONE CONFIG
# ============================================================

def evaluate_config(
    acoustic_weight,
    lm_weight,
    rank_weight,
):

    total_cer_errors = 0
    total_cer_chars = 0

    total_wer_errors = 0
    total_wer_words = 0

    selected_ranks = []

    for sample in prepared:

        reference = sample["reference"]
        candidates = sample["candidates"]

        best_candidate = None
        best_score = None

        for c in candidates:

            score = (
                acoustic_weight * c["logit_score_norm"]
                + lm_weight * c["lm_score_norm"]
                + rank_weight * c["rank_norm"]
            )

            if best_score is None or score > best_score:

                best_score = score
                best_candidate = c

        prediction = best_candidate["text"]

        c_dist = edit_distance(
            list(reference),
            list(prediction)
        )

        ref_words = reference.split()
        pred_words = prediction.split()

        w_dist = edit_distance(
            ref_words,
            pred_words
        )

        total_cer_errors += c_dist
        total_cer_chars += max(len(reference), 1)

        total_wer_errors += w_dist
        total_wer_words += max(len(ref_words), 1)

        selected_ranks.append(
            best_candidate["rank"]
        )

    cer_value = (
        100 * total_cer_errors / total_cer_chars
    )

    wer_value = (
        100 * total_wer_errors / total_wer_words
    )

    avg_rank = (
        sum(selected_ranks) /
        len(selected_ranks)
    )

    return {
        "acoustic_weight": acoustic_weight,
        "lm_weight": lm_weight,
        "rank_weight": rank_weight,
        "cer": cer_value,
        "wer": wer_value,
        "average_selected_rank": avg_rank,
    }


# ============================================================
# BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

# Top-1 from generated N-best
baseline = evaluate_config(
    acoustic_weight=0.0,
    lm_weight=0.0,
    rank_weight=1.0,
)

print(
    f"\nN-best Top-1:"
    f"\n  CER = {baseline['cer']:.4f}%"
    f"\n  WER = {baseline['wer']:.4f}%"
)


# ============================================================
# GRID SEARCH
# ============================================================

results = []

total_configs = (
    len(ACOUSTIC_WEIGHTS)
    * len(LM_WEIGHTS)
    * len(RANK_WEIGHTS)
)

print("\n" + "=" * 70)
print("GRID SEARCH")
print("=" * 70)

print(f"\nConfigurations: {total_configs}")

counter = 0

for aw in ACOUSTIC_WEIGHTS:

    for lw in LM_WEIGHTS:

        for rw in RANK_WEIGHTS:

            counter += 1

            result = evaluate_config(
                acoustic_weight=aw,
                lm_weight=lw,
                rank_weight=rw,
            )

            results.append(result)

            if counter % 100 == 0:

                print(
                    f"[{counter}/{total_configs}] "
                    f"CER={result['cer']:.4f}% "
                    f"WER={result['wer']:.4f}%"
                )


# ============================================================
# SORT RESULTS
# ============================================================

best_cer = sorted(
    results,
    key=lambda x: x["cer"]
)[:10]

best_wer = sorted(
    results,
    key=lambda x: x["wer"]
)[:10]


# ============================================================
# DISPLAY
# ============================================================

print("\n" + "=" * 70)
print("TOP CONFIGURATIONS BY CER")
print("=" * 70)

for i, r in enumerate(best_cer, start=1):

    print(
        f"{i:2d}. "
        f"A={r['acoustic_weight']:.2f} "
        f"LM={r['lm_weight']:.2f} "
        f"R={r['rank_weight']:.2f} | "
        f"CER={r['cer']:.4f}% | "
        f"WER={r['wer']:.4f}% | "
        f"AvgRank={r['average_selected_rank']:.2f}"
    )


print("\n" + "=" * 70)
print("TOP CONFIGURATIONS BY WER")
print("=" * 70)

for i, r in enumerate(best_wer, start=1):

    print(
        f"{i:2d}. "
        f"A={r['acoustic_weight']:.2f} "
        f"LM={r['lm_weight']:.2f} "
        f"R={r['rank_weight']:.2f} | "
        f"CER={r['cer']:.4f}% | "
        f"WER={r['wer']:.4f}% | "
        f"AvgRank={r['average_selected_rank']:.2f}"
    )


# ============================================================
# BEST RESULTS
# ============================================================

best_cer_result = best_cer[0]
best_wer_result = best_wer[0]

print("\n" + "=" * 70)
print("BEST SCORE-FUSION RESULTS")
print("=" * 70)

print("\nBEST CER:")
print(
    f"  CER = {best_cer_result['cer']:.4f}%"
)
print(
    f"  WER = {best_cer_result['wer']:.4f}%"
)
print(
    f"  Acoustic weight = "
    f"{best_cer_result['acoustic_weight']:.2f}"
)
print(
    f"  LM weight = "
    f"{best_cer_result['lm_weight']:.2f}"
)
print(
    f"  Rank weight = "
    f"{best_cer_result['rank_weight']:.2f}"
)

print("\nBEST WER:")
print(
    f"  CER = {best_wer_result['cer']:.4f}%"
)
print(
    f"  WER = {best_wer_result['wer']:.4f}%"
)
print(
    f"  Acoustic weight = "
    f"{best_wer_result['acoustic_weight']:.2f}"
)
print(
    f"  LM weight = "
    f"{best_wer_result['lm_weight']:.2f}"
)
print(
    f"  Rank weight = "
    f"{best_wer_result['rank_weight']:.2f}"
)


# ============================================================
# SAVE
# ============================================================

output = {
    "n_best_file": str(NBEST_JSON),
    "num_samples": len(prepared),
    "n_best": data["n_best"],
    "baseline": baseline,
    "best_cer": best_cer_result,
    "best_wer": best_wer_result,
    "top_10_cer": best_cer,
    "top_10_wer": best_wer,
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
        indent=2,
    )


print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_JSON)