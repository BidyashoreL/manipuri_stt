import json
import re
import unicodedata
from pathlib import Path
from collections import Counter

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


def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
                prev[j - 1] + cost,
            )

        prev = curr

    return prev[m]


def cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 1.0

    return edit_distance(list(ref), list(hyp)) / len(ref)


def wer(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return edit_distance(ref_words, hyp_words) / len(ref_words)


with open(NBEST_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"] if isinstance(data, dict) else data

cer_rank_counter = Counter()
wer_rank_counter = Counter()

cer_best_rank = []
wer_best_rank = []

cer_improved = 0
wer_improved = 0

total = 0

# Exact-match counts
exact_match_rank = Counter()

# Distribution of best CER/WER
best_cers = []
best_wers = []

for sample in samples:

    reference = normalize_text(
        sample.get("reference", sample.get("text", ""))
    )

    hypotheses = sample.get(
        "hypotheses",
        sample.get("nbest", [])
    )

    if not hypotheses:
        continue

    hyps = []

    for h in hypotheses:
        if isinstance(h, dict):
            text = h.get("text", "")
        else:
            text = h[0]

        hyps.append(normalize_text(text))

    total += 1

    # --------------------------------------------------------
    # CER
    # --------------------------------------------------------

    cer_scores = [
        cer(reference, h)
        for h in hyps
    ]

    best_cer = min(cer_scores)
    best_cers.append(best_cer)

    best_cer_rank = cer_scores.index(best_cer) + 1
    cer_best_rank.append(best_cer_rank)

    cer_rank_counter[best_cer_rank] += 1

    if best_cer < cer_scores[0]:
        cer_improved += 1

    # --------------------------------------------------------
    # WER
    # --------------------------------------------------------

    wer_scores = [
        wer(reference, h)
        for h in hyps
    ]

    best_wer = min(wer_scores)
    best_wers.append(best_wer)

    best_wer_rank = wer_scores.index(best_wer) + 1
    wer_best_rank.append(best_wer_rank)

    wer_rank_counter[best_wer_rank] += 1

    if best_wer < wer_scores[0]:
        wer_improved += 1

    # --------------------------------------------------------
    # EXACT REFERENCE
    # --------------------------------------------------------

    for rank, hyp in enumerate(hyps, start=1):
        if hyp == reference:
            exact_match_rank[rank] += 1
            break


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("N-BEST RANK ANALYSIS")
print("=" * 70)

print(f"\nSamples evaluated: {total}")
print("N-best size: 50")

print("\n" + "-" * 70)
print("BEST CER HYPOTHESIS RANK")
print("-" * 70)

for rank in sorted(cer_rank_counter):
    count = cer_rank_counter[rank]
    pct = 100 * count / total
    print(f"Rank {rank:2d}: {count:4d} samples ({pct:6.2f}%)")

print("\n" + "-" * 70)
print("BEST WER HYPOTHESIS RANK")
print("-" * 70)

for rank in sorted(wer_rank_counter):
    count = wer_rank_counter[rank]
    pct = 100 * count / total
    print(f"Rank {rank:2d}: {count:4d} samples ({pct:6.2f}%)")


# ============================================================
# CUMULATIVE RANKS
# ============================================================

print("\n" + "-" * 70)
print("CUMULATIVE ORACLE COVERAGE")
print("-" * 70)

for k in [1, 2, 3, 5, 10, 20, 30, 40, 50]:

    cer_covered = sum(
        1 for r in cer_best_rank
        if r <= k
    )

    wer_covered = sum(
        1 for r in wer_best_rank
        if r <= k
    )

    print(
        f"Top-{k:2d}: "
        f"CER {cer_covered:4d}/{total} "
        f"({100 * cer_covered / total:6.2f}%) | "
        f"WER {wer_covered:4d}/{total} "
        f"({100 * wer_covered / total:6.2f}%)"
    )


# ============================================================
# EXACT REFERENCE
# ============================================================

exact_total = sum(exact_match_rank.values())

print("\n" + "-" * 70)
print("EXACT REFERENCE MATCH")
print("-" * 70)

print(
    f"Exact reference found in N-best: "
    f"{exact_total}/{total} "
    f"({100 * exact_total / total:.2f}%)"
)

for rank in sorted(exact_match_rank):
    print(
        f"  Rank {rank:2d}: "
        f"{exact_match_rank[rank]} samples"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"\nBetter CER hypothesis exists: "
    f"{cer_improved}/{total} "
    f"({100 * cer_improved / total:.2f}%)"
)

print(
    f"Better WER hypothesis exists: "
    f"{wer_improved}/{total} "
    f"({100 * wer_improved / total:.2f}%)"
)

print(
    f"\nMedian best-CER rank: "
    f"{sorted(cer_best_rank)[len(cer_best_rank)//2]}"
)

print(
    f"Median best-WER rank: "
    f"{sorted(wer_best_rank)[len(wer_best_rank)//2]}"
)

print("\n" + "=" * 70)