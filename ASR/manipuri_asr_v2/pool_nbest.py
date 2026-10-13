import json
import unicodedata
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

NBEST_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
)

FILE_BETA1 = NBEST_DIR / "b3_4gram_nbest50_val.json"
FILE_BETA2 = NBEST_DIR / "b3_4gram_nbest50_alpha05_beta20_val.json"

OUTPUT_FILE = NBEST_DIR / "b3_4gram_pooled100_val.json"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_distance(a, b):
    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        cur = [i]

        for j, cb in enumerate(b, start=1):
            if ca == cb:
                cost = 0
            else:
                cost = 1

            cur.append(
                min(
                    cur[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + cost,
                )
            )

        prev = cur

    return prev[-1]


def cer(reference, hypothesis):
    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    if not reference:
        return 0.0 if not hypothesis else 1.0

    return edit_distance(
        list(reference),
        list(hypothesis),
    ) / len(reference)


def wer(reference, hypothesis):
    reference_words = normalize_text(reference).split()
    hypothesis_words = normalize_text(hypothesis).split()

    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0

    return edit_distance(
        reference_words,
        hypothesis_words,
    ) / len(reference_words)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("POOLED N-BEST EVALUATION")
print("=" * 70)

print("\nLoading:")
print(FILE_BETA1)
print(FILE_BETA2)

with open(FILE_BETA1, "r", encoding="utf-8") as f:
    data1 = json.load(f)

with open(FILE_BETA2, "r", encoding="utf-8") as f:
    data2 = json.load(f)

samples1 = data1["samples"]
samples2 = data2["samples"]

print(f"\nBeta=1.0 samples: {len(samples1)}")
print(f"Beta=2.0 samples: {len(samples2)}")

assert len(samples1) == len(samples2), "Sample count mismatch."


# ============================================================
# BUILD POOLED CANDIDATES
# ============================================================

pooled_samples = []

total_original = 0
total_unique = 0

oracle_cers = []
oracle_wers = []

beta2_unique_contribution = 0

for idx, (s1, s2) in enumerate(zip(samples1, samples2)):

    assert s1["sample_index"] == s2["sample_index"]

    reference = normalize_text(s1["reference"])

    candidates = []
    seen = set()

    # --------------------------------------------------------
    # Add beta=1.0 candidates first
    # --------------------------------------------------------

    for hyp in s1["hypotheses"]:
        text = normalize_text(hyp["text"])

        if text not in seen:
            seen.add(text)

            candidates.append({
                "source": "beta1.0",
                "original_rank": hyp["rank"],
                "text": text,
                "lm_score": hyp["lm_score"],
                "logit_score": hyp["logit_score"],
            })

    beta1_unique_count = len(candidates)

    # --------------------------------------------------------
    # Add beta=2.0 candidates
    # --------------------------------------------------------

    for hyp in s2["hypotheses"]:
        text = normalize_text(hyp["text"])

        if text not in seen:
            seen.add(text)

            candidates.append({
                "source": "beta2.0",
                "original_rank": hyp["rank"],
                "text": text,
                "lm_score": hyp["lm_score"],
                "logit_score": hyp["logit_score"],
            })

            beta2_unique_contribution += 1

    total_original += 100
    total_unique += len(candidates)

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    candidate_cers = [
        cer(reference, x["text"])
        for x in candidates
    ]

    candidate_wers = [
        wer(reference, x["text"])
        for x in candidates
    ]

    best_cer = min(candidate_cers)
    best_wer = min(candidate_wers)

    oracle_cers.append(best_cer)
    oracle_wers.append(best_wer)

    pooled_samples.append({
        "sample_index": s1["sample_index"],
        "reference": reference,
        "hypotheses": candidates,
    })

    if (idx + 1) % 100 == 0:
        print(
            f"[{idx + 1}/{len(samples1)}] "
            f"unique candidates={len(candidates)}"
        )


# ============================================================
# RESULTS
# ============================================================

avg_unique = total_unique / len(samples1)

oracle_cer = sum(oracle_cers) / len(oracle_cers) * 100
oracle_wer = sum(oracle_wers) / len(oracle_wers) * 100

print("\n" + "=" * 70)
print("POOLED N-BEST RESULTS")
print("=" * 70)

print(f"\nSamples: {len(pooled_samples)}")

print("\nCandidate count:")
print(f"  Maximum possible: 100")
print(f"  Average unique:   {avg_unique:.2f}")
print(f"  Total unique:     {total_unique:,}")

print("\nBeta=2.0 unique contribution:")
print(f"  New candidates: {beta2_unique_contribution:,}")

print(
    f"  Average new/sample: "
    f"{beta2_unique_contribution / len(samples1):.2f}"
)

print("\nPOOLED ORACLE:")
print(f"  CER = {oracle_cer:.4f}%")
print(f"  WER = {oracle_wer:.4f}%")

# ============================================================
# SAVE
# ============================================================

output = {
    "description": "Pooled 100-best candidates from 4-gram alpha=.5 beta=1.0 and beta=2.0",
    "num_samples": len(pooled_samples),
    "source_files": [
        str(FILE_BETA1),
        str(FILE_BETA2),
    ],
    "pool_size_max": 100,
    "average_unique_candidates": avg_unique,
    "total_unique_candidates": total_unique,
    "beta2_unique_contribution": beta2_unique_contribution,
    "oracle_cer": oracle_cer,
    "oracle_wer": oracle_wer,
    "samples": pooled_samples,
}

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nSaved:")
print(OUTPUT_FILE)