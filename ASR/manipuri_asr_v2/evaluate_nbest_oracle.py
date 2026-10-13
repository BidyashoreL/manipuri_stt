import json
import re
import unicodedata
from pathlib import Path

# ============================================================
# PATH
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

# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(ref, hyp):
    """
    Standard Levenshtein distance.
    """
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
    """
    Character Error Rate.
    """
    ref = list(ref)
    hyp = list(hyp)

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    return edit_distance(ref, hyp) / len(ref)


def wer(ref, hyp):
    """
    Word Error Rate.
    """
    ref = ref.split()
    hyp = hyp.split()

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    return edit_distance(ref, hyp) / len(ref)


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("N-BEST ORACLE EVALUATION")
print("=" * 70)

print(f"\nLoading:")
print(NBEST_JSON)

with open(NBEST_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\nSamples: {len(data)}")

# ============================================================
# HANDLE JSON STRUCTURE
# ============================================================

if isinstance(data, dict):

    if "samples" in data:
        samples = data["samples"]

    elif "results" in data:
        samples = data["results"]

    else:
        raise RuntimeError(
            f"Unknown JSON dictionary structure. Keys: {list(data.keys())}"
        )

elif isinstance(data, list):
    samples = data

else:
    raise RuntimeError("Unexpected JSON format.")


# ============================================================
# EVALUATION
# ============================================================

oracle_cer_values = []
oracle_wer_values = []

top1_cer_values = []
top1_wer_values = []

oracle_cer_sum = 0.0
oracle_wer_sum = 0.0

top1_cer_sum = 0.0
top1_wer_sum = 0.0

oracle_cer_ref_total = 0
oracle_wer_ref_total = 0

oracle_cer_wins = 0
oracle_wer_wins = 0

total = 0

for idx, sample in enumerate(samples, start=1):

    reference = normalize_text(
        sample.get("reference", sample.get("text", ""))
    )

    hypotheses = sample.get(
        "hypotheses",
        sample.get("nbest", [])
    )

    if not hypotheses:
        print(f"[WARNING] Sample {idx} has no hypotheses.")
        continue

    # --------------------------------------------------------
    # Extract hypothesis text
    # --------------------------------------------------------

    texts = []

    for h in hypotheses:

        if isinstance(h, dict):
            text = h.get("text", "")

        elif isinstance(h, (list, tuple)):
            text = h[0]

        else:
            text = str(h)

        texts.append(normalize_text(text))

    # --------------------------------------------------------
    # TOP-1
    # --------------------------------------------------------

    top1 = texts[0]

    top1_c = cer(reference, top1)
    top1_w = wer(reference, top1)

    # --------------------------------------------------------
    # ORACLE
    # --------------------------------------------------------

    candidate_cers = [
        cer(reference, h)
        for h in texts
    ]

    candidate_wers = [
        wer(reference, h)
        for h in texts
    ]

    best_c = min(candidate_cers)
    best_w = min(candidate_wers)

    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    top1_cer_values.append(top1_c)
    top1_wer_values.append(top1_w)

    oracle_cer_values.append(best_c)
    oracle_wer_values.append(best_w)

    top1_cer_sum += top1_c * max(len(reference), 1)
    oracle_cer_sum += best_c * max(len(reference), 1)

    ref_words = max(len(reference.split()), 1)

    top1_wer_sum += top1_w * ref_words
    oracle_wer_sum += best_w * ref_words

    oracle_cer_ref_total += max(len(reference), 1)
    oracle_wer_ref_total += ref_words

    if best_c < top1_c:
        oracle_cer_wins += 1

    if best_w < top1_w:
        oracle_wer_wins += 1

    total += 1

    if idx % 100 == 0:
        print(f"[{idx}/{len(samples)}] processed")


# ============================================================
# RESULTS
# ============================================================

top1_cer = 100 * top1_cer_sum / oracle_cer_ref_total
oracle_cer = 100 * oracle_cer_sum / oracle_cer_ref_total

top1_wer = 100 * top1_wer_sum / oracle_wer_ref_total
oracle_wer = 100 * oracle_wer_sum / oracle_wer_ref_total

cer_gain = top1_cer - oracle_cer
wer_gain = top1_wer - oracle_wer

print("\n" + "=" * 70)
print("N-BEST ORACLE RESULTS")
print("=" * 70)

print(f"\nSamples evaluated: {total}")

print("\nTOP-1:")
print(f"  CER: {top1_cer:.4f}%")
print(f"  WER: {top1_wer:.4f}%")

print("\nN-BEST ORACLE:")
print(f"  CER: {oracle_cer:.4f}%")
print(f"  WER: {oracle_wer:.4f}%")

print("\nPotential improvement:")
print(f"  CER: {cer_gain:+.4f} percentage points")
print(f"  WER: {wer_gain:+.4f} percentage points")

print("\nSamples where N-best contains a better hypothesis:")
print(
    f"  CER: {oracle_cer_wins}/{total} "
    f"({100 * oracle_cer_wins / total:.2f}%)"
)

print(
    f"  WER: {oracle_wer_wins}/{total} "
    f"({100 * oracle_wer_wins / total:.2f}%)"
)

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

if oracle_cer < top1_cer:
    print(
        "N-best contains hypotheses with substantially lower CER "
        "than the current top-1 decoder output."
    )
else:
    print(
        "N-best does not provide CER improvement over top-1."
    )

if oracle_wer < top1_wer:
    print(
        "N-best contains hypotheses with substantially lower WER "
        "than the current top-1 decoder output."
    )
else:
    print(
        "N-best does not provide WER improvement over top-1."
    )

print("\nThis establishes the upper bound for a future rescoring model.")
print("=" * 70)