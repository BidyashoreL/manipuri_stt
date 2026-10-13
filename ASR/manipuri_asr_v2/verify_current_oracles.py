import json
import unicodedata
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent
NBEST = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "nbest"

BETA1 = NBEST / "b3_4gram_nbest50_val.json"
BETA2 = NBEST / "b3_4gram_nbest50_alpha05_beta20_val.json"


def norm(x):
    return " ".join(
        unicodedata.normalize("NFC", x).split()
    ).strip()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))

    for i, x in enumerate(a, 1):
        cur = [i]

        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1

            cur.append(
                min(
                    cur[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + cost
                )
            )

        prev = cur

    return prev[-1]


def cer(ref, hyp):
    ref = list(norm(ref))
    hyp = list(norm(hyp))

    if not ref:
        return 0.0 if not hyp else 1.0

    return edit_distance(ref, hyp) / len(ref)


def wer(ref, hyp):
    ref = norm(ref).split()
    hyp = norm(hyp).split()

    if not ref:
        return 0.0 if not hyp else 1.0

    return edit_distance(ref, hyp) / len(ref)


with open(BETA1, encoding="utf-8") as f:
    d1 = json.load(f)

with open(BETA2, encoding="utf-8") as f:
    d2 = json.load(f)


assert len(d1["samples"]) == 864
assert len(d2["samples"]) == 864


beta1_cer = []
beta1_wer = []

pooled_cer = []
pooled_wer = []

new_candidate_count = 0

for s1, s2 in zip(d1["samples"], d2["samples"]):

    assert s1["sample_index"] == s2["sample_index"]

    ref = s1["reference"]

    # Current beta=1.0 oracle
    h1 = [
        norm(h["text"])
        for h in s1["hypotheses"]
    ]

    c1 = [cer(ref, h) for h in h1]
    w1 = [wer(ref, h) for h in h1]

    beta1_cer.append(min(c1))
    beta1_wer.append(min(w1))

    # Pool and deduplicate
    seen = set(h1)
    pooled = list(h1)

    before = len(pooled)

    for h in s2["hypotheses"]:
        text = norm(h["text"])

        if text not in seen:
            seen.add(text)
            pooled.append(text)

    new_candidate_count += len(pooled) - before

    cp = [cer(ref, h) for h in pooled]
    wp = [wer(ref, h) for h in pooled]

    pooled_cer.append(min(cp))
    pooled_wer.append(min(wp))


b1_cer = sum(beta1_cer) / len(beta1_cer) * 100
b1_wer = sum(beta1_wer) / len(beta1_wer) * 100

p_cer = sum(pooled_cer) / len(pooled_cer) * 100
p_wer = sum(pooled_wer) / len(pooled_wer) * 100


print("=" * 70)
print("CURRENT N-BEST ORACLE VERIFICATION")
print("=" * 70)

print("\nCURRENT β=1.0:")
print(f"  CER = {b1_cer:.4f}%")
print(f"  WER = {b1_wer:.4f}%")

print("\nCURRENT POOLED β=1.0 + β=2.0:")
print(f"  CER = {p_cer:.4f}%")
print(f"  WER = {p_wer:.4f}%")

print("\nCHANGE FROM CURRENT β=1.0:")
print(f"  CER = {p_cer - b1_cer:+.4f} pp")
print(f"  WER = {p_wer - b1_wer:+.4f} pp")

print("\nβ=2.0 NEW CANDIDATES:")
print(f"  Total = {new_candidate_count:,}")
print(f"  Average/sample = {new_candidate_count / 864:.2f}")

print("\nMONOTONICITY CHECK:")

if p_cer <= b1_cer + 1e-12:
    print("  CER: PASS")
else:
    print("  CER: FAIL")

if p_wer <= b1_wer + 1e-12:
    print("  WER: PASS")
else:
    print("  WER: FAIL")

print("=" * 70)