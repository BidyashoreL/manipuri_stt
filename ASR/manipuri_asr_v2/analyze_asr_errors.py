import json
import csv
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

NBEST_PATH = (
    ROOT.parent
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "b3_4gram_pooled100_val.json"
)

GRU_PATH = (
    ROOT.parent
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "gru_lm"
    / "gru_lm_rescoring_results.json"
)
OUT_DIR = (
    ROOT.parent
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "error_analysis"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


# ============================================================
# LEVENSHTEIN
# ============================================================

def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):
        current = [i]

        for j, cb in enumerate(b, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )

        previous = current

    return previous[-1]


def cer(ref, hyp):
    ref_chars = list(normalize(ref))
    hyp_chars = list(normalize(hyp))

    return levenshtein(ref_chars, hyp_chars) / max(len(ref_chars), 1)


def wer(ref, hyp):
    ref_words = normalize(ref).split()
    hyp_words = normalize(hyp).split()

    return levenshtein(ref_words, hyp_words) / max(len(ref_words), 1)


# ============================================================
# LOAD
# ============================================================

print("=" * 72)
print("MANIPURI ASR ERROR ANALYSIS")
print("=" * 72)

print("\nN-BEST:")
print(NBEST_PATH)

print("\nGRU:")
print(GRU_PATH)

if not NBEST_PATH.exists():
    raise FileNotFoundError(f"N-best file not found:\n{NBEST_PATH}")

if not GRU_PATH.exists():
    raise FileNotFoundError(f"GRU result file not found:\n{GRU_PATH}")

with open(NBEST_PATH, "r", encoding="utf-8") as f:
    nbest_data = json.load(f)

with open(GRU_PATH, "r", encoding="utf-8") as f:
    gru_data = json.load(f)

samples = nbest_data["samples"]

print(f"\nValidation samples: {len(samples):,}")


# ============================================================
# UNDERSTAND GRU RESULT STRUCTURE
# ============================================================

print("\nGRU RESULT KEYS:")
print(list(gru_data.keys()))

# The rescorer result may store detailed sample predictions.
# Find the first likely list automatically.

gru_samples = None

for key, value in gru_data.items():
    if isinstance(value, list) and len(value) == len(samples):
        if len(value) > 0 and isinstance(value[0], dict):
            gru_samples = value
            print(f"Using GRU sample list: {key}")
            break

if gru_samples is None:
    raise RuntimeError(
        "Could not locate the per-sample GRU results in "
        "gru_lm_rescoring_results.json.\n"
        "Print the JSON structure and inspect it."
    )


# ============================================================
# HELPERS
# ============================================================

def get_text(obj):
    if not isinstance(obj, dict):
        return ""

    for key in [
        "text",
        "prediction",
        "hypothesis",
        "selected_text",
        "selected_hypothesis",
        "gru_text",
        "final_text",
    ]:
        if key in obj and obj[key] is not None:
            return normalize(obj[key])

    return ""


def get_rank(obj):
    if not isinstance(obj, dict):
        return None

    for key in [
        "selected_rank",
        "rank",
        "original_rank",
    ]:
        if key in obj:
            try:
                return int(obj[key])
            except Exception:
                pass

    return None


# ============================================================
# ANALYSIS
# ============================================================

rows = []

improved = []
worsened = []
unchanged = []

correct_top20 = []
correct_21_50 = []
correct_51_100 = []
missing_100 = []

gru_fixes = []
gru_breaks = []

for i, sample in enumerate(samples):

    reference = normalize(sample["reference"])
    hypotheses = sample["hypotheses"]

    # --------------------------------------------------------
    # Original N-best Top-1
    # --------------------------------------------------------

    top1 = hypotheses[0]
    top1_text = normalize(top1["text"])
    top1_rank = int(top1.get("original_rank", 1))

    # --------------------------------------------------------
    # Best candidate by CER / WER
    # --------------------------------------------------------

    best_cer = None
    best_cer_rank = None
    best_cer_text = None

    best_wer = None
    best_wer_rank = None
    best_wer_text = None

    for h in hypotheses:
        text = normalize(h["text"])
        rank = int(h.get("original_rank", 999))

        c = cer(reference, text)
        w = wer(reference, text)

        if best_cer is None or c < best_cer:
            best_cer = c
            best_cer_rank = rank
            best_cer_text = text

        if best_wer is None or w < best_wer:
            best_wer = w
            best_wer_rank = rank
            best_wer_text = text

    # --------------------------------------------------------
    # Is exact reference present?
    # --------------------------------------------------------

    exact_rank = None

    for h in hypotheses:
        if normalize(h["text"]) == reference:
            exact_rank = int(h.get("original_rank", 999))
            break

    if exact_rank is not None:
        if exact_rank <= 20:
            correct_top20.append(i)
        elif exact_rank <= 50:
            correct_21_50.append(i)
        else:
            correct_51_100.append(i)
    else:
        missing_100.append(i)

    # --------------------------------------------------------
    # Find GRU selected result
    # --------------------------------------------------------

    g = gru_samples[i]

    gru_text = get_text(g)
    gru_rank = get_rank(g)

    # Sometimes the GRU result stores a selected hypothesis object.
    if not gru_text:

        for key in [
            "selected",
            "best",
            "gru_selected",
            "final",
            "prediction",
        ]:
            value = g.get(key)

            if isinstance(value, dict):
                gru_text = get_text(value)

                if gru_rank is None:
                    gru_rank = get_rank(value)

                if gru_text:
                    break

    # Last fallback: if sample contains hypotheses
    if not gru_text and isinstance(g.get("hypothesis"), dict):
        gru_text = get_text(g["hypothesis"])

    if not gru_text:
        print(f"[WARNING] Could not extract GRU prediction for sample {i}")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    top1_cer = cer(reference, top1_text)
    top1_wer = wer(reference, top1_text)

    gru_cer = cer(reference, gru_text)
    gru_wer = wer(reference, gru_text)

    delta_cer = gru_cer - top1_cer
    delta_wer = gru_wer - top1_wer

    if delta_cer < -1e-12:
        improved.append(i)
    elif delta_cer > 1e-12:
        worsened.append(i)
    else:
        unchanged.append(i)

    if gru_cer < top1_cer:
        gru_fixes.append(i)

    if gru_cer > top1_cer:
        gru_breaks.append(i)

    rows.append(
        {
            "sample_index": i,
            "reference": reference,
            "top1": top1_text,
            "gru": gru_text,
            "top1_cer": top1_cer,
            "gru_cer": gru_cer,
            "delta_cer": delta_cer,
            "top1_wer": top1_wer,
            "gru_wer": gru_wer,
            "delta_wer": delta_wer,
            "gru_rank": gru_rank,
            "best_cer": best_cer,
            "best_cer_rank": best_cer_rank,
            "best_cer_text": best_cer_text,
            "best_wer": best_wer,
            "best_wer_rank": best_wer_rank,
            "best_wer_text": best_wer_text,
            "exact_reference_rank": exact_rank,
        }
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 72)
print("CANDIDATE POOL ANALYSIS")
print("=" * 72)

print(f"Reference found exactly in Top-20 : {len(correct_top20):4d}")
print(f"Reference found exactly in 21-50  : {len(correct_21_50):4d}")
print(f"Reference found exactly in 51-100 : {len(correct_51_100):4d}")
print(f"Reference absent from Top-100     : {len(missing_100):4d}")

print("\n" + "=" * 72)
print("GRU EFFECT")
print("=" * 72)

print(f"GRU improves CER : {len(improved):4d}")
print(f"GRU worsens CER  : {len(worsened):4d}")
print(f"Unchanged        : {len(unchanged):4d}")

# Average selected rank
ranks = [
    r["gru_rank"]
    for r in rows
    if r["gru_rank"] is not None
]

if ranks:
    print(f"\nAverage GRU rank : {sum(ranks)/len(ranks):.2f}")
    print(f"Median GRU rank  : {sorted(ranks)[len(ranks)//2]}")


# ============================================================
# SAVE ALL SAMPLES
# ============================================================

all_csv = OUT_DIR / "all_samples.csv"

fieldnames = list(rows[0].keys())

with open(all_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SAVE SORTED REPORTS
# ============================================================

def save_csv(name, data):
    path = OUT_DIR / name

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"[SAVED] {path}")


# Worst CER
save_csv(
    "worst_50_cer.csv",
    sorted(
        rows,
        key=lambda x: x["gru_cer"],
        reverse=True
    )[:50]
)

# Worst WER
save_csv(
    "worst_50_wer.csv",
    sorted(
        rows,
        key=lambda x: x["gru_wer"],
        reverse=True
    )[:50]
)

# Biggest GRU improvements
save_csv(
    "biggest_gru_improvements.csv",
    sorted(
        rows,
        key=lambda x: x["delta_cer"]
    )[:100]
)

# Biggest GRU failures
save_csv(
    "biggest_gru_failures.csv",
    sorted(
        rows,
        key=lambda x: x["delta_cer"],
        reverse=True
    )[:100]
)

# Correct answer in Top-20 but GRU selected something else
save_csv(
    "correct_in_top20_but_not_selected.csv",
    [
        r for r in rows
        if r["exact_reference_rank"] is not None
        and r["exact_reference_rank"] <= 20
        and normalize(r["gru"]) != r["reference"]
    ]
)

# Correct answer exists somewhere in 100-best
save_csv(
    "correct_in_nbest_not_selected.csv",
    [
        r for r in rows
        if r["exact_reference_rank"] is not None
        and normalize(r["gru"]) != r["reference"]
    ]
)

# Correct answer absent
save_csv(
    "reference_absent_from_top100.csv",
    [
        r for r in rows
        if r["exact_reference_rank"] is None
    ]
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "num_samples": len(rows),

    "candidate_pool": {
        "exact_reference_top20": len(correct_top20),
        "exact_reference_21_50": len(correct_21_50),
        "exact_reference_51_100": len(correct_51_100),
        "reference_absent_top100": len(missing_100),
    },

    "gru_effect": {
        "improved_cer": len(improved),
        "worsened_cer": len(worsened),
        "unchanged_cer": len(unchanged),
    },

    "gru_rank": {
        "average": (
            sum(ranks) / len(ranks)
            if ranks else None
        ),
        "median": (
            sorted(ranks)[len(ranks)//2]
            if ranks else None
        ),
    },

    "files": {
        "all_samples": str(all_csv),
    },
}

summary_path = OUT_DIR / "summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# PRINT IMPORTANT EXAMPLES
# ============================================================

print("\n" + "=" * 72)
print("TOP 15 BIGGEST GRU IMPROVEMENTS")
print("=" * 72)

for r in sorted(rows, key=lambda x: x["delta_cer"])[:15]:

    print(f"\n[{r['sample_index']}]")
    print("REF :", r["reference"])
    print("4G  :", r["top1"])
    print("GRU :", r["gru"])
    print(
        f"CER : "
        f"{r['top1_cer']*100:.2f}% -> "
        f"{r['gru_cer']*100:.2f}%"
    )
    print("GRU rank:", r["gru_rank"])


print("\n" + "=" * 72)
print("TOP 15 BIGGEST GRU FAILURES")
print("=" * 72)

for r in sorted(rows, key=lambda x: x["delta_cer"], reverse=True)[:15]:

    print(f"\n[{r['sample_index']}]")
    print("REF :", r["reference"])
    print("4G  :", r["top1"])
    print("GRU :", r["gru"])
    print(
        f"CER : "
        f"{r['top1_cer']*100:.2f}% -> "
        f"{r['gru_cer']*100:.2f}%"
    )
    print("GRU rank:", r["gru_rank"])


print("\n" + "=" * 72)
print("ERROR ANALYSIS COMPLETE")
print("=" * 72)

print(f"\nOutput directory:\n{OUT_DIR}")