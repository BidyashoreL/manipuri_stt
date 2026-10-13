from pathlib import Path
import json
import numpy as np
import pandas as pd
from collections import Counter

ROOT = Path(__file__).resolve().parent
ASR = ROOT.parent

CACHE_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "val_logit_cache"
)

OUTPUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD CACHE
# ---------------------------------------------------------

logits = np.load(CACHE_DIR / "logits.npy", mmap_mode="r")
offsets = np.load(CACHE_DIR / "offsets.npy")
texts = json.loads(
    (CACHE_DIR / "texts.json").read_text(encoding="utf-8")
)

print("=" * 80)
print("B3 CTC T/L ANALYSIS")
print("=" * 80)

print(f"Logit shape : {logits.shape}")
print(f"Samples     : {len(texts)}")


# ---------------------------------------------------------
# BASIC CTC LENGTH ANALYSIS
# ---------------------------------------------------------

rows = []

for i, text in enumerate(texts):

    start = int(offsets[i])
    end = int(offsets[i + 1])

    T = end - start

    # Character length.
    # Spaces are retained because the CTC vocabulary contains space.
    L_chars = len(text)

    # Number of non-space characters.
    L_no_space = len(text.replace(" ", ""))

    # Number of spaces.
    spaces = text.count(" ")

    # Number of repeated adjacent characters.
    # CTC needs an additional blank between repeated labels.
    repeats = 0

    for j in range(1, len(text)):
        if text[j] == text[j - 1]:
            repeats += 1

    # Minimum CTC time steps required:
    # L + number of adjacent repeated labels
    min_T = L_chars + repeats

    # Ratios
    T_over_L = T / L_chars if L_chars else np.nan
    T_over_nonspace = T / L_no_space if L_no_space else np.nan
    slack = T - min_T
    slack_ratio = slack / T if T else np.nan

    rows.append(
        {
            "index": i,
            "text": text,
            "T": T,
            "L_chars": L_chars,
            "L_no_space": L_no_space,
            "spaces": spaces,
            "adjacent_repeats": repeats,
            "min_T_required": min_T,
            "T_minus_min_required": slack,
            "T_over_L": T_over_L,
            "T_over_nonspace": T_over_nonspace,
            "slack_ratio": slack_ratio,
            "ctc_feasible": T >= min_T,
        }
    )


df = pd.DataFrame(rows)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print()
print("=" * 80)
print("GLOBAL SUMMARY")
print("=" * 80)

print(f"Samples                  : {len(df)}")
print(f"Total encoder frames T   : {df['T'].sum():,}")
print(f"Total target chars L     : {df['L_chars'].sum():,}")
print(f"Mean T                   : {df['T'].mean():.2f}")
print(f"Mean L                   : {df['L_chars'].mean():.2f}")
print(f"Median T                 : {df['T'].median():.2f}")
print(f"Median L                 : {df['L_chars'].median():.2f}")

print()
print(f"Mean T/L                : {df['T_over_L'].mean():.4f}")
print(f"Median T/L              : {df['T_over_L'].median():.4f}")

print()
print(
    f"CTC feasible samples     : "
    f"{df['ctc_feasible'].sum()} / {len(df)} "
    f"({100 * df['ctc_feasible'].mean():.2f}%)"
)

print(
    f"CTC impossible samples   : "
    f"{(~df['ctc_feasible']).sum()} / {len(df)} "
    f"({100 * (~df['ctc_feasible']).mean():.2f}%)"
)


# ---------------------------------------------------------
# DANGER ZONES
# ---------------------------------------------------------

print()
print("=" * 80)
print("T/L DANGER ZONES")
print("=" * 80)

zones = [
    ("T/L < 1.0", df["T_over_L"] < 1.0),
    ("T/L < 1.2", df["T_over_L"] < 1.2),
    ("T/L < 1.5", df["T_over_L"] < 1.5),
    ("T/L < 2.0", df["T_over_L"] < 2.0),
    ("T/L < 2.5", df["T_over_L"] < 2.5),
    ("T/L < 3.0", df["T_over_L"] < 3.0),
]

for name, mask in zones:

    count = int(mask.sum())
    pct = 100 * count / len(df)

    print(f"{name:12s}: {count:4d} / {len(df)} ({pct:6.2f}%)")


# ---------------------------------------------------------
# LENGTH BINS
# ---------------------------------------------------------

print()
print("=" * 80)
print("T/L BY TARGET LENGTH")
print("=" * 80)

bins = [0, 5, 10, 20, 40, 80, 120, 10_000]
labels = [
    "1-5",
    "6-10",
    "11-20",
    "21-40",
    "41-80",
    "81-120",
    "121+",
]

df["length_bin"] = pd.cut(
    df["L_chars"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)

length_summary = (
    df.groupby("length_bin", observed=False)
    .agg(
        samples=("index", "count"),
        mean_T=("T", "mean"),
        mean_L=("L_chars", "mean"),
        mean_T_over_L=("T_over_L", "mean"),
        median_T_over_L=("T_over_L", "median"),
        min_T_over_L=("T_over_L", "min"),
        ctc_feasible_pct=("ctc_feasible", "mean"),
    )
    .reset_index()
)

length_summary["ctc_feasible_pct"] *= 100

print(
    length_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

length_summary.to_csv(
    OUTPUT_DIR / "ctc_TL_length_summary.csv",
    index=False,
)


# ---------------------------------------------------------
# WORST T/L SAMPLES
# ---------------------------------------------------------

print()
print("=" * 80)
print("20 LOWEST T/L SAMPLES")
print("=" * 80)

worst = (
    df.sort_values(
        ["T_over_L", "L_chars"],
        ascending=[True, False],
    )
    .head(20)
)

for _, r in worst.iterrows():

    print()
    print(f"INDEX       : {int(r['index'])}")
    print(f"T           : {int(r['T'])}")
    print(f"L           : {int(r['L_chars'])}")
    print(f"Min T       : {int(r['min_T_required'])}")
    print(f"T/L         : {r['T_over_L']:.4f}")
    print(f"Slack       : {int(r['T_minus_min_required'])}")
    print(f"Feasible    : {r['ctc_feasible']}")
    print(f"TEXT        : {r['text']}")


# ---------------------------------------------------------
# CTC-IMPOSSIBLE SAMPLES
# ---------------------------------------------------------

impossible = df[~df["ctc_feasible"]].copy()

impossible.to_csv(
    OUTPUT_DIR / "ctc_impossible_samples.csv",
    index=False,
)

print()
print("=" * 80)
print("CTC IMPOSSIBLE SAMPLES")
print("=" * 80)

print(
    f"Count: {len(impossible)}"
)

if len(impossible) > 0:

    print(
        impossible[
            [
                "index",
                "T",
                "L_chars",
                "min_T_required",
                "T_over_L",
                "text",
            ]
        ]
        .head(30)
        .to_string(index=False)
    )


# ---------------------------------------------------------
# SAVE FULL DATA
# ---------------------------------------------------------

output_csv = OUTPUT_DIR / "ctc_TL_per_sample.csv"

df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print(output_csv)
print(OUTPUT_DIR / "ctc_TL_length_summary.csv")
print(OUTPUT_DIR / "ctc_impossible_samples.csv")

print()
print("DONE.")