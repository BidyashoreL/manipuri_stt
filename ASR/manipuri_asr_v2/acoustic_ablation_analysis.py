from pathlib import Path
import pandas as pd
import numpy as np
import json

# ============================================================
# CONTROLLED ACOUSTIC BIN + CATASTROPHIC ABLATION ANALYSIS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

BASE = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
    / "acoustic_error_analysis"
)

INPUT = BASE / "merged_validation_acoustic_errors.csv"

OUT = BASE / "ablation_analysis"
OUT.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 80)
print("CONTROLLED ACOUSTIC BIN + CATASTROPHIC ABLATION ANALYSIS")
print("=" * 80)

print()
print("Input:")
print(INPUT)

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)

print()
print("Rows loaded:", len(df))


# ============================================================
# 1. CLEAN COLUMN NAMES
# ============================================================

df.columns = [
    str(c).strip()
    for c in df.columns
]


# ============================================================
# 2. NUMERIC COLUMNS
# ============================================================

NUMERIC = [
    "duration",
    "snr",
    "c50",
    "speaking_rate",
    "utterance_pitch_mean",
    "utterance_pitch_std",

    "beam4g_cer",
    "beam4g_wer",
    "greedy_cer",
    "greedy_wer",

    "blank_blank_fraction",
    "blank_mean_blank_probability",
    "blank_pred_ref_length_ratio",

    "blank_reference_length",
    "blank_predicted_length",
]

for col in NUMERIC:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


# ============================================================
# 3. COLLAPSE CLASS
# ============================================================

df["collapse_class_clean"] = (
    df["collapse_class"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# 4. BASIC BASELINE
# ============================================================

print()
print("=" * 80)
print("FULL 749-SAMPLE BASELINE")
print("=" * 80)

print(
    f"Samples: {len(df)}"
)

print(
    f"Mean CER: {df['beam4g_cer'].mean():.6f}"
)

print(
    f"Median CER: {df['beam4g_cer'].median():.6f}"
)

print(
    f"Mean WER: {df['beam4g_wer'].mean():.6f}"
)

print(
    f"Mean blank fraction: "
    f"{df['blank_blank_fraction'].mean():.6f}"
)

print(
    f"Mean prediction/reference ratio: "
    f"{df['blank_pred_ref_length_ratio'].mean():.6f}"
)


# ============================================================
# 5. IDENTIFY CATASTROPHIC CASES
# ============================================================

severe = df[
    df["collapse_class_clean"] == "SEVERE"
].copy()

high = df[
    df["collapse_class_clean"] == "HIGH"
].copy()

moderate = df[
    df["collapse_class_clean"] == "MODERATE"
].copy()

normal = df[
    df["collapse_class_clean"] == "NORMAL"
].copy()


print()
print("=" * 80)
print("COLLAPSE GROUPS")
print("=" * 80)

for name, group in [
    ("NORMAL", normal),
    ("MODERATE", moderate),
    ("HIGH", high),
    ("SEVERE", severe),
]:

    print(
        f"{name:<10} "
        f"n={len(group):3d} "
        f"mean CER={group['beam4g_cer'].mean():.6f} "
        f"mean blank={group['blank_blank_fraction'].mean():.6f}"
    )


# ============================================================
# 6. CATASTROPHIC ABLATION
# ============================================================

print()
print("=" * 80)
print("CATASTROPHIC ABLATION")
print("=" * 80)

without_severe = df[
    df["collapse_class_clean"] != "SEVERE"
].copy()

without_severe_high = df[
    ~df["collapse_class_clean"].isin(
        ["SEVERE", "HIGH"]
    )
].copy()


def summarize(name, data):

    return {
        "dataset": name,
        "n": len(data),
        "mean_cer": data["beam4g_cer"].mean(),
        "median_cer": data["beam4g_cer"].median(),
        "mean_wer": data["beam4g_wer"].mean(),
        "mean_blank_fraction":
            data["blank_blank_fraction"].mean(),
        "mean_pred_ref_ratio":
            data["blank_pred_ref_length_ratio"].mean(),
    }


ablation_rows = [

    summarize(
        "ALL_749",
        df
    ),

    summarize(
        "WITHOUT_4_SEVERE",
        without_severe
    ),

    summarize(
        "WITHOUT_HIGH_AND_SEVERE",
        without_severe_high
    ),
]


ablation_df = pd.DataFrame(
    ablation_rows
)


print(
    ablation_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Relative change
# ------------------------------------------------------------

full = ablation_df.iloc[0]
no_severe = ablation_df.iloc[1]

cer_change = (
    no_severe["mean_cer"]
    -
    full["mean_cer"]
)

cer_relative = (
    cer_change
    /
    full["mean_cer"]
    *
    100
)

print()
print(
    f"Removing severe CER delta: "
    f"{cer_change:+.6f}"
)

print(
    f"Relative CER change: "
    f"{cer_relative:+.2f}%"
)


# ============================================================
# 7. QUANTILE BINS
# ============================================================

print()
print("=" * 80)
print("ROBUST ACOUSTIC QUANTILE ANALYSIS")
print("=" * 80)


FEATURES = [
    "snr",
    "c50",
    "speaking_rate",
    "utterance_pitch_mean",
    "utterance_pitch_std",
    "duration",
]


quantile_outputs = []


for feature in FEATURES:

    print()
    print(
        "-" * 80
    )

    print(
        f"FEATURE: {feature}"
    )

    work = df[
        [
            feature,
            "beam4g_cer",
            "beam4g_wer",
            "blank_blank_fraction",
            "blank_mean_blank_probability",
            "blank_pred_ref_length_ratio",
            "collapse_class_clean",
        ]
    ].dropna(
        subset=[feature]
    ).copy()

    if len(work) < 20:

        print(
            "Not enough samples."
        )

        continue


    # qcut with duplicates dropped prevents failure
    try:

        work["quartile"] = pd.qcut(
            work[feature],
            q=4,
            labels=[
                "Q1_lowest",
                "Q2",
                "Q3",
                "Q4_highest",
            ],
            duplicates="drop",
        )

    except Exception as e:

        print(
            "Could not create quartiles:",
            e
        )

        continue


    grouped = (
        work
        .groupby(
            "quartile",
            observed=True
        )
        .agg(
            samples=(
                feature,
                "size"
            ),

            feature_mean=(
                feature,
                "mean"
            ),

            feature_median=(
                feature,
                "median"
            ),

            cer_mean=(
                "beam4g_cer",
                "mean"
            ),

            cer_median=(
                "beam4g_cer",
                "median"
            ),

            wer_mean=(
                "beam4g_wer",
                "mean"
            ),

            blank_fraction_mean=(
                "blank_blank_fraction",
                "mean"
            ),

            blank_probability_mean=(
                "blank_mean_blank_probability",
                "mean"
            ),

            pred_ref_ratio_mean=(
                "blank_pred_ref_length_ratio",
                "mean"
            ),
        )
        .reset_index()
    )


    grouped["feature"] = feature


    # severe rate by quartile

    severe_rates = (
        work
        .assign(
            severe=(
                work["collapse_class_clean"]
                == "SEVERE"
            )
        )
        .groupby(
            "quartile",
            observed=True
        )["severe"]
        .mean()
        .reset_index(
            name="severe_rate"
        )
    )


    grouped = grouped.merge(
        severe_rates,
        on="quartile",
        how="left"
    )


    print(
        grouped.to_string(
            index=False
        )
    )


    grouped.to_csv(
        OUT
        / f"quartiles_{feature}.csv",
        index=False
    )


    quantile_outputs.append(
        grouped
    )


# Combined quantile table

if quantile_outputs:

    all_quantiles = pd.concat(
        quantile_outputs,
        ignore_index=True
    )

    all_quantiles.to_csv(
        OUT
        / "all_acoustic_quantiles.csv",
        index=False
    )


# ============================================================
# 8. EXTREME-TAIL ANALYSIS
# ============================================================

print()
print("=" * 80)
print("ACOUSTIC EXTREME-TAIL ANALYSIS")
print("=" * 80)


tail_rows = []


for feature in FEATURES:

    work = df[
        [
            feature,
            "beam4g_cer",
            "beam4g_wer",
            "blank_blank_fraction",
            "collapse_class_clean",
        ]
    ].dropna(
        subset=[feature]
    ).copy()


    if len(work) < 20:
        continue


    q01 = work[feature].quantile(0.01)
    q05 = work[feature].quantile(0.05)
    q95 = work[feature].quantile(0.95)
    q99 = work[feature].quantile(0.99)


    # Low tail

    low5 = work[
        work[feature] <= q05
    ]

    low1 = work[
        work[feature] <= q01
    ]

    # High tail

    high95 = work[
        work[feature] >= q95
    ]

    high99 = work[
        work[feature] >= q99
    ]


    def tail_summary(name, group):

        return {
            "feature": feature,
            "tail": name,
            "n": len(group),
            "threshold":

                (
                    group[feature].max()
                    if name.startswith("LOW")
                    else group[feature].min()
                ),

            "mean_feature":
                group[feature].mean(),

            "mean_cer":
                group["beam4g_cer"].mean(),

            "median_cer":
                group["beam4g_cer"].median(),

            "mean_wer":
                group["beam4g_wer"].mean(),

            "mean_blank":
                group["blank_blank_fraction"].mean(),

            "severe_count":
                (
                    group[
                        "collapse_class_clean"
                    ]
                    .eq("SEVERE")
                    .sum()
                ),

            "severe_rate":
                (
                    group[
                        "collapse_class_clean"
                    ]
                    .eq("SEVERE")
                    .mean()
                ),
        }


    tail_rows.extend(
        [
            tail_summary(
                "LOW_1PCT",
                low1
            ),

            tail_summary(
                "LOW_5PCT",
                low5
            ),

            tail_summary(
                "HIGH_95PCT",
                high95
            ),

            tail_summary(
                "HIGH_99PCT",
                high99
            ),
        ]
    )


tails_df = pd.DataFrame(
    tail_rows
)


print(
    tails_df.to_string(
        index=False
    )
)


tails_df.to_csv(
    OUT
    / "acoustic_extreme_tails.csv",
    index=False
)


# ============================================================
# 9. SAME-SPEAKER ANALYSIS
# ============================================================

print()
print("=" * 80)
print("SAME-SPEAKER ANALYSIS OF CATASTROPHIC SPEAKER")
print("=" * 80)


if "speaker_id" in df.columns:

    severe_speakers = (
        severe[
            "speaker_id"
        ]
        .value_counts()
    )

    if len(severe_speakers):

        target_speaker = (
            severe_speakers.index[0]
        )

        print(
            "Target speaker:",
            target_speaker
        )

        speaker_data = df[
            df["speaker_id"]
            == target_speaker
        ].copy()

        print(
            "Validation samples:",
            len(speaker_data)
        )

        print()

        cols = [
            "sample_index",
            "duration",
            "snr",
            "c50",
            "speaking_rate",
            "utterance_pitch_mean",
            "utterance_pitch_std",
            "beam4g_cer",
            "beam4g_wer",
            "blank_blank_fraction",
            "blank_pred_ref_length_ratio",
            "collapse_class_clean",
        ]

        cols = [
            c
            for c in cols
            if c in speaker_data.columns
        ]

        print(
            speaker_data[
                cols
            ]
            .sort_values(
                "sample_index"
            )
            .to_string(
                index=False
            )
        )


        speaker_data.to_csv(
            OUT
            / "catastrophic_speaker_samples.csv",
            index=False
        )


        # Compare target speaker to all others

        target = speaker_data
        others = df[
            df["speaker_id"]
            != target_speaker
        ]


        comparison = []


        for feature in FEATURES:

            target_mean = (
                target[feature]
                .mean()
            )

            others_mean = (
                others[feature]
                .mean()
            )

            target_std = (
                target[feature]
                .std()
            )

            others_std = (
                others[feature]
                .std()
            )


            comparison.append(
                {
                    "feature":
                        feature,

                    "target_speaker_mean":
                        target_mean,

                    "others_mean":
                        others_mean,

                    "target_speaker_std":
                        target_std,

                    "others_std":
                        others_std,

                    "difference":
                        target_mean
                        -
                        others_mean,
                }
            )


        comparison_df = pd.DataFrame(
            comparison
        )


        print()
        print(
            "Target speaker vs all other speakers:"
        )

        print(
            comparison_df.to_string(
                index=False
            )
        )


        comparison_df.to_csv(
            OUT
            / "catastrophic_speaker_vs_others.csv",
            index=False
        )


# ============================================================
# 10. TOP ERROR CASES
# ============================================================

print()
print("=" * 80)
print("TOP 30 B3 ERROR CASES")
print("=" * 80)


top_errors = (
    df
    .sort_values(
        [
            "beam4g_cer",
            "blank_blank_fraction",
        ],
        ascending=False
    )
    .head(30)
)


cols = [
    "sample_index",
    "speaker_id",
    "duration",
    "snr",
    "c50",
    "speaking_rate",
    "utterance_pitch_mean",
    "beam4g_cer",
    "beam4g_wer",
    "blank_blank_fraction",
    "blank_pred_ref_length_ratio",
    "collapse_class_clean",
]


cols = [
    c
    for c in cols
    if c in top_errors.columns
]


print(
    top_errors[
        cols
    ].to_string(
        index=False
    )
)


top_errors.to_csv(
    OUT
    / "top_30_error_cases.csv",
    index=False
)


# ============================================================
# 11. CLEAN SUMMARY
# ============================================================

summary = {

    "n_samples":
        int(len(df)),

    "n_severe":
        int(len(severe)),

    "n_high":
        int(len(high)),

    "n_moderate":
        int(len(moderate)),

    "n_normal":
        int(len(normal)),

    "full_mean_cer":
        float(
            df["beam4g_cer"].mean()
        ),

    "without_severe_mean_cer":
        float(
            without_severe[
                "beam4g_cer"
            ].mean()
        ),

    "without_high_severe_mean_cer":
        float(
            without_severe_high[
                "beam4g_cer"
            ].mean()
        ),

    "severe_relative_contribution_to_mean_cer":
        float(
            (
                df["beam4g_cer"].mean()
                -
                without_severe[
                    "beam4g_cer"
                ].mean()
            )
            /
            df["beam4g_cer"].mean()
            *
            100
        ),

    "speaking_rate_vs_blank_fraction":
        float(
            df[
                [
                    "speaking_rate",
                    "blank_blank_fraction",
                ]
            ]
            .corr()
            .iloc[0, 1]
        ),

    "snr_vs_cer":
        float(
            df[
                [
                    "snr",
                    "beam4g_cer",
                ]
            ]
            .corr()
            .iloc[0, 1]
        ),

    "c50_vs_cer":
        float(
            df[
                [
                    "c50",
                    "beam4g_cer",
                ]
            ]
            .corr()
            .iloc[0, 1]
        ),
}


with open(
    OUT / "ablation_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 12. FINAL
# ============================================================

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print()
print("Output directory:")
print(OUT)

print()
print("Important files:")

for file in sorted(
    OUT.glob("*")
):

    print(
        " ",
        file.name
    )

print()
print("=" * 80)
print("DO NOT RETRAIN B3 YET")
print("=" * 80)