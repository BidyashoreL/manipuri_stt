from pathlib import Path
import pandas as pd
import numpy as np
import json


# ============================================================
# B3 BLANK-COLLAPSE ATTRIBUTION ANALYSIS
# ============================================================
#
# This script analyzes the existing B3 blank-bias diagnostic
# and joins it with the validation manifest.
#
# It does NOT train or modify the model.
#
# Project structure:
#
# ASR/
# ├── combined_dataset/
# │   └── val.csv
# │
# ├── checkpoints/
# │   └── v3_ctc/
# │       └── b3_narrow/
# │           └── diagnostics/
# │               └── blank_bias_per_sample.csv
# │
# └── manipuri_asr_v2/
#     └── analyze_blank_attribution.py
#
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# ...\ASR
ASR = SCRIPT_DIR.parent

# ...\ASR\checkpoints\v3_ctc\b3_narrow\diagnostics
DIAG_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
)

# Existing blank-bias diagnostic
BLANK_CSV = (
    DIAG_DIR
    / "blank_bias_per_sample.csv"
)

# Validation manifest
VAL_CSV = (
    ASR
    / "combined_dataset"
    / "val.csv"
)

# Output directory
OUT_DIR = (
    DIAG_DIR
    / "blank_attribution"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. START
# ============================================================

print("=" * 80)
print("B3 BLANK-COLLAPSE ATTRIBUTION ANALYSIS")
print("=" * 80)

print("\nProject root:")
print(ASR)

print("\nDiagnostics directory:")
print(DIAG_DIR)

print("\nBlank-bias CSV:")
print(BLANK_CSV)

print("\nValidation CSV:")
print(VAL_CSV)


# ============================================================
# 3. INPUT FILE CHECK
# ============================================================

print("\n" + "=" * 80)
print("INPUT FILE CHECK")
print("=" * 80)

if not BLANK_CSV.exists():

    raise FileNotFoundError(
        "\nBlank-bias CSV was not found:\n"
        f"{BLANK_CSV}"
    )

print("OK:", BLANK_CSV)


if not VAL_CSV.exists():

    raise FileNotFoundError(
        "\nValidation CSV was not found:\n"
        f"{VAL_CSV}"
    )

print("OK:", VAL_CSV)


# ============================================================
# 4. LOAD BLANK DIAGNOSTICS
# ============================================================

print("\n" + "=" * 80)
print("[1] LOADING BLANK-BIAS DIAGNOSTICS")
print("=" * 80)

blank = pd.read_csv(
    BLANK_CSV
)

print(
    f"Blank diagnostic rows: {len(blank):,}"
)

print("\nBlank diagnostic columns:")

for col in blank.columns:
    print("  ", col)


# ============================================================
# 5. LOAD VALIDATION MANIFEST
# ============================================================

print("\n" + "=" * 80)
print("[2] LOADING VALIDATION MANIFEST")
print("=" * 80)

val = pd.read_csv(
    VAL_CSV
)

print(
    f"Validation rows: {len(val):,}"
)

print("\nValidation columns:")

for col in val.columns:
    print("  ", col)


# ============================================================
# 6. REQUIRED COLUMN CHECK
# ============================================================

required_blank_columns = [
    "index",
    "T",
    "reference_length",
    "predicted_length",
    "pred_ref_length_ratio",
    "blank_frames",
    "nonblank_frames",
    "blank_fraction",
    "nonblank_fraction",
    "mean_blank_probability",
    "median_blank_probability",
    "mean_max_probability",
    "text",
]

missing_blank_columns = [
    col
    for col in required_blank_columns
    if col not in blank.columns
]

if missing_blank_columns:

    raise RuntimeError(
        "\nThe blank-bias CSV is missing required columns:\n"
        + "\n".join(
            f"  - {x}"
            for x in missing_blank_columns
        )
    )


# ============================================================
# 7. ROW ALIGNMENT
# ============================================================

print("\n" + "=" * 80)
print("[3] CHECKING SAMPLE ALIGNMENT")
print("=" * 80)

if len(blank) != len(val):

    raise RuntimeError(
        "\nRow count mismatch!\n"
        f"Blank diagnostic rows = {len(blank)}\n"
        f"Validation rows       = {len(val)}"
    )

print(
    f"Row counts match: {len(blank):,} = {len(val):,}"
)


# The blank diagnostic was created using the same validation
# order, so row-order alignment is intentional.
blank = blank.reset_index(
    drop=True
)

val = val.reset_index(
    drop=True
)


# ============================================================
# 8. MERGE
# ============================================================

print(
    "\nMerging validation metadata with blank diagnostics..."
)

blank_prefixed = blank.add_prefix(
    "blank_"
)

merged = pd.concat(
    [
        val,
        blank_prefixed
    ],
    axis=1
)

print(
    f"Merged rows: {len(merged):,}"
)


# ============================================================
# 9. NUMERIC CONVERSION
# ============================================================

numeric_columns = [
    "blank_index",
    "blank_T",
    "blank_reference_length",
    "blank_predicted_length",
    "blank_pred_ref_length_ratio",
    "blank_blank_frames",
    "blank_nonblank_frames",
    "blank_blank_fraction",
    "blank_nonblank_fraction",
    "blank_mean_blank_probability",
    "blank_median_blank_probability",
    "blank_mean_max_probability",
]

for col in numeric_columns:

    if col in merged.columns:

        merged[col] = pd.to_numeric(
            merged[col],
            errors="coerce"
        )


# ============================================================
# 10. COLLAPSE CLASSIFICATION
# ============================================================
#
# SEVERE:
#   blank fraction >= 0.95
#   OR predicted/reference ratio <= 0.10
#
# HIGH:
#   blank fraction >= 0.90
#   OR predicted/reference ratio <= 0.50
#
# MODERATE:
#   blank fraction >= 0.80
#   OR predicted/reference ratio <= 0.75
#
# NORMAL:
#   otherwise
#
# ============================================================

def classify_collapse(row):

    blank_fraction = row[
        "blank_blank_fraction"
    ]

    length_ratio = row[
        "blank_pred_ref_length_ratio"
    ]

    if pd.isna(blank_fraction):
        blank_fraction = -np.inf

    if pd.isna(length_ratio):
        length_ratio = np.inf

    if (
        blank_fraction >= 0.95
        or length_ratio <= 0.10
    ):

        return "SEVERE"

    if (
        blank_fraction >= 0.90
        or length_ratio <= 0.50
    ):

        return "HIGH"

    if (
        blank_fraction >= 0.80
        or length_ratio <= 0.75
    ):

        return "MODERATE"

    return "NORMAL"


merged[
    "collapse_class"
] = merged.apply(
    classify_collapse,
    axis=1
)


# ============================================================
# 11. OVERALL COLLAPSE DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("OVERALL COLLAPSE DISTRIBUTION")
print("=" * 80)

collapse_order = [
    "SEVERE",
    "HIGH",
    "MODERATE",
    "NORMAL"
]

counts = (
    merged[
        "collapse_class"
    ]
    .value_counts()
    .reindex(
        collapse_order,
        fill_value=0
    )
)

percentages = (
    merged[
        "collapse_class"
    ]
    .value_counts(
        normalize=True
    )
    .reindex(
        collapse_order,
        fill_value=0
    )
    * 100
)

print("\nCounts:")

for category in collapse_order:

    print(
        f"  {category:<10} : "
        f"{counts[category]:>5,}"
    )

print("\nPercentages:")

for category in collapse_order:

    print(
        f"  {category:<10} : "
        f"{percentages[category]:>8.2f}%"
    )


# ============================================================
# 12. SOURCE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("SOURCE ANALYSIS")
print("=" * 80)

if "source" in merged.columns:

    source_summary = (
        merged
        .groupby("source")
        .agg(
            samples=(
                "source",
                "size"
            ),

            mean_blank_fraction=(
                "blank_blank_fraction",
                "mean"
            ),

            mean_blank_probability=(
                "blank_mean_blank_probability",
                "mean"
            ),

            mean_pred_ref_ratio=(
                "blank_pred_ref_length_ratio",
                "mean"
            ),

            mean_reference_length=(
                "blank_reference_length",
                "mean"
            ),

            severe=(
                "collapse_class",
                lambda x:
                (
                    x == "SEVERE"
                ).sum()
            ),

            high=(
                "collapse_class",
                lambda x:
                (
                    x == "HIGH"
                ).sum()
            ),

            moderate=(
                "collapse_class",
                lambda x:
                (
                    x == "MODERATE"
                ).sum()
            ),
        )
        .reset_index()
    )

    source_summary[
        "severe_pct"
    ] = (
        source_summary["severe"]
        / source_summary["samples"]
        * 100
    )

    source_summary[
        "high_or_severe_pct"
    ] = (
        (
            source_summary["high"]
            + source_summary["severe"]
        )
        / source_summary["samples"]
        * 100
    )

    print(
        source_summary.to_string(
            index=False
        )
    )

    source_summary.to_csv(
        OUT_DIR
        / "source_attribution.csv",
        index=False
    )

else:

    print(
        "WARNING: source column not found."
    )


# ============================================================
# 13. SPEAKER ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("SPEAKER ANALYSIS")
print("=" * 80)

if "speaker_id" in merged.columns:

    speaker_summary = (
        merged
        .groupby("speaker_id")
        .agg(
            samples=(
                "speaker_id",
                "size"
            ),

            mean_blank_fraction=(
                "blank_blank_fraction",
                "mean"
            ),

            mean_blank_probability=(
                "blank_mean_blank_probability",
                "mean"
            ),

            mean_pred_ref_ratio=(
                "blank_pred_ref_length_ratio",
                "mean"
            ),

            mean_reference_length=(
                "blank_reference_length",
                "mean"
            ),

            severe=(
                "collapse_class",
                lambda x:
                (
                    x == "SEVERE"
                ).sum()
            ),

            high=(
                "collapse_class",
                lambda x:
                (
                    x == "HIGH"
                ).sum()
            ),

            moderate=(
                "collapse_class",
                lambda x:
                (
                    x == "MODERATE"
                ).sum()
            ),
        )
        .reset_index()
    )

    speaker_summary[
        "severe_pct"
    ] = (
        speaker_summary["severe"]
        / speaker_summary["samples"]
        * 100
    )

    speaker_summary[
        "high_or_severe_pct"
    ] = (
        (
            speaker_summary["high"]
            + speaker_summary["severe"]
        )
        / speaker_summary["samples"]
        * 100
    )

    speaker_display = (
        speaker_summary[
            speaker_summary["samples"] >= 3
        ]
        .sort_values(
            [
                "high_or_severe_pct",
                "mean_blank_fraction"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    print(
        "\nSpeakers with >= 3 validation samples,"
        " ranked by collapse rate:\n"
    )

    if len(speaker_display) > 0:

        print(
            speaker_display
            .head(30)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No speaker has >= 3 validation samples."
        )

    speaker_summary.to_csv(
        OUT_DIR
        / "speaker_attribution.csv",
        index=False
    )

else:

    print(
        "WARNING: speaker_id column not found."
    )


# ============================================================
# 14. DURATION ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("DURATION ANALYSIS")
print("=" * 80)

if "duration" in merged.columns:

    merged[
        "duration_numeric"
    ] = pd.to_numeric(
        merged["duration"],
        errors="coerce"
    )

    duration_bins = [
        -np.inf,
        2,
        4,
        8,
        12,
        np.inf
    ]

    duration_labels = [
        "<=2s",
        "2-4s",
        "4-8s",
        "8-12s",
        "12+s"
    ]

    merged[
        "duration_bin"
    ] = pd.cut(
        merged["duration_numeric"],
        bins=duration_bins,
        labels=duration_labels
    )

    duration_summary = (
        merged
        .groupby(
            "duration_bin",
            observed=False
        )
        .agg(
            samples=(
                "duration_numeric",
                "size"
            ),

            mean_blank_fraction=(
                "blank_blank_fraction",
                "mean"
            ),

            mean_blank_probability=(
                "blank_mean_blank_probability",
                "mean"
            ),

            mean_pred_ref_ratio=(
                "blank_pred_ref_length_ratio",
                "mean"
            ),

            mean_reference_length=(
                "blank_reference_length",
                "mean"
            ),

            severe=(
                "collapse_class",
                lambda x:
                (
                    x == "SEVERE"
                ).sum()
            ),

            high=(
                "collapse_class",
                lambda x:
                (
                    x == "HIGH"
                ).sum()
            ),

            moderate=(
                "collapse_class",
                lambda x:
                (
                    x == "MODERATE"
                ).sum()
            ),
        )
        .reset_index()
    )

    duration_summary[
        "severe_pct"
    ] = (
        duration_summary["severe"]
        / duration_summary["samples"]
        * 100
    )

    duration_summary[
        "high_or_severe_pct"
    ] = (
        (
            duration_summary["high"]
            + duration_summary["severe"]
        )
        / duration_summary["samples"]
        * 100
    )

    print(
        duration_summary.to_string(
            index=False
        )
    )

    duration_summary.to_csv(
        OUT_DIR
        / "duration_attribution.csv",
        index=False
    )

else:

    print(
        "WARNING: duration column not found."
    )


# ============================================================
# 15. REFERENCE LENGTH ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("REFERENCE LENGTH ANALYSIS")
print("=" * 80)

reference_length_bins = [
    0,
    5,
    10,
    20,
    40,
    80,
    120,
    np.inf
]

reference_length_labels = [
    "1-5",
    "6-10",
    "11-20",
    "21-40",
    "41-80",
    "81-120",
    "121+"
]

merged[
    "ref_length_bin"
] = pd.cut(
    merged["blank_reference_length"],
    bins=reference_length_bins,
    labels=reference_length_labels
)

length_summary = (
    merged
    .groupby(
        "ref_length_bin",
        observed=False
    )
    .agg(
        samples=(
            "blank_reference_length",
            "size"
        ),

        mean_blank_fraction=(
            "blank_blank_fraction",
            "mean"
        ),

        mean_blank_probability=(
            "blank_mean_blank_probability",
            "mean"
        ),

        mean_pred_ref_ratio=(
            "blank_pred_ref_length_ratio",
            "mean"
        ),

        severe=(
            "collapse_class",
            lambda x:
            (
                x == "SEVERE"
            ).sum()
        ),

        high=(
            "collapse_class",
            lambda x:
            (
                x == "HIGH"
            ).sum()
        ),

        moderate=(
            "collapse_class",
            lambda x:
            (
                x == "MODERATE"
            ).sum()
        ),
    )
    .reset_index()
)

length_summary[
    "severe_pct"
] = (
    length_summary["severe"]
    / length_summary["samples"]
    * 100
)

length_summary[
    "high_or_severe_pct"
] = (
    (
        length_summary["high"]
        + length_summary["severe"]
    )
    / length_summary["samples"]
    * 100
)

print(
    length_summary.to_string(
        index=False
    )
)

length_summary.to_csv(
    OUT_DIR
    / "reference_length_attribution.csv",
    index=False
)


# ============================================================
# 16. TOP 30 COLLAPSE CASES
# ============================================================

print("\n" + "=" * 80)
print("TOP 30 BLANK-COLLAPSE CASES")
print("=" * 80)

worst = (
    merged
    .sort_values(
        [
            "blank_blank_fraction",
            "blank_mean_blank_probability"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(30)
)

display_columns = [
    "blank_index",
    "source",
    "speaker_id",
    "duration",
    "text",
    "blank_T",
    "blank_reference_length",
    "blank_predicted_length",
    "blank_blank_fraction",
    "blank_mean_blank_probability",
    "blank_pred_ref_length_ratio",
    "collapse_class",
]

display_columns = [
    col
    for col in display_columns
    if col in worst.columns
]

print(
    worst[
        display_columns
    ].to_string(
        index=False
    )
)

worst[
    display_columns
].to_csv(
    OUT_DIR
    / "top_30_collapse_cases.csv",
    index=False
)


# ============================================================
# 17. ALL SEVERE CASES
# ============================================================

print("\n" + "=" * 80)
print("ALL SEVERE COLLAPSE CASES")
print("=" * 80)

severe = merged[
    merged["collapse_class"]
    == "SEVERE"
].copy()

severe = (
    severe
    .sort_values(
        [
            "blank_pred_ref_length_ratio",
            "blank_blank_fraction"
        ],
        ascending=[
            True,
            False
        ]
    )
)

print(
    f"Total severe cases: {len(severe):,}"
)

if len(severe) > 0:

    print(
        "\nSevere cases:\n"
    )

    print(
        severe[
            display_columns
        ].to_string(
            index=False
        )
    )

severe[
    display_columns
].to_csv(
    OUT_DIR
    / "all_severe_cases.csv",
    index=False
)


# ============================================================
# 18. LOWEST PRED/REF RATIOS
# ============================================================

print("\n" + "=" * 80)
print("LOWEST PREDICTED/REFERENCE LENGTH RATIOS")
print("=" * 80)

lowest_ratio = (
    merged
    .sort_values(
        [
            "blank_pred_ref_length_ratio",
            "blank_blank_fraction"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(30)
)

print(
    lowest_ratio[
        display_columns
    ].to_string(
        index=False
    )
)

lowest_ratio[
    display_columns
].to_csv(
    OUT_DIR
    / "lowest_pred_ref_ratio_cases.csv",
    index=False
)


# ============================================================
# 19. CORRELATIONS
# ============================================================

print("\n" + "=" * 80)
print("CORRELATIONS")
print("=" * 80)

correlation_columns = [
    "blank_reference_length",
    "blank_T",
    "blank_blank_fraction",
    "blank_mean_blank_probability",
    "blank_pred_ref_length_ratio",
]

correlation_columns = [
    col
    for col in correlation_columns
    if col in merged.columns
]

correlation_matrix = (
    merged[
        correlation_columns
    ]
    .corr()
)

print(
    correlation_matrix
    .round(4)
    .to_string()
)

correlation_matrix.to_csv(
    OUT_DIR
    / "correlations.csv"
)


# ============================================================
# 20. SAVE MERGED DATASET
# ============================================================

merged_output = (
    OUT_DIR
    / "merged_blank_attribution.csv"
)

merged.to_csv(
    merged_output,
    index=False
)

print(
    "\nComplete merged dataset saved:"
)

print(
    merged_output
)


# ============================================================
# 21. SAVE SUMMARY JSON
# ============================================================

summary = {
    "validation_samples": int(
        len(merged)
    ),

    "severe_cases": int(
        (
            merged["collapse_class"]
            == "SEVERE"
        ).sum()
    ),

    "high_cases": int(
        (
            merged["collapse_class"]
            == "HIGH"
        ).sum()
    ),

    "moderate_cases": int(
        (
            merged["collapse_class"]
            == "MODERATE"
        ).sum()
    ),

    "normal_cases": int(
        (
            merged["collapse_class"]
            == "NORMAL"
        ).sum()
    ),

    "severe_percent": float(
        (
            merged["collapse_class"]
            == "SEVERE"
        ).mean()
        * 100
    ),

    "high_or_severe_percent": float(
        merged["collapse_class"]
        .isin(
            [
                "HIGH",
                "SEVERE"
            ]
        )
        .mean()
        * 100
    ),
}

summary_path = (
    OUT_DIR
    / "summary.json"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nOutput directory:")
print(OUT_DIR)

print("\nFiles created:")

for path in sorted(
    OUT_DIR.iterdir()
):

    if path.is_file():

        print(
            f"  {path.name}"
        )

print("\nFinal summary:")

print(
    json.dumps(
        summary,
        indent=2
    )
)


# ============================================================
# 23. NEXT STEP
# ============================================================

print("\n" + "=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    "DO NOT RETRAIN B3 YET."
)

print(
    "We will use these results to determine whether "
    "blank-collapse is associated with:"
)

print(
    "  1. Dataset/source"
)

print(
    "  2. Speaker"
)

print(
    "  3. Duration"
)

print(
    "  4. Reference length"
)

print(
    "  5. Individual acoustic failures"
)

print(
    "Then we will decide the next B3/B4 experiment."
)