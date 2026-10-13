from pathlib import Path
import pandas as pd
import numpy as np
import json

# ============================================================
# B3 SPEAKER FORENSIC ANALYSIS
# ============================================================
#
# Purpose:
#   Investigate the strongest B3 blank-collapse outlier:
#
#       S4259682900336548
#
# This script is DIAGNOSTIC ONLY.
# It does NOT modify:
#   - model checkpoints
#   - train.csv
#   - val.csv
#   - official test set
#   - audio files
#
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"

DIAG_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
)

BLANK_DIR = DIAG_DIR / "blank_attribution"

BLANK_CSV = (
    BLANK_DIR
    / "merged_blank_attribution.csv"
)

OUT_DIR = (
    DIAG_DIR
    / "speaker_forensics"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET_SPEAKER = "S4259682900336548"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def text_length(x):
    """
    Number of non-space characters.
    """
    return len(
        clean_text(x).replace(" ", "")
    )


def safe_numeric(series):
    return pd.to_numeric(
        series,
        errors="coerce"
    )


def describe_series(series):
    """
    Return basic descriptive statistics.
    """
    s = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(s) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }

    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(
            s.std()
        ) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 3. START
# ============================================================

print("=" * 80)
print("B3 SPEAKER FORENSIC ANALYSIS")
print("=" * 80)

print("\nProject root:")
print(ASR)

print("\nTarget speaker:")
print(TARGET_SPEAKER)


# ============================================================
# 4. CHECK FILES
# ============================================================

print_section("FILE CHECK")

required_files = {
    "TRAIN_CSV": TRAIN_CSV,
    "VAL_CSV": VAL_CSV,
    "BLANK_CSV": BLANK_CSV,
}

for name, path in required_files.items():

    print(
        f"{name}: "
        f"{'FOUND' if path.exists() else 'MISSING'}"
    )

    print(f"  {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file missing:\n{path}"
        )


# ============================================================
# 5. LOAD MANIFESTS
# ============================================================

print_section("LOADING MANIFESTS")

train = pd.read_csv(
    TRAIN_CSV
)

val = pd.read_csv(
    VAL_CSV
)

print(
    f"Train rows : {len(train)}"
)

print(
    f"Val rows   : {len(val)}"
)

print("\nTrain columns:")
print(
    train.columns.tolist()
)

print("\nVal columns:")
print(
    val.columns.tolist()
)


# ============================================================
# 6. PREPARE MANIFESTS
# ============================================================

if "speaker_id" not in train.columns:
    raise RuntimeError(
        "speaker_id missing from train.csv"
    )

if "speaker_id" not in val.columns:
    raise RuntimeError(
        "speaker_id missing from val.csv"
    )

if "text" not in train.columns:
    raise RuntimeError(
        "text missing from train.csv"
    )

if "text" not in val.columns:
    raise RuntimeError(
        "text missing from val.csv"
    )

train["speaker_id"] = (
    train["speaker_id"]
    .astype(str)
)

val["speaker_id"] = (
    val["speaker_id"]
    .astype(str)
)

train["text_len_chars"] = (
    train["text"]
    .apply(text_length)
)

val["text_len_chars"] = (
    val["text"]
    .apply(text_length)
)

if "duration" in train.columns:

    train["duration_num"] = (
        safe_numeric(
            train["duration"]
        )
    )

else:

    train["duration_num"] = np.nan


if "duration" in val.columns:

    val["duration_num"] = (
        safe_numeric(
            val["duration"]
        )
    )

else:

    val["duration_num"] = np.nan


# ============================================================
# 7. TARGET SPEAKER
# ============================================================

target_train = train[
    train["speaker_id"] == TARGET_SPEAKER
].copy()

target_val = val[
    val["speaker_id"] == TARGET_SPEAKER
].copy()


print_section("TARGET SPEAKER")

print(
    f"Speaker ID: {TARGET_SPEAKER}"
)

print(
    f"Training samples   : {len(target_train)}"
)

print(
    f"Validation samples : {len(target_val)}"
)


# ============================================================
# 8. SPEAKER COVERAGE
# ============================================================

print_section("SPEAKER COVERAGE")

speaker_train_counts = (
    train
    .groupby("speaker_id")
    .size()
    .sort_values(
        ascending=False
    )
)

speaker_val_counts = (
    val
    .groupby("speaker_id")
    .size()
    .sort_values(
        ascending=False
    )
)

print(
    f"Unique train speakers: "
    f"{train['speaker_id'].nunique()}"
)

print(
    f"Unique val speakers  : "
    f"{val['speaker_id'].nunique()}"
)

target_train_count = (
    len(target_train)
)

target_val_count = (
    len(target_val)
)

print(
    f"\nTarget speaker train representation: "
    f"{target_train_count} samples"
)

print(
    f"Target speaker validation samples: "
    f"{target_val_count}"
)

if target_train_count > 0:

    val_train_ratio = (
        target_val_count
        / target_train_count
    )

    print(
        f"Validation/train ratio: "
        f"{val_train_ratio:.4f}"
    )

else:

    val_train_ratio = None

    print(
        "WARNING: target speaker has "
        "ZERO training samples."
    )


# ============================================================
# 9. TARGET SPEAKER VS TRAIN SPEAKER DISTRIBUTION
# ============================================================

print_section(
    "TARGET SPEAKER VS TRAIN SPEAKER DISTRIBUTION"
)

train_speaker_stats = (
    describe_series(
        speaker_train_counts
    )
)

print(
    "\nTrain count statistics across speakers:"
)

print(
    json.dumps(
        train_speaker_stats,
        indent=2
    )
)

if TARGET_SPEAKER in speaker_train_counts.index:

    target_rank_pct = (
        speaker_train_counts
        .rank(
            pct=True
        )
        .loc[TARGET_SPEAKER]
        * 100
    )

    print(
        f"\nTarget speaker frequency percentile: "
        f"{target_rank_pct:.2f}"
    )

print(
    f"\nTarget speaker train count: "
    f"{target_train_count}"
)


# ============================================================
# 10. TARGET TRAINING DETAILS
# ============================================================

print_section(
    "TARGET SPEAKER — TRAINING DETAILS"
)

if len(target_train) > 0:

    print("\nTranscript length:")

    print(
        json.dumps(
            describe_series(
                target_train[
                    "text_len_chars"
                ]
            ),
            indent=2
        )
    )

    print("\nDuration:")

    print(
        json.dumps(
            describe_series(
                target_train[
                    "duration_num"
                ]
            ),
            indent=2
        )
    )

    if "source" in target_train.columns:

        print("\nSource distribution:")

        print(
            target_train[
                "source"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    if "lang" in target_train.columns:

        print("\nLanguage distribution:")

        print(
            target_train[
                "lang"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

else:

    print(
        "NO TRAINING DATA FOR TARGET SPEAKER."
    )


# ============================================================
# 11. TARGET VALIDATION DETAILS
# ============================================================

print_section(
    "TARGET SPEAKER — VALIDATION DETAILS"
)

if len(target_val) > 0:

    print("\nTranscript length:")

    print(
        json.dumps(
            describe_series(
                target_val[
                    "text_len_chars"
                ]
            ),
            indent=2
        )
    )

    print("\nDuration:")

    print(
        json.dumps(
            describe_series(
                target_val[
                    "duration_num"
                ]
            ),
            indent=2
        )
    )

    if "source" in target_val.columns:

        print("\nSource distribution:")

        print(
            target_val[
                "source"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    if "lang" in target_val.columns:

        print("\nLanguage distribution:")

        print(
            target_val[
                "lang"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

else:

    print(
        "NO VALIDATION DATA FOR TARGET SPEAKER."
    )


# ============================================================
# 12. TRAIN VS VALIDATION COMPARISON
# ============================================================

print_section(
    "TARGET SPEAKER — TRAIN VS VALIDATION COMPARISON"
)

comparison = {
    "train_samples": len(target_train),
    "val_samples": len(target_val),
    "train_mean_text_length": (
        float(
            target_train[
                "text_len_chars"
            ].mean()
        )
        if len(target_train) > 0
        else None
    ),
    "val_mean_text_length": (
        float(
            target_val[
                "text_len_chars"
            ].mean()
        )
        if len(target_val) > 0
        else None
    ),
    "train_mean_duration": (
        float(
            target_train[
                "duration_num"
            ].mean()
        )
        if len(target_train) > 0
        else None
    ),
    "val_mean_duration": (
        float(
            target_val[
                "duration_num"
            ].mean()
        )
        if len(target_val) > 0
        else None
    ),
}

print(
    json.dumps(
        comparison,
        indent=2
    )
)


# ============================================================
# 13. LOAD MERGED BLANK ATTRIBUTION
# ============================================================

print_section(
    "LOADING B3 BLANK ATTRIBUTION"
)

blank = pd.read_csv(
    BLANK_CSV
)

print(
    f"Blank rows: {len(blank)}"
)

print("\nBlank attribution columns:")

print(
    blank.columns.tolist()
)


# ============================================================
# 14. VERIFY EXPECTED BLANK COLUMNS
# ============================================================

required_blank_columns = [
    "blank_index",
    "blank_T",
    "blank_reference_length",
    "blank_predicted_length",
    "blank_pred_ref_length_ratio",
    "blank_blank_fraction",
    "blank_mean_blank_probability",
    "blank_length_bin",
    "collapse_class",
]

missing_blank_columns = [
    c
    for c in required_blank_columns
    if c not in blank.columns
]

if missing_blank_columns:

    raise RuntimeError(
        "\nMissing expected blank columns:\n"
        + "\n".join(
            missing_blank_columns
        )
    )


# ============================================================
# 15. ALIGN VALIDATION WITH BLANK DATA
# ============================================================

blank["blank_index"] = pd.to_numeric(
    blank["blank_index"],
    errors="coerce"
).astype("Int64")

val_reset = (
    val
    .reset_index(drop=True)
    .copy()
)

val_reset["blank_index"] = (
    np.arange(
        len(val_reset)
    )
)

merged = val_reset.merge(
    blank,
    on="blank_index",
    how="left",
    suffixes=(
        "",
        "_blank"
    )
)

print(
    f"\nValidation rows : {len(val_reset)}"
)

print(
    f"Merged rows     : {len(merged)}"
)

if len(merged) != len(val):

    raise RuntimeError(
        "Merged row count does not match validation row count."
    )


# ============================================================
# 16. TARGET SPEAKER BLANK BEHAVIOR
# ============================================================

target_blank = merged[
    merged["speaker_id"]
    == TARGET_SPEAKER
].copy()

print_section(
    "TARGET SPEAKER — BLANK BEHAVIOR"
)

print(
    f"Target speaker blank rows: "
    f"{len(target_blank)}"
)


if len(target_blank) == 0:

    print(
        "NO BLANK ATTRIBUTION ROWS FOUND "
        "FOR TARGET SPEAKER."
    )

else:

    numeric_cols = [
        "blank_T",
        "blank_reference_length",
        "blank_predicted_length",
        "blank_pred_ref_length_ratio",
        "blank_blank_fraction",
        "blank_mean_blank_probability",
    ]

    for col in numeric_cols:

        target_blank[col] = pd.to_numeric(
            target_blank[col],
            errors="coerce"
        )

    cols_to_show = [
        "blank_index",
        "duration_num",
        "text_len_chars",
        "blank_T",
        "blank_reference_length",
        "blank_predicted_length",
        "blank_pred_ref_length_ratio",
        "blank_blank_fraction",
        "blank_mean_blank_probability",
        "collapse_class",
        "blank_length_bin",
        "text",
    ]

    cols_to_show = [
        c
        for c in cols_to_show
        if c in target_blank.columns
    ]

    print(
        "\nSix validation samples for target speaker:"
    )

    print(
        target_blank[
            cols_to_show
        ]
        .sort_values(
            "blank_blank_fraction",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nTarget speaker aggregate blank behavior:"
    )

    aggregate_cols = [
        "blank_T",
        "blank_reference_length",
        "blank_predicted_length",
        "blank_pred_ref_length_ratio",
        "blank_blank_fraction",
        "blank_mean_blank_probability",
    ]

    print(
        target_blank[
            aggregate_cols
        ]
        .describe()
        .T
        .to_string()
    )


# ============================================================
# 17. COLLAPSE CLASS DISTRIBUTION
# ============================================================

print_section(
    "TARGET SPEAKER COLLAPSE CLASS DISTRIBUTION"
)

if len(target_blank) > 0:

    print(
        target_blank[
            "collapse_class"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )


# ============================================================
# 18. SPEAKER-LEVEL BLANK COMPARISON
# ============================================================

print_section(
    "SPEAKER-LEVEL BLANK COLLAPSE COMPARISON"
)

speaker_blank = (
    merged
    .groupby(
        "speaker_id"
    )
    .agg(
        samples=(
            "blank_index",
            "count"
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

        mean_ref_length=(
            "blank_reference_length",
            "mean"
        ),
    )
    .reset_index()
)

speaker_blank["target"] = (
    speaker_blank[
        "speaker_id"
    ]
    == TARGET_SPEAKER
)


print("\nTarget speaker:")

print(
    speaker_blank[
        speaker_blank["target"]
    ]
    .to_string(
        index=False
    )
)


print(
    "\n10 speakers with highest mean blank fraction:"
)

print(
    speaker_blank
    .sort_values(
        "mean_blank_fraction",
        ascending=False
    )
    .head(10)
    .to_string(
        index=False
    )
)


print(
    "\n10 speakers with lowest predicted/reference ratio:"
)

print(
    speaker_blank
    .sort_values(
        "mean_pred_ref_ratio",
        ascending=True
    )
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 19. TARGET SAMPLE IDENTIFIERS
# ============================================================

print_section(
    "TARGET SPEAKER SAMPLE IDENTIFIERS"
)

id_cols = [
    "blank_index",
    "audio_file",
    "audio_path",
    "parquet_file",
    "parquet_row",
    "source",
    "speaker_id",
    "duration",
    "text",
]

available_id_cols = [
    c
    for c in id_cols
    if c in target_val.columns
]

print(
    target_val[
        available_id_cols
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE REPORTS
# ============================================================

print_section(
    "SAVING FORENSIC REPORTS"
)

target_blank.to_csv(
    OUT_DIR
    / "target_speaker_validation.csv",
    index=False
)

speaker_blank.to_csv(
    OUT_DIR
    / "speaker_blank_comparison.csv",
    index=False
)


# Validation speaker summary
speaker_val_summary = (
    val
    .groupby("speaker_id")
    .agg(
        samples=(
            "speaker_id",
            "count"
        ),

        mean_text_len=(
            "text_len_chars",
            "mean"
        ),

        median_text_len=(
            "text_len_chars",
            "median"
        ),

        mean_duration=(
            "duration_num",
            "mean"
        ),

        median_duration=(
            "duration_num",
            "median"
        ),
    )
    .reset_index()
)

speaker_val_summary.to_csv(
    OUT_DIR
    / "speaker_validation_summary.csv",
    index=False
)


if len(target_train) > 0:

    target_train.to_csv(
        OUT_DIR
        / "target_speaker_train.csv",
        index=False
    )


# ============================================================
# 21. SAVE JSON SUMMARY
# ============================================================

summary = {
    "speaker_id": TARGET_SPEAKER,

    "train_samples": int(
        target_train_count
    ),

    "val_samples": int(
        target_val_count
    ),

    "val_train_ratio": (
        float(val_train_ratio)
        if val_train_ratio is not None
        else None
    ),

    "train_text_length": (
        describe_series(
            target_train[
                "text_len_chars"
            ]
        )
        if len(target_train) > 0
        else None
    ),

    "val_text_length": (
        describe_series(
            target_val[
                "text_len_chars"
            ]
        )
        if len(target_val) > 0
        else None
    ),

    "train_duration": (
        describe_series(
            target_train[
                "duration_num"
            ]
        )
        if len(target_train) > 0
        else None
    ),

    "val_duration": (
        describe_series(
            target_val[
                "duration_num"
            ]
        )
        if len(target_val) > 0
        else None
    ),
}


if len(target_blank) > 0:

    summary[
        "mean_blank_fraction"
    ] = float(
        target_blank[
            "blank_blank_fraction"
        ].mean()
    )

    summary[
        "mean_blank_probability"
    ] = float(
        target_blank[
            "blank_mean_blank_probability"
        ].mean()
    )

    summary[
        "mean_pred_ref_ratio"
    ] = float(
        target_blank[
            "blank_pred_ref_length_ratio"
        ].mean()
    )


with open(
    OUT_DIR
    / "target_speaker_summary.json",
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
# 22. INITIAL AUTOMATIC INTERPRETATION
# ============================================================

print_section(
    "INITIAL INTERPRETATION"
)

if target_train_count == 0:

    print(
        "CRITICAL:"
    )

    print(
        "The target validation speaker does NOT "
        "appear in the training set."
    )

    print(
        "This would strongly suggest an "
        "unseen-speaker generalization problem."
    )

elif target_train_count < 20:

    print(
        "WARNING:"
    )

    print(
        f"The target speaker has only "
        f"{target_train_count} training samples."
    )

    print(
        "Speaker under-representation is "
        "plausible and should be investigated."
    )

else:

    print(
        f"The target speaker has "
        f"{target_train_count} training samples."
    )

    print(
        "The speaker is therefore represented "
        "reasonably in training."
    )

    print(
        "Speaker absence alone does NOT explain "
        "the validation collapse."
    )


if len(target_blank) > 0:

    mean_ratio = (
        target_blank[
            "blank_pred_ref_length_ratio"
        ]
        .mean()
    )

    mean_blank = (
        target_blank[
            "blank_blank_fraction"
        ]
        .mean()
    )

    severe_count = (
        target_blank[
            "collapse_class"
        ]
        .isin(
            [
                "severe",
                "high",
            ]
        )
        .sum()
    )

    print(
        "\nTarget speaker blank statistics:"
    )

    print(
        f"Mean predicted/reference ratio: "
        f"{mean_ratio:.4f}"
    )

    print(
        f"Mean blank fraction: "
        f"{mean_blank:.4f}"
    )

    print(
        f"High/severe collapse samples: "
        f"{severe_count}/{len(target_blank)}"
    )


print(
    "\nIMPORTANT:"
)

print(
    "This script is diagnostic only."
)

print(
    "No model checkpoint, manifest, "
    "official test set, or audio file was modified."
)

print(
    "\nReports saved to:"
)

print(
    OUT_DIR
)

print(
    "\nDONE."
)