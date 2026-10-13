from pathlib import Path
import pandas as pd
import numpy as np
import json
import warnings

warnings.filterwarnings("ignore")


# ============================================================
# GLOBAL ACOUSTIC QUALITY <-> B3 ERROR ANALYSIS
# ============================================================
#
# PURPOSE
# -------
# Determine whether acoustic properties of IndicVoices
# validation recordings are associated with B3 ASR failures.
#
#
# CORRECT DATA ALIGNMENT
# ----------------------
#
# sample_diagnostics.csv
#       sample_index = 0..863
#                    |
#                    v
# combined_dataset/val.csv
#       row order = 0..863
#                    |
#                    +---- parquet_file
#                    +---- parquet_row
#                              |
#              +---------------+---------------+
#              |                               |
#              v                               v
# IndicVoices metadata                 blank attribution
#
#
# IMPORTANT
# ---------
# We NEVER assume sample_index == manifest_index.
#
# sample_index refers to the LOCAL 864-row validation
# diagnostic ordering.
#
# ============================================================


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

DIAG_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
)

VAL_CSV = (
    ASR
    / "combined_dataset"
    / "val.csv"
)

METADATA_FILE = (
    DIAG_DIR
    / "indicvoices_metadata"
    / "all_indicvoices_metadata.csv"
)

SAMPLE_FILE = (
    DIAG_DIR
    / "sample_diagnostics.csv"
)

BLANK_FILE = (
    DIAG_DIR
    / "blank_attribution"
    / "merged_blank_attribution.csv"
)

OUT_DIR = (
    DIAG_DIR
    / "acoustic_error_analysis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CONSTANTS
# ============================================================

EXPECTED_VAL_ROWS = 864
EXPECTED_INDICVOICES_VAL = 749

ACOUSTIC_FIELDS = [
    "duration",
    "snr",
    "c50",
    "speaking_rate",
    "utterance_pitch_mean",
    "utterance_pitch_std",
]

ERROR_FIELDS = [
    "greedy_cer",
    "beam4g_cer",
    "greedy_wer",
    "beam4g_wer",
    "blank_blank_fraction",
    "blank_mean_blank_probability",
    "blank_pred_ref_length_ratio",
    "blank_reference_length",
    "blank_predicted_length",
]

TARGET_ERROR_FIELDS = [
    "beam4g_cer",
    "beam4g_wer",
    "blank_blank_fraction",
    "blank_mean_blank_probability",
    "blank_pred_ref_length_ratio",
]

COLLAPSE_CLASSES = [
    "NORMAL",
    "MODERATE",
    "HIGH",
    "SEVERE",
]


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def normalize_file(value):
    """
    Convert a parquet path into a normalized filename.

    Example:
        E:\\foo\\train-00001-of-00029.parquet

    becomes:

        train-00001-of-00029.parquet
    """

    if pd.isna(value):
        return ""

    return Path(
        str(value)
        .replace("\\", "/")
        .strip()
    ).name.lower()


def to_numeric(df, columns):
    """
    Safely convert selected columns to numeric.
    """

    for col in columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


def get_stats(series):
    """
    Return basic statistics.
    """

    x = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(x) == 0:

        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def pearson_correlation(x, y):
    """
    Pearson correlation using only paired non-null values.
    """

    temp = pd.concat(
        [
            pd.to_numeric(
                x,
                errors="coerce"
            ),
            pd.to_numeric(
                y,
                errors="coerce"
            ),
        ],
        axis=1,
    ).dropna()

    if len(temp) < 3:
        return np.nan

    return float(
        temp.iloc[:, 0].corr(
            temp.iloc[:, 1]
        )
    )


def cohens_d(group_a, group_b):
    """
    Cohen's d:
        (mean A - mean B) / pooled standard deviation
    """

    a = pd.to_numeric(
        group_a,
        errors="coerce"
    ).dropna().values

    b = pd.to_numeric(
        group_b,
        errors="coerce"
    ).dropna().values

    if len(a) < 2 or len(b) < 2:
        return np.nan

    var_a = np.var(
        a,
        ddof=1
    )

    var_b = np.var(
        b,
        ddof=1
    )

    pooled = np.sqrt(
        (
            (len(a) - 1) * var_a
            +
            (len(b) - 1) * var_b
        )
        /
        (
            len(a)
            +
            len(b)
            -
            2
        )
    )

    if pooled == 0:
        return np.nan

    return float(
        (
            np.mean(a)
            -
            np.mean(b)
        )
        /
        pooled
    )


def print_stat_line(label, series):

    s = get_stats(series)

    print(
        f"{label:<35}"
        f"n={s['n']:4d} "
        f"mean={s['mean']:10.4f} "
        f"median={s['median']:10.4f} "
        f"std={s['std']:10.4f} "
        f"min={s['min']:10.4f} "
        f"max={s['max']:10.4f}"
    )


# ============================================================
# 4. LOAD FILES
# ============================================================

section(
    "LOADING INPUT FILES"
)

required_files = {
    "Validation manifest": VAL_CSV,
    "IndicVoices metadata": METADATA_FILE,
    "B3 sample diagnostics": SAMPLE_FILE,
    "Blank attribution": BLANK_FILE,
}

for label, path in required_files.items():

    print(f"{label}:")
    print(path)
    print(
        "exists =",
        path.exists()
    )
    print()

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file does not exist:\n{path}"
        )


# ------------------------------------------------------------
# Read CSV files
# ------------------------------------------------------------

val = pd.read_csv(
    VAL_CSV
)

metadata = pd.read_csv(
    METADATA_FILE
)

sample = pd.read_csv(
    SAMPLE_FILE
)

blank = pd.read_csv(
    BLANK_FILE
)


# Clean column names

for df in [
    val,
    metadata,
    sample,
    blank,
]:

    df.columns = [
        str(c)
        .strip()
        .replace("\ufeff", "")
        for c in df.columns
    ]


print(
    f"val.csv rows    : {len(val)}"
)

print(
    f"metadata rows   : {len(metadata)}"
)

print(
    f"sample rows     : {len(sample)}"
)

print(
    f"blank rows      : {len(blank)}"
)


# ============================================================
# 5. VALIDATION MANIFEST
# ============================================================

section(
    "VALIDATION MANIFEST"
)

if len(val) != EXPECTED_VAL_ROWS:

    print(
        f"WARNING: expected {EXPECTED_VAL_ROWS} "
        f"validation rows, found {len(val)}"
    )

else:

    print(
        f"Validation rows: {len(val)}"
    )


required_val_columns = [
    "parquet_file",
    "parquet_row",
]

for col in required_val_columns:

    if col not in val.columns:

        raise KeyError(
            f"val.csv is missing required column: {col}"
        )


print()
print(
    "Validation columns:"
)

print(
    val.columns.tolist()
)


# ============================================================
# 6. BUILD LOCAL SAMPLE INDEX
# ============================================================

section(
    "BUILDING SAMPLE INDEX -> VALIDATION MANIFEST MAPPING"
)

# IMPORTANT:
#
# sample_diagnostics.csv uses:
#
#     sample_index = 0..863
#
# This corresponds to the row ordering of val.csv.
#
# We therefore explicitly recreate that mapping.

val = val.reset_index(
    drop=True
)

val["sample_index"] = np.arange(
    len(val)
)


print(
    "sample_index range:",
    val["sample_index"].min(),
    "to",
    val["sample_index"].max()
)


print()

print(
    "First 10 sample -> parquet mappings:"
)

print(
    val[
        [
            "sample_index",
            "parquet_file",
            "parquet_row",
        ]
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# 7. NORMALIZE PARQUET KEYS
# ============================================================

section(
    "NORMALIZING PARQUET KEYS"
)

for df in [
    val,
    metadata,
    blank,
]:

    df["__parquet_file_norm"] = (
        df["parquet_file"]
        .apply(normalize_file)
    )

    df["__parquet_row_norm"] = pd.to_numeric(
        df["parquet_row"],
        errors="coerce"
    )


# ============================================================
# 8. FILTER INDICVOICES VALIDATION METADATA
# ============================================================

section(
    "FILTERING INDICVOICES VALIDATION METADATA"
)

if "split" not in metadata.columns:

    raise KeyError(
        "Metadata file does not contain 'split'."
    )


val_metadata = metadata[
    metadata["split"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("val")
].copy()


print(
    "IndicVoices validation metadata:",
    len(val_metadata)
)


if len(val_metadata) != EXPECTED_INDICVOICES_VAL:

    print(
        f"WARNING: expected "
        f"{EXPECTED_INDICVOICES_VAL} "
        f"IndicVoices validation rows."
    )


# ============================================================
# 9. REMOVE DUPLICATE METADATA KEYS
# ============================================================

metadata_dup_count = (
    val_metadata
    .duplicated(
        subset=[
            "__parquet_file_norm",
            "__parquet_row_norm",
        ],
        keep=False
    )
    .sum()
)


print(
    "Duplicate metadata parquet keys:",
    metadata_dup_count
)


val_metadata = (
    val_metadata
    .drop_duplicates(
        subset=[
            "__parquet_file_norm",
            "__parquet_row_norm",
        ],
        keep="first"
    )
    .copy()
)


# ============================================================
# 10. MERGE VALIDATION MANIFEST + METADATA
# ============================================================

section(
    "MERGING VALIDATION MANIFEST WITH INDICVOICES METADATA"
)

merged = pd.merge(

    val,

    val_metadata,

    on=[
        "__parquet_file_norm",
        "__parquet_row_norm",
    ],

    how="left",

    suffixes=(
        "_val",
        "_meta",
    ),
)


print(
    "Validation rows:",
    len(val)
)

print(
    "Merged rows:",
    len(merged)
)


# ------------------------------------------------------------
# Resolve speaker_id safely
# ------------------------------------------------------------

if "speaker_id_meta" in merged.columns:

    merged["speaker_id"] = (
        merged["speaker_id_meta"]
    )

elif "speaker_id_val" in merged.columns:

    merged["speaker_id"] = (
        merged["speaker_id_val"]
    )

elif "speaker_id" in merged.columns:

    pass

else:

    merged["speaker_id"] = np.nan


# ------------------------------------------------------------
# Resolve source safely
# ------------------------------------------------------------

if "source_meta" in merged.columns:

    merged["source"] = (
        merged["source_meta"]
    )

elif "source_val" in merged.columns:

    merged["source"] = (
        merged["source_val"]
    )


# ------------------------------------------------------------
# Metadata coverage
# ------------------------------------------------------------

print(
    "Rows with metadata speaker:",
    merged["speaker_id"].notna().sum()
)

print(
    "Rows with metadata SNR:",
    merged["snr"].notna().sum()
)

print(
    "Rows with metadata C50:",
    merged["c50"].notna().sum()
)


# ============================================================
# 11. MERGE B3 SAMPLE DIAGNOSTICS
# ============================================================

section(
    "MERGING B3 SAMPLE DIAGNOSTICS"
)

if "sample_index" not in sample.columns:

    raise KeyError(
        "sample_diagnostics.csv does not contain "
        "'sample_index'."
    )


sample["sample_index"] = pd.to_numeric(
    sample["sample_index"],
    errors="coerce"
)


# Check sample indices

sample_unique = (
    sample["sample_index"]
    .dropna()
    .astype(int)
    .unique()
)

print(
    "Unique sample indices:",
    len(sample_unique)
)

print(
    "Minimum sample index:",
    sample["sample_index"].min()
)

print(
    "Maximum sample index:",
    sample["sample_index"].max()
)


# Remove duplicate diagnostics if any

sample_dup_count = (
    sample
    .duplicated(
        subset=["sample_index"],
        keep=False
    )
    .sum()
)


print(
    "Duplicate sample diagnostic rows:",
    sample_dup_count
)


sample = (
    sample
    .drop_duplicates(
        subset=["sample_index"],
        keep="first"
    )
    .copy()
)


# Merge

merged = pd.merge(

    merged,

    sample,

    on="sample_index",

    how="left",

    suffixes=(
        "",
        "_diag",
    ),
)


print(
    "Rows after B3 merge:",
    len(merged)
)


print(
    "Rows with B3 CER:",
    merged["beam4g_cer"].notna().sum()
)


print(
    "Rows with B3 WER:",
    merged["beam4g_wer"].notna().sum()
)


# ============================================================
# 12. MERGE BLANK ATTRIBUTION
# ============================================================

section(
    "MERGING BLANK ATTRIBUTION"
)

required_blank_columns = [
    "__parquet_file_norm",
    "__parquet_row_norm",
]

for col in required_blank_columns:

    if col not in blank.columns:

        blank[col] = None


# Create normalized keys if they weren't present

blank["__parquet_file_norm"] = (
    blank["parquet_file"]
    .apply(normalize_file)
)

blank["__parquet_row_norm"] = pd.to_numeric(
    blank["parquet_row"],
    errors="coerce"
)


blank_columns = [
    "__parquet_file_norm",
    "__parquet_row_norm",

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

    "collapse_class",
]


blank_columns = [
    c
    for c in blank_columns
    if c in blank.columns
]


blank_subset = blank[
    blank_columns
].copy()


blank_dup_count = (
    blank_subset
    .duplicated(
        subset=[
            "__parquet_file_norm",
            "__parquet_row_norm",
        ],
        keep=False
    )
    .sum()
)


print(
    "Duplicate blank-attribution keys:",
    blank_dup_count
)


blank_subset = (
    blank_subset
    .drop_duplicates(
        subset=[
            "__parquet_file_norm",
            "__parquet_row_norm",
        ],
        keep="first"
    )
)


merged = pd.merge(

    merged,

    blank_subset,

    on=[
        "__parquet_file_norm",
        "__parquet_row_norm",
    ],

    how="left",
)


print(
    "Rows after blank merge:",
    len(merged)
)

print(
    "Rows with blank fraction:",
    merged[
        "blank_blank_fraction"
    ].notna().sum()
)

print(
    "Rows with collapse class:",
    merged[
        "collapse_class"
    ].notna().sum()
)


# ============================================================
# 13. CREATE CANONICAL ACOUSTIC COLUMNS
# ============================================================

# Pandas adds suffixes because duration exists in both
# val.csv and IndicVoices metadata.
#
# For acoustic-quality analysis, use the ORIGINAL
# IndicVoices metadata duration when available.

if "duration_meta" in merged.columns:
    merged["duration"] = pd.to_numeric(
        merged["duration_meta"],
        errors="coerce"
    )
elif "duration_val" in merged.columns:
    merged["duration"] = pd.to_numeric(
        merged["duration_val"],
        errors="coerce"
    )
elif "duration" in merged.columns:
    merged["duration"] = pd.to_numeric(
        merged["duration"],
        errors="coerce"
    )
else:
    merged["duration"] = np.nan


# Canonical speaker ID
if "speaker_id_meta" in merged.columns:
    merged["speaker_id"] = merged["speaker_id_meta"]
elif "speaker_id_val" in merged.columns:
    merged["speaker_id"] = merged["speaker_id_val"]


# Canonical source
if "source_meta" in merged.columns:
    merged["source"] = merged["source_meta"]
elif "source_val" in merged.columns:
    merged["source"] = merged["source_val"]


# Convert acoustic metadata to numeric
for col in [
    "duration",
    "snr",
    "c50",
    "speaking_rate",
    "utterance_pitch_mean",
    "utterance_pitch_std",
]:

    if col in merged.columns:

        merged[col] = pd.to_numeric(
            merged[col],
            errors="coerce"
        )


# Convert B3/blank metrics
for col in [
    "greedy_cer",
    "beam4g_cer",
    "greedy_wer",
    "beam4g_wer",
    "blank_blank_fraction",
    "blank_mean_blank_probability",
    "blank_pred_ref_length_ratio",
    "blank_reference_length",
    "blank_predicted_length",
]:

    if col in merged.columns:

        merged[col] = pd.to_numeric(
            merged[col],
            errors="coerce"
        )


print()
print("Canonical acoustic columns:")

for col in ACOUSTIC_FIELDS:

    print(
        f"  {col:<30} "
        f"{'OK' if col in merged.columns else 'MISSING'}"
    )


# ============================================================
# 14. CRITICAL ALIGNMENT CHECK
# ============================================================
# ============================================================
# 14. CRITICAL ALIGNMENT CHECK
# ============================================================

section(
    "FINAL ALIGNMENT CHECK"
)

print(
    f"Expected validation samples : "
    f"{EXPECTED_VAL_ROWS}"
)

print(
    f"Actual merged rows           : "
    f"{len(merged)}"
)

print(
    f"Rows with metadata speaker   : "
    f"{merged['speaker_id'].notna().sum()}"
)

print(
    f"Rows with SNR                : "
    f"{merged['snr'].notna().sum()}"
)

print(
    f"Rows with C50                : "
    f"{merged['c50'].notna().sum()}"
)

print(
    f"Rows with B3 CER             : "
    f"{merged['beam4g_cer'].notna().sum()}"
)

print(
    f"Rows with blank fraction     : "
    f"{merged['blank_blank_fraction'].notna().sum()}"
)

print(
    f"Rows with collapse class     : "
    f"{merged['collapse_class'].notna().sum()}"
)


# ------------------------------------------------------------
# Hard validation
# ------------------------------------------------------------

if len(merged) != EXPECTED_VAL_ROWS:

    raise RuntimeError(
        "\nALIGNMENT FAILURE:\n"
        f"Expected {EXPECTED_VAL_ROWS} merged rows, "
        f"got {len(merged)}."
    )


if merged["beam4g_cer"].notna().sum() != EXPECTED_VAL_ROWS:

    raise RuntimeError(
        "\nB3 diagnostic alignment failure:\n"
        "Not all 864 samples received B3 CER."
    )


# ============================================================
# 15. INDICVOICES SUBSET
# ============================================================

section(
    "SELECTING INDICVOICES VALIDATION SUBSET"
)

iv = merged[
    merged["snr"].notna()
].copy()


print(
    "IndicVoices validation rows with metadata:",
    len(iv)
)


if len(iv) != EXPECTED_INDICVOICES_VAL:

    print(
        "WARNING:"
    )

    print(
        f"Expected {EXPECTED_INDICVOICES_VAL}, "
        f"found {len(iv)}."
    )


# ============================================================
# 16. SAVE MASTER MERGED DATASET
# ============================================================

MASTER_FILE = (
    OUT_DIR
    / "merged_validation_acoustic_errors.csv"
)


iv.to_csv(
    MASTER_FILE,
    index=False,
    encoding="utf-8-sig"
)


print()
print(
    "Saved master dataset:"
)

print(
    MASTER_FILE
)


# ============================================================
# 17. GLOBAL ACOUSTIC STATISTICS
# ============================================================

section(
    "GLOBAL INDICVOICES ACOUSTIC STATISTICS"
)

global_rows = []

for field in ACOUSTIC_FIELDS:

    s = get_stats(
        iv[field]
    )

    global_rows.append(
        {
            "feature": field,
            **s,
        }
    )

    print_stat_line(
        field,
        iv[field]
    )


global_df = pd.DataFrame(
    global_rows
)


global_df.to_csv(
    OUT_DIR
    / "global_acoustic_statistics.csv",
    index=False
)


# ============================================================
# 18. B3 ERROR STATISTICS
# ============================================================

section(
    "GLOBAL B3 ERROR STATISTICS"
)

error_rows = []

for field in ERROR_FIELDS:

    s = get_stats(
        iv[field]
    )

    error_rows.append(
        {
            "metric": field,
            **s,
        }
    )

    print_stat_line(
        field,
        iv[field]
    )


error_df = pd.DataFrame(
    error_rows
)


error_df.to_csv(
    OUT_DIR
    / "global_error_statistics.csv",
    index=False
)


# ============================================================
# 19. COLLAPSE CLASS
# ============================================================

section(
    "COLLAPSE CLASS DISTRIBUTION"
)

iv["collapse_class_clean"] = (
    iv["collapse_class"]
    .astype(str)
    .str.strip()
    .str.upper()
)


collapse_counts = (
    iv["collapse_class_clean"]
    .value_counts()
)


print(
    collapse_counts
)

print()

for cls, count in collapse_counts.items():

    print(
        f"{cls:<15}"
        f"{count:4d} "
        f"({count / len(iv) * 100:.3f}%)"
    )


# ============================================================
# 20. ACOUSTIC FEATURES BY COLLAPSE CLASS
# ============================================================

section(
    "ACOUSTIC FEATURES BY COLLAPSE CLASS"
)

group_rows = []

for cls in COLLAPSE_CLASSES:

    group = iv[
        iv["collapse_class_clean"]
        == cls
    ]

    if len(group) == 0:
        continue

    print()
    print(
        f"========== {cls} "
        f"(n={len(group)}) =========="
    )

    row = {
        "collapse_class": cls,
        "samples": len(group),
    }

    for field in ACOUSTIC_FIELDS:

        s = get_stats(
            group[field]
        )

        row[
            f"{field}_mean"
        ] = s["mean"]

        row[
            f"{field}_median"
        ] = s["median"]

        row[
            f"{field}_std"
        ] = s["std"]

        print(
            f"{field:<30}"
            f"mean={s['mean']:.4f} "
            f"median={s['median']:.4f} "
            f"std={s['std']:.4f}"
        )


    for field in TARGET_ERROR_FIELDS:

        s = get_stats(
            group[field]
        )

        row[
            f"{field}_mean"
        ] = s["mean"]

        row[
            f"{field}_median"
        ] = s["median"]


    group_rows.append(
        row
    )


collapse_summary = pd.DataFrame(
    group_rows
)


collapse_summary.to_csv(
    OUT_DIR
    / "collapse_group_summary.csv",
    index=False
)


# ============================================================
# 21. ACOUSTIC <-> ERROR CORRELATIONS
# ============================================================

section(
    "ACOUSTIC <-> B3 ERROR CORRELATIONS"
)

corr_rows = []

for acoustic in ACOUSTIC_FIELDS:

    for error in TARGET_ERROR_FIELDS:

        r = pearson_correlation(
            iv[acoustic],
            iv[error]
        )

        corr_rows.append(
            {
                "acoustic_feature":
                    acoustic,

                "error_metric":
                    error,

                "pearson_r":
                    r,

                "abs_r":
                    (
                        abs(r)
                        if not pd.isna(r)
                        else np.nan
                    ),
            }
        )

        print(
            f"{acoustic:<30}"
            f"vs "
            f"{error:<36}"
            f"r={r:.5f}"
        )


corr_df = pd.DataFrame(
    corr_rows
)


corr_df = corr_df.sort_values(
    "abs_r",
    ascending=False
)


CORR_FILE = (
    OUT_DIR
    / "correlation_matrix.csv"
)


corr_df.to_csv(
    CORR_FILE,
    index=False
)


# ============================================================
# 22. STRONGEST RELATIONSHIPS
# ============================================================

section(
    "STRONGEST ACOUSTIC <-> ERROR RELATIONSHIPS"
)

print(
    corr_df
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# 23. NORMAL VS SEVERE EFFECT SIZES
# ============================================================

section(
    "NORMAL VS SEVERE EFFECT SIZES"
)

normal = iv[
    iv["collapse_class_clean"]
    == "NORMAL"
].copy()


severe = iv[
    iv["collapse_class_clean"]
    == "SEVERE"
].copy()


print(
    "NORMAL samples:",
    len(normal)
)

print(
    "SEVERE samples:",
    len(severe)
)


effect_rows = []


for field in ACOUSTIC_FIELDS:

    normal_mean = pd.to_numeric(
        normal[field],
        errors="coerce"
    ).mean()

    severe_mean = pd.to_numeric(
        severe[field],
        errors="coerce"
    ).mean()

    d = cohens_d(
        severe[field],
        normal[field]
    )


    effect_rows.append(
        {
            "feature":
                field,

            "normal_mean":
                normal_mean,

            "severe_mean":
                severe_mean,

            "cohens_d_severe_minus_normal":
                d,
        }
    )


    print(
        f"{field:<30}"
        f"NORMAL={normal_mean:.4f} "
        f"SEVERE={severe_mean:.4f} "
        f"d={d:.4f}"
    )


effect_df = pd.DataFrame(
    effect_rows
)


EFFECT_FILE = (
    OUT_DIR
    / "normal_vs_severe_effect_sizes.csv"
)


effect_df.to_csv(
    EFFECT_FILE,
    index=False
)


# ============================================================
# 24. SEVERE CASES
# ============================================================

section(
    "SEVERE CASES"
)

severe_cases = severe.copy()


severe_cases = severe_cases.sort_values(
    [
        "blank_blank_fraction",
        "beam4g_cer",
    ],
    ascending=False
)


severe_columns = [
    "sample_index",
    "speaker_id",
    "parquet_file_val",
    "parquet_row_val",
    "duration",
    "snr",
    "c50",
    "speaking_rate",
    "utterance_pitch_mean",
    "utterance_pitch_std",
    "beam4g_cer",
    "beam4g_wer",
    "blank_blank_fraction",
    "blank_mean_blank_probability",
    "blank_pred_ref_length_ratio",
    "reference",
    "hypothesis",
]


# Some columns may have names without _val
# depending on pandas merge behavior.

fallback_columns = {
    "parquet_file_val": "parquet_file",
    "parquet_row_val": "parquet_row",
}


for wanted, fallback in fallback_columns.items():

    if wanted not in severe_cases.columns:

        if fallback in severe_cases.columns:

            severe_cases[wanted] = (
                severe_cases[fallback]
            )


severe_columns = [
    c
    for c in severe_columns
    if c in severe_cases.columns
]


print(
    severe_cases[
        severe_columns
    ]
    .head(30)
    .to_string(
        index=False
    )
)


SEVERE_FILE = (
    OUT_DIR
    / "severe_cases.csv"
)


severe_cases.to_csv(
    SEVERE_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 25. SPEAKER-LEVEL ANALYSIS
# ============================================================

section(
    "SPEAKER-LEVEL FAILURE ANALYSIS"
)

speaker_rows = []


for speaker, group in iv.groupby(
    "speaker_id",
    dropna=False
):

    n = len(group)

    severe_count = (
        group[
            "collapse_class_clean"
        ]
        .eq("SEVERE")
        .sum()
    )

    high_count = (
        group[
            "collapse_class_clean"
        ]
        .eq("HIGH")
        .sum()
    )

    moderate_count = (
        group[
            "collapse_class_clean"
        ]
        .eq("MODERATE")
        .sum()
    )

    normal_count = (
        group[
            "collapse_class_clean"
        ]
        .eq("NORMAL")
        .sum()
    )


    speaker_rows.append(
        {
            "speaker_id":
                speaker,

            "samples":
                n,

            "severe_count":
                severe_count,

            "high_count":
                high_count,

            "moderate_count":
                moderate_count,

            "normal_count":
                normal_count,

            "severe_rate":
                (
                    severe_count / n
                    if n
                    else np.nan
                ),

            "mean_cer":
                pd.to_numeric(
                    group["beam4g_cer"],
                    errors="coerce"
                ).mean(),

            "mean_blank_fraction":
                pd.to_numeric(
                    group[
                        "blank_blank_fraction"
                    ],
                    errors="coerce"
                ).mean(),

            "mean_pred_ref_ratio":
                pd.to_numeric(
                    group[
                        "blank_pred_ref_length_ratio"
                    ],
                    errors="coerce"
                ).mean(),

            "mean_snr":
                pd.to_numeric(
                    group["snr"],
                    errors="coerce"
                ).mean(),

            "mean_c50":
                pd.to_numeric(
                    group["c50"],
                    errors="coerce"
                ).mean(),

            "mean_pitch":
                pd.to_numeric(
                    group[
                        "utterance_pitch_mean"
                    ],
                    errors="coerce"
                ).mean(),
        }
    )


speaker_df = pd.DataFrame(
    speaker_rows
)


speaker_df = speaker_df.sort_values(
    [
        "severe_rate",
        "mean_blank_fraction",
    ],
    ascending=False
)


print(
    speaker_df
    .head(20)
    .to_string(
        index=False
    )
)


SPEAKER_FILE = (
    OUT_DIR
    / "speaker_summary.csv"
)


speaker_df.to_csv(
    SPEAKER_FILE,
    index=False
)


# ============================================================
# 26. HIGH-PITCH ANALYSIS
# ============================================================

section(
    "HIGH-PITCH CHECK"
)

high_pitch = iv[
    iv[
        "utterance_pitch_mean"
    ] >= 250
].copy()


print(
    "Samples with pitch >= 250 Hz:",
    len(high_pitch)
)


if len(high_pitch):

    display_columns = [
        "sample_index",
        "speaker_id",
        "duration",
        "snr",
        "c50",
        "speaking_rate",
        "utterance_pitch_mean",
        "utterance_pitch_std",
        "beam4g_cer",
        "blank_blank_fraction",
        "collapse_class_clean",
    ]

    print(
        high_pitch[
            display_columns
        ]
        .sort_values(
            "utterance_pitch_mean",
            ascending=False
        )
        .to_string(
            index=False
        )
    )


HIGH_PITCH_FILE = (
    OUT_DIR
    / "high_pitch_cases.csv"
)


high_pitch.to_csv(
    HIGH_PITCH_FILE,
    index=False
)


# ============================================================
# 27. LOW-SNR ANALYSIS
# ============================================================

section(
    "LOW-SNR CHECK"
)

low_snr = iv[
    iv["snr"] <= 45
].copy()


print(
    "Samples with SNR <= 45:",
    len(low_snr)
)


if len(low_snr):

    display_columns = [
        "sample_index",
        "speaker_id",
        "duration",
        "snr",
        "c50",
        "utterance_pitch_mean",
        "beam4g_cer",
        "blank_blank_fraction",
        "collapse_class_clean",
    ]

    print(
        low_snr[
            display_columns
        ]
        .sort_values(
            "snr"
        )
        .head(30)
        .to_string(
            index=False
        )
    )


LOW_SNR_FILE = (
    OUT_DIR
    / "low_snr_cases.csv"
)


low_snr.to_csv(
    LOW_SNR_FILE,
    index=False
)


# ============================================================
# 28. LOW-C50 ANALYSIS
# ============================================================

section(
    "LOW-C50 CHECK"
)

low_c50 = iv[
    iv["c50"] <= 40
].copy()


print(
    "Samples with C50 <= 40:",
    len(low_c50)
)


if len(low_c50):

    display_columns = [
        "sample_index",
        "speaker_id",
        "duration",
        "snr",
        "c50",
        "utterance_pitch_mean",
        "beam4g_cer",
        "blank_blank_fraction",
        "collapse_class_clean",
    ]

    print(
        low_c50[
            display_columns
        ]
        .sort_values(
            "c50"
        )
        .head(30)
        .to_string(
            index=False
        )
    )


LOW_C50_FILE = (
    OUT_DIR
    / "low_c50_cases.csv"
)


low_c50.to_csv(
    LOW_C50_FILE,
    index=False
)


# ============================================================
# 29. ACOUSTIC OUTLIERS
# ============================================================

section(
    "ACOUSTIC OUTLIERS"
)

outlier_rows = []


for field in ACOUSTIC_FIELDS:

    x = pd.to_numeric(
        iv[field],
        errors="coerce"
    )

    mean = x.mean()
    std = x.std()

    if pd.isna(std) or std == 0:
        continue

    z = (
        x - mean
    ) / std


    mask = (
        z.abs()
        >= 2.5
    )


    for idx in iv.index[mask]:

        outlier_rows.append(
            {
                "row_index":
                    int(idx),

                "sample_index":
                    iv.loc[
                        idx,
                        "sample_index"
                    ],

                "feature":
                    field,

                "value":
                    iv.loc[
                        idx,
                        field
                    ],

                "z_score":
                    z.loc[idx],

                "speaker_id":
                    iv.loc[
                        idx,
                        "speaker_id"
                    ],

                "collapse_class":
                    iv.loc[
                        idx,
                        "collapse_class_clean"
                    ],

                "beam4g_cer":
                    iv.loc[
                        idx,
                        "beam4g_cer"
                    ],

                "blank_fraction":
                    iv.loc[
                        idx,
                        "blank_blank_fraction"
                    ],
            }
        )


outliers = pd.DataFrame(
    outlier_rows
)


OUTLIER_FILE = (
    OUT_DIR
    / "acoustic_outliers.csv"
)


outliers.to_csv(
    OUTLIER_FILE,
    index=False
)


print(
    "Outlier observations:",
    len(outliers)
)


if len(outliers):

    print(
        outliers
        .head(30)
        .to_string(
            index=False
        )
    )


# ============================================================
# 30. REPORT
# ============================================================

section(
    "GENERATING REPORT"
)

report = []

report.append(
    "GLOBAL ACOUSTIC QUALITY <-> B3 ERROR ANALYSIS"
)

report.append(
    "=" * 70
)

report.append(
    f"Validation manifest rows: {len(val)}"
)

report.append(
    f"IndicVoices validation rows: {len(iv)}"
)

report.append(
    f"B3 diagnostic rows: {len(sample)}"
)

report.append(
    f"Blank attribution rows: {len(blank)}"
)

report.append("")


# ------------------------------------------------------------
# Collapse distribution
# ------------------------------------------------------------

report.append(
    "COLLAPSE DISTRIBUTION"
)

report.append(
    "-" * 70
)


for cls, count in collapse_counts.items():

    report.append(
        f"{cls:15s}"
        f"{count:5d}"
        f" "
        f"({count / len(iv) * 100:.3f}%)"
    )


report.append("")


# ------------------------------------------------------------
# Strongest correlations
# ------------------------------------------------------------

report.append(
    "STRONGEST ACOUSTIC <-> ERROR CORRELATIONS"
)

report.append(
    "-" * 70
)


for _, row in corr_df.head(15).iterrows():

    report.append(
        f"{row['acoustic_feature']:30s}"
        f" vs "
        f"{row['error_metric']:36s}"
        f" r={row['pearson_r']:.5f}"
    )


report.append("")


# ------------------------------------------------------------
# Effect sizes
# ------------------------------------------------------------

report.append(
    "NORMAL VS SEVERE EFFECT SIZES"
)

report.append(
    "-" * 70
)


for _, row in effect_df.iterrows():

    report.append(
        f"{row['feature']:30s}"
        f" NORMAL={row['normal_mean']:.4f}"
        f" SEVERE={row['severe_mean']:.4f}"
        f" d={row['cohens_d_severe_minus_normal']:.4f}"
    )


report.append("")


report.append(
    "INTERPRETATION NOTE"
)

report.append(
    "-" * 70
)

report.append(
    "Correlation does not establish causation."
)

report.append(
    "Acoustic outliers should not automatically be removed."
)

report.append(
    "Use these results to select the next controlled experiment."
)


REPORT_FILE = (
    OUT_DIR
    / "report.txt"
)


REPORT_FILE.write_text(
    "\n".join(report),
    encoding="utf-8"
)


# ============================================================
# 31. JSON SUMMARY
# ============================================================

summary = {

    "validation_manifest_rows":
        int(len(val)),

    "indicvoices_validation_rows":
        int(len(iv)),

    "b3_diagnostic_rows":
        int(len(sample)),

    "blank_attribution_rows":
        int(len(blank)),

    "final_alignment": {

        "merged_rows":
            int(len(merged)),

        "metadata_speaker_rows":
            int(
                merged[
                    "speaker_id"
                ].notna().sum()
            ),

        "snr_rows":
            int(
                merged[
                    "snr"
                ].notna().sum()
            ),

        "c50_rows":
            int(
                merged[
                    "c50"
                ].notna().sum()
            ),

        "b3_cer_rows":
            int(
                merged[
                    "beam4g_cer"
                ].notna().sum()
            ),

        "blank_fraction_rows":
            int(
                merged[
                    "blank_blank_fraction"
                ].notna().sum()
            ),
    },

    "collapse_distribution":
        {
            str(k): int(v)
            for k, v
            in collapse_counts.items()
        },

    "strongest_correlations":
        (
            corr_df
            .head(15)
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),

    "normal_vs_severe_effect_sizes":
        (
            effect_df
            .replace(
                {np.nan: None}
            )
            .to_dict(
                orient="records"
            )
        ),

    "output_directory":
        str(OUT_DIR),
}


JSON_FILE = (
    OUT_DIR
    / "analysis_summary.json"
)


with open(
    JSON_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
        default=str
    )


# ============================================================
# 32. FINAL OUTPUT
# ============================================================

section(
    "ANALYSIS COMPLETE"
)

print(
    "Output directory:"
)

print(
    OUT_DIR
)

print()

print(
    "Generated files:"
)

output_files = [
    MASTER_FILE,
    OUT_DIR / "global_acoustic_statistics.csv",
    OUT_DIR / "global_error_statistics.csv",
    OUT_DIR / "collapse_group_summary.csv",
    CORR_FILE,
    EFFECT_FILE,
    SEVERE_FILE,
    SPEAKER_FILE,
    HIGH_PITCH_FILE,
    LOW_SNR_FILE,
    LOW_C50_FILE,
    OUTLIER_FILE,
    REPORT_FILE,
    JSON_FILE,
]


for path in output_files:

    print(
        " ",
        path.name
    )


print()

print(
    "=" * 80
)

print(
    "FINAL ALIGNMENT STATUS"
)

print(
    "=" * 80
)

print(
    f"Validation manifest : {len(val)}"
)

print(
    f"Merged rows         : {len(merged)}"
)

print(
    f"IndicVoices rows    : {len(iv)}"
)

print(
    f"B3 CER rows         : "
    f"{merged['beam4g_cer'].notna().sum()}"
)

print(
    f"Blank rows          : "
    f"{merged['blank_blank_fraction'].notna().sum()}"
)

print()

print(
    "DO NOT RETRAIN B3 YET."
)

print(
    "Analyze the acoustic/error relationships first."
)

print(
    "=" * 80
)