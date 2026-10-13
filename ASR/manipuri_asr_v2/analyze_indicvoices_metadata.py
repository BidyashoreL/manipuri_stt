from pathlib import Path
import json
import math
import pandas as pd
import numpy as np
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"

OUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
    / "indicvoices_metadata"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TARGET SPEAKER / COLLAPSE CASES
# ============================================================

TARGET_SPEAKER = "S4259682900336548"

# Known validation behavior from our previous blank-forensics analysis.
COLLAPSE_LABELS = {
    144: "NORMAL",
    187: "SEVERE",
    491: "SEVERE",
    568: "SEVERE",
    702: "MODERATE",
    814: "SEVERE",
}


# ============================================================
# METADATA FIELDS TO INVESTIGATE
# ============================================================

METADATA_FIELDS = [
    "utterance_pitch_mean",
    "utterance_pitch_std",
    "snr",
    "c50",
    "speaking_rate",
    "duration",
    "scenario",
    "task_name",
    "gender",
    "age_group",
    "job_type",
    "qualification",
    "area",
    "district",
    "state",
    "occupation",
]


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    """
    Convert Arrow/Python values into something safely serializable.
    """
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, (list, tuple)):
        return [clean_value(x) for x in value]

    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}

    return value


def safe_mean(series):
    x = pd.to_numeric(series, errors="coerce")
    return float(x.mean()) if x.notna().any() else np.nan


def safe_median(series):
    x = pd.to_numeric(series, errors="coerce")
    return float(x.median()) if x.notna().any() else np.nan


def safe_std(series):
    x = pd.to_numeric(series, errors="coerce")
    return float(x.std()) if x.notna().any() else np.nan


# ============================================================
# LOAD MANIFESTS
# ============================================================

print("=" * 80)
print("INDICVOICES METADATA FORENSICS")
print("=" * 80)

print(f"ASR root : {ASR}")
print(f"Train CSV: {TRAIN_CSV}")
print(f"Val CSV  : {VAL_CSV}")
print()

train = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)

print(f"Train rows: {len(train)}")
print(f"Val rows  : {len(val)}")
print()


# ============================================================
# FILTER INDICVOICES
# ============================================================

train_iv = train[
    train["source"].astype(str).str.lower() == "indicvoices"
].copy()

val_iv = val[
    val["source"].astype(str).str.lower() == "indicvoices"
].copy()

print(f"IndicVoices train: {len(train_iv)}")
print(f"IndicVoices val  : {len(val_iv)}")
print()


# ============================================================
# FIND REQUIRED PARQUET FILES
# ============================================================

all_rows = pd.concat(
    [
        train_iv.assign(split="train"),
        val_iv.assign(split="val"),
    ],
    ignore_index=True,
)

parquet_files = sorted(
    set(
        str(x)
        for x in all_rows["parquet_file"].dropna()
    )
)

print(f"Unique parquet files needed: {len(parquet_files)}")
print()


# ============================================================
# READ ORIGINAL PARQUET METADATA
# ============================================================

records = []

for parquet_path in parquet_files:

    print(f"Reading: {parquet_path}")

    if not Path(parquet_path).exists():
        print("  WARNING: parquet file does not exist")
        continue

    needed = all_rows[
        all_rows["parquet_file"].astype(str) == parquet_path
    ]

    table = pq.read_table(parquet_path)

    available_columns = set(table.column_names)

    # Read the entire requested parquet rows.
    # This is safe because only the metadata columns we need
    # are extracted below.
    df_parquet = table.to_pandas()

    for _, row in needed.iterrows():

        parquet_row = int(row["parquet_row"])

        if parquet_row < 0 or parquet_row >= len(df_parquet):
            print(
                f"  WARNING: invalid parquet row "
                f"{parquet_row} in {parquet_path}"
            )
            continue

        p = df_parquet.iloc[parquet_row]

        record = {
            "manifest_index": int(row.name),
            "split": row["split"],
            "source": row["source"],
            "speaker_id": row.get("speaker_id"),
            "audio_file": row.get("audio_file"),
            "parquet_file": parquet_path,
            "parquet_row": parquet_row,
            "text": row.get("text"),
            "manifest_duration": row.get("duration"),
        }

        for field in METADATA_FIELDS:
            if field in available_columns:
                record[field] = clean_value(p[field])
            else:
                record[field] = None

        records.append(record)


metadata = pd.DataFrame(records)

print()
print(f"Extracted metadata rows: {len(metadata)}")
print()


# ============================================================
# SAVE RAW EXTRACT
# ============================================================

raw_path = OUT_DIR / "all_indicvoices_metadata.csv"
metadata.to_csv(raw_path, index=False)

print(f"Saved: {raw_path}")
print()


# ============================================================
# TARGET SPEAKER ANALYSIS
# ============================================================

target = metadata[
    metadata["speaker_id"].astype(str) == TARGET_SPEAKER
].copy()

print("=" * 80)
print("TARGET SPEAKER")
print("=" * 80)

print(f"Speaker: {TARGET_SPEAKER}")
print(f"Total samples found: {len(target)}")
print()

print(
    target[
        [
            "split",
            "parquet_row",
            "duration",
            "snr",
            "c50",
            "speaking_rate",
            "utterance_pitch_mean",
            "utterance_pitch_std",
        ]
    ].to_string(index=False)
)

print()


# ============================================================
# TARGET VALIDATION SAMPLES
# ============================================================

target_val = target[target["split"] == "val"].copy()

# Match our known validation sample indices using the
# original validation dataframe position.
val_iv_reset = val_iv.reset_index(drop=True)

target_val_rows = []

for i, row in val_iv_reset.iterrows():

    if str(row.get("speaker_id")) != TARGET_SPEAKER:
        continue

    label = COLLAPSE_LABELS.get(i, "UNKNOWN")

    target_val_rows.append(
        {
            "val_index": i,
            "speaker_id": TARGET_SPEAKER,
            "collapse_label": label,
            "text": row.get("text"),
            "parquet_file": row.get("parquet_file"),
            "parquet_row": row.get("parquet_row"),
        }
    )

target_val_map = pd.DataFrame(target_val_rows)

if len(target_val_map):

    target_val_map = target_val_map.merge(
        target,
        on=["speaker_id", "parquet_file", "parquet_row"],
        how="left",
        suffixes=("", "_metadata"),
    )

    target_val_path = OUT_DIR / "target_speaker_val_metadata.csv"
    target_val_map.to_csv(target_val_path, index=False)

    print("=" * 80)
    print("TARGET VALIDATION METADATA")
    print("=" * 80)

    cols = [
        "val_index",
        "collapse_label",
        "duration",
        "snr",
        "c50",
        "speaking_rate",
        "utterance_pitch_mean",
        "utterance_pitch_std",
        "parquet_row",
    ]

    print(target_val_map[cols].to_string(index=False))
    print()

    print(f"Saved: {target_val_path}")
    print()


# ============================================================
# TRAIN VS VALIDATION TARGET SPEAKER
# ============================================================

print("=" * 80)
print("TARGET SPEAKER: TRAIN VS VALIDATION")
print("=" * 80)

summary_rows = []

for split in ["train", "val"]:

    subset = target[target["split"] == split]

    row = {
        "split": split,
        "samples": len(subset),
    }

    for field in [
        "duration",
        "snr",
        "c50",
        "speaking_rate",
        "utterance_pitch_mean",
        "utterance_pitch_std",
    ]:

        row[f"{field}_mean"] = safe_mean(subset[field])
        row[f"{field}_median"] = safe_median(subset[field])
        row[f"{field}_std"] = safe_std(subset[field])

    summary_rows.append(row)

target_summary = pd.DataFrame(summary_rows)

print(target_summary.to_string(index=False))
print()

target_summary.to_csv(
    OUT_DIR / "target_speaker_train_vs_val.csv",
    index=False,
)


# ============================================================
# ALL INDICVOICES VALIDATION SUMMARY
# ============================================================

print("=" * 80)
print("ALL INDICVOICES VALIDATION METADATA")
print("=" * 80)

global_summary = []

for field in [
    "duration",
    "snr",
    "c50",
    "speaking_rate",
    "utterance_pitch_mean",
    "utterance_pitch_std",
]:

    x = pd.to_numeric(
            metadata.loc[metadata["split"] == "val", field],
            errors="coerce"
    )

    global_summary.append(
        {
            "field": field,
            "count": int(x.notna().sum()),
            "mean": float(x.mean()) if x.notna().any() else np.nan,
            "median": float(x.median()) if x.notna().any() else np.nan,
            "std": float(x.std()) if x.notna().any() else np.nan,
            "min": float(x.min()) if x.notna().any() else np.nan,
            "max": float(x.max()) if x.notna().any() else np.nan,
        }
    )

global_summary = pd.DataFrame(global_summary)

print(global_summary.to_string(index=False))
print()

global_summary.to_csv(
    OUT_DIR / "indicvoices_val_metadata_summary.csv",
    index=False,
)


# ============================================================
# TARGET VAL Z-SCORES AGAINST ALL INDICVOICES VAL
# ============================================================

print("=" * 80)
print("TARGET VALIDATION SAMPLES VS GLOBAL INDICVOICES")
print("=" * 80)

z_rows = []

for _, row in target_val_map.iterrows():

    result = {
        "val_index": row["val_index"],
        "collapse_label": row["collapse_label"],
    }

    for field in [
        "duration",
        "snr",
        "c50",
        "speaking_rate",
        "utterance_pitch_mean",
        "utterance_pitch_std",
    ]:

        target_value = pd.to_numeric(
            pd.Series([row[field]]),
            errors="coerce",
        ).iloc[0]

        global_values = pd.to_numeric(
            metadata.loc[metadata["split"] == "val", field],
            errors="coerce",
        ).dropna()

        if (
            pd.isna(target_value)
            or len(global_values) < 2
            or global_values.std() == 0
        ):
            result[f"{field}_z"] = np.nan
        else:
            result[f"{field}_z"] = (
                target_value - global_values.mean()
            ) / global_values.std()

    z_rows.append(result)

z_df = pd.DataFrame(z_rows)

print(z_df.to_string(index=False))
print()

z_df.to_csv(
    OUT_DIR / "target_validation_global_zscores.csv",
    index=False,
)


# ============================================================
# CORRELATION CHECK
# ============================================================

print("=" * 80)
print("METADATA CORRELATION WITH B3 ERROR PROXIES")
print("=" * 80)

# We do not have per-sample CER here, so use the six known
# collapse cases only for descriptive inspection.
#
# This is intentionally NOT treated as a statistical test.

if len(target_val_map):

    numeric_fields = [
        "duration",
        "snr",
        "c50",
        "speaking_rate",
        "utterance_pitch_mean",
        "utterance_pitch_std",
    ]

    for field in numeric_fields:

        values = pd.to_numeric(
            target_val_map[field],
            errors="coerce",
        )

        print(
            f"{field:25s} "
            f"mean={values.mean():.4f} "
            f"median={values.median():.4f} "
            f"min={values.min():.4f} "
            f"max={values.max():.4f}"
        )


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

summary = {
    "target_speaker": TARGET_SPEAKER,
    "total_metadata_rows": len(metadata),
    "target_speaker_total": len(target),
    "target_speaker_train": int(
        (target["split"] == "train").sum()
    ),
    "target_speaker_val": int(
        (target["split"] == "val").sum()
    ),
    "known_validation_collapse_labels": COLLAPSE_LABELS,
}

with open(
    OUT_DIR / "metadata_forensics_summary.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)


print()
print("=" * 80)
print("DONE")
print("=" * 80)
print(f"Output directory:")
print(OUT_DIR)
print()