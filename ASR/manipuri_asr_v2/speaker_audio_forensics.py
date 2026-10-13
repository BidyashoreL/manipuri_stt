"""
speaker_audio_forensics.py

B3 Speaker Audio Forensics
--------------------------

Target speaker:
    S4259682900336548

Analyzes all available train + validation audio for this speaker.

Outputs:
    checkpoints/v3_ctc/b3_narrow/diagnostics/speaker_forensics/
        all_61_audio_features.csv
        train_audio_summary.csv
        validation_audio_summary.csv
        severe_vs_normal.csv
        validation_audio/
            <index>_<class>.wav
        report.txt
"""

from pathlib import Path
import io
import json
import wave
import math
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"

B3_DIR = ASR / "checkpoints" / "v3_ctc" / "b3_narrow"

BLANK_CSV = (
    B3_DIR
    / "diagnostics"
    / "blank_attribution"
    / "merged_blank_attribution.csv"
)

OUTPUT_DIR = (
    B3_DIR
    / "diagnostics"
    / "speaker_forensics_audio"
)

WAV_DIR = OUTPUT_DIR / "validation_audio"

TARGET_SPEAKER = "S4259682900336548"

TARGET_SR = 16000


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 80)
print("B3 SPEAKER AUDIO FORENSICS")
print("=" * 80)

print(f"Script directory : {SCRIPT_DIR}")
print(f"ASR directory    : {ASR}")
print(f"Train CSV        : {TRAIN_CSV}")
print(f"Val CSV          : {VAL_CSV}")
print(f"Blank CSV        : {BLANK_CSV}")
print(f"Target speaker   : {TARGET_SPEAKER}")
print()


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    TRAIN_CSV,
    VAL_CSV,
    BLANK_CSV,
]

for p in required_files:
    if not p.exists():
        raise FileNotFoundError(f"Required file not found:\n{p}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
WAV_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
blank_df = pd.read_csv(BLANK_CSV)

print(f"Train rows       : {len(train_df)}")
print(f"Validation rows  : {len(val_df)}")
print(f"Blank rows       : {len(blank_df)}")
print()


# ============================================================
# NORMALIZE COLUMN NAMES
# ============================================================

for df in [train_df, val_df, blank_df]:

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]


# ============================================================
# FIND TARGET SPEAKER
# ============================================================

if "speaker_id" not in train_df.columns:
    raise RuntimeError(
        "speaker_id column not found in train.csv"
    )

if "speaker_id" not in val_df.columns:
    raise RuntimeError(
        "speaker_id column not found in val.csv"
    )

train_spk = train_df[
    train_df["speaker_id"].astype(str) == TARGET_SPEAKER
].copy()

val_spk = val_df[
    val_df["speaker_id"].astype(str) == TARGET_SPEAKER
].copy()

print("=" * 80)
print("TARGET SPEAKER")
print("=" * 80)

print(f"Training samples  : {len(train_spk)}")
print(f"Validation samples: {len(val_spk)}")
print()


if len(train_spk) == 0:
    raise RuntimeError(
        "Target speaker not found in training set."
    )

if len(val_spk) == 0:
    raise RuntimeError(
        "Target speaker not found in validation set."
    )


# ============================================================
# AUDIO DECODING
# ============================================================

def decode_wav_bytes(audio_bytes):
    """
    Decode WAV bytes using Python wave module.

    Returns:
        waveform float32 in [-1, 1]
        sample_rate
        channels
    """

    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:

        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

    if sample_width == 1:

        audio = np.frombuffer(
            raw,
            dtype=np.uint8
        ).astype(np.float32)

        audio = (audio - 128.0) / 128.0

    elif sample_width == 2:

        audio = np.frombuffer(
            raw,
            dtype=np.int16
        ).astype(np.float32)

        audio /= 32768.0

    elif sample_width == 4:

        audio = np.frombuffer(
            raw,
            dtype=np.int32
        ).astype(np.float32)

        audio /= 2147483648.0

    else:
        raise RuntimeError(
            f"Unsupported WAV sample width: {sample_width}"
        )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        )

        audio = audio.mean(axis=1)

    return audio.astype(np.float32), sample_rate, channels


def extract_audio_object(audio_obj):
    """
    Handle the common HuggingFace/IndicVoices parquet audio formats.

    Possible forms:

        bytes
        dict with {'bytes': ...}
        dict with {'path': ...}
        path string
    """

    # --------------------------------------------------------
    # Raw bytes
    # --------------------------------------------------------

    if isinstance(audio_obj, (bytes, bytearray)):

        return decode_wav_bytes(
            bytes(audio_obj)
        )


    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(audio_obj, dict):

        # HuggingFace Audio feature
        if "bytes" in audio_obj:

            if audio_obj["bytes"] is not None:

                return decode_wav_bytes(
                    audio_obj["bytes"]
                )

        if "path" in audio_obj:

            if audio_obj["path"] is not None:

                path = Path(
                    str(audio_obj["path"])
                )

                if path.exists():

                    return read_audio_file(
                        path
                    )


    # --------------------------------------------------------
    # String path
    # --------------------------------------------------------

    if isinstance(audio_obj, str):

        path = Path(audio_obj)

        if path.exists():

            return read_audio_file(path)


    raise RuntimeError(
        f"Unable to decode audio object of type "
        f"{type(audio_obj)}"
    )


def read_audio_file(path):
    """
    Read an external WAV file.
    """

    try:

        import soundfile as sf

        audio, sr = sf.read(
            str(path),
            always_2d=False
        )

        audio = np.asarray(
            audio,
            dtype=np.float32
        )

        if audio.ndim == 2:

            audio = audio.mean(axis=1)

        return audio, sr, 1

    except Exception:

        with open(path, "rb") as f:

            data = f.read()

        return decode_wav_bytes(data)


# ============================================================
# PARQUET CACHE
# ============================================================

PARQUET_CACHE = {}


def load_parquet_row(
    parquet_file,
    row_index
):
    """
    Load one row from an IndicVoices parquet file.
    """

    parquet_file = str(parquet_file)

    if parquet_file not in PARQUET_CACHE:

        print(
            f"Loading parquet: {parquet_file}"
        )

        try:

            import pyarrow.parquet as pq

            table = pq.read_table(
                parquet_file
            )

            df = table.to_pandas()

            PARQUET_CACHE[parquet_file] = df

        except Exception as e:

            raise RuntimeError(
                f"Failed to load parquet:\n"
                f"{parquet_file}\n"
                f"Error: {e}"
            )

    df = PARQUET_CACHE[parquet_file]

    row_index = int(row_index)

    if row_index < 0 or row_index >= len(df):

        raise IndexError(
            f"Parquet row {row_index} out of range "
            f"for {parquet_file} "
            f"(rows={len(df)})"
        )

    return df.iloc[row_index]


# ============================================================
# RESOLVE AUDIO
# ============================================================

def resolve_audio(row):

    source = str(
        row.get("source", "")
    ).lower()

    # --------------------------------------------------------
    # IndicVoices
    # --------------------------------------------------------

    if source == "indicvoices":

        parquet_file = row.get(
            "parquet_file",
            None
        )

        parquet_row = row.get(
            "parquet_row",
            None
        )

        if (
            pd.isna(parquet_file)
            or pd.isna(parquet_row)
        ):

            raise RuntimeError(
                "IndicVoices row missing "
                "parquet_file/parquet_row"
            )

        parquet_row_data = load_parquet_row(
            parquet_file,
            int(parquet_row)
        )

        if "audio" not in parquet_row_data.index:

            raise RuntimeError(
                "audio column missing from parquet"
            )

        audio_obj = parquet_row_data["audio"]

        return extract_audio_object(
            audio_obj
        )

    # --------------------------------------------------------
    # Other datasets
    # --------------------------------------------------------

    audio_path = row.get(
        "audio_path",
        None
    )

    if pd.isna(audio_path):

        audio_path = row.get(
            "audio_file",
            None
        )

    if pd.isna(audio_path):

        raise RuntimeError(
            "No audio_path/audio_file available"
        )

    return read_audio_file(
        Path(str(audio_path))
    )


# ============================================================
# AUDIO METRICS
# ============================================================

def calculate_audio_features(
    audio,
    sr
):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    n = len(audio)

    if n == 0:

        raise RuntimeError(
            "Audio contains zero samples"
        )

    duration = n / float(sr)

    abs_audio = np.abs(audio)

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = float(
        np.sqrt(
            np.mean(
                audio ** 2
            )
        )
    )

    rms_db = (
        20.0 * math.log10(
            max(rms, 1e-12)
        )
    )

    # --------------------------------------------------------
    # Peak
    # --------------------------------------------------------

    peak = float(
        np.max(abs_audio)
    )

    peak_db = (
        20.0 * math.log10(
            max(peak, 1e-12)
        )
    )

    # --------------------------------------------------------
    # Clipping
    # --------------------------------------------------------

    clipping_threshold = 0.999

    clipping_samples = int(
        np.sum(
            abs_audio >= clipping_threshold
        )
    )

    clipping_pct = (
        clipping_samples / n * 100.0
    )

    # --------------------------------------------------------
    # Exact zero samples
    # --------------------------------------------------------

    zero_samples = int(
        np.sum(
            abs_audio <= 1e-8
        )
    )

    zero_pct = (
        zero_samples / n * 100.0
    )

    # --------------------------------------------------------
    # Silence / activity
    #
    # Adaptive threshold based on RMS.
    # --------------------------------------------------------

    activity_threshold = max(
        rms * 0.10,
        1e-4
    )

    active_samples = int(
        np.sum(
            abs_audio > activity_threshold
        )
    )

    active_ratio = (
        active_samples / n
    )

    silence_ratio = (
        1.0 - active_ratio
    )

    # --------------------------------------------------------
    # Zero crossing rate
    # --------------------------------------------------------

    signs = np.signbit(audio)

    zc = np.sum(
        signs[1:] != signs[:-1]
    )

    zcr = (
        zc / max(n - 1, 1)
    )

    # --------------------------------------------------------
    # Dynamic range
    #
    # Compare RMS with peak.
    # --------------------------------------------------------

    dynamic_range_db = (
        peak_db - rms_db
    )

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    p01 = float(
        np.percentile(abs_audio, 1)
    )

    p50 = float(
        np.percentile(abs_audio, 50)
    )

    p95 = float(
        np.percentile(abs_audio, 95)
    )

    p99 = float(
        np.percentile(abs_audio, 99)
    )

    # --------------------------------------------------------
    # Energy variance
    # --------------------------------------------------------

    frame_size = max(
        int(sr * 0.025),
        1
    )

    hop = max(
        int(sr * 0.010),
        1
    )

    frame_rms = []

    for start in range(
        0,
        max(n - frame_size + 1, 1),
        hop
    ):

        frame = audio[
            start:start + frame_size
        ]

        if len(frame) == 0:
            continue

        frame_rms.append(
            np.sqrt(
                np.mean(
                    frame ** 2
                )
            )
        )

    frame_rms = np.asarray(
        frame_rms,
        dtype=np.float32
    )

    if len(frame_rms) > 0:

        frame_rms_mean = float(
            np.mean(frame_rms)
        )

        frame_rms_std = float(
            np.std(frame_rms)
        )

        frame_rms_min = float(
            np.min(frame_rms)
        )

        frame_rms_max = float(
            np.max(frame_rms)
        )

    else:

        frame_rms_mean = np.nan
        frame_rms_std = np.nan
        frame_rms_min = np.nan
        frame_rms_max = np.nan

    return {
        "sample_rate_actual": sr,
        "num_samples": n,
        "duration_audio": duration,

        "rms": rms,
        "rms_db": rms_db,

        "peak": peak,
        "peak_db": peak_db,

        "clipping_samples": clipping_samples,
        "clipping_pct": clipping_pct,

        "zero_samples": zero_samples,
        "zero_pct": zero_pct,

        "activity_threshold": activity_threshold,
        "active_ratio": active_ratio,
        "silence_ratio": silence_ratio,

        "zero_crossing_rate": zcr,

        "dynamic_range_db": dynamic_range_db,

        "abs_p01": p01,
        "abs_p50": p50,
        "abs_p95": p95,
        "abs_p99": p99,

        "frame_rms_mean": frame_rms_mean,
        "frame_rms_std": frame_rms_std,
        "frame_rms_min": frame_rms_min,
        "frame_rms_max": frame_rms_max,
    }


# ============================================================
# WRITE WAV
# ============================================================

def write_wav(
    path,
    audio,
    sr
):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    pcm = (
        audio * 32767.0
    ).astype(
        np.int16
    )

    with wave.open(
        str(path),
        "wb"
    ) as wf:

        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))

        wf.writeframes(
            pcm.tobytes()
        )


# ============================================================
# LOAD BLANK ATTRIBUTION
# ============================================================

blank_lookup = {}

for _, r in blank_df.iterrows():

    try:
        idx = int(r["index"])
        blank_lookup[idx] = r.to_dict()

    except Exception:
        pass


# ============================================================
# ANALYZE ONE ROW
# ============================================================

def analyze_row(
    row,
    split
):

    index = int(row.name)

    # IMPORTANT:
    # The dataframe index isn't necessarily the original
    # validation "index" used by blank attribution.
    #
    # Prefer the explicit audio/sample index if available.

    possible_index_columns = [
        "index",
        "sample_index",
        "row_index"
    ]

    sample_index = None

    for c in possible_index_columns:

        if c in row.index:

            try:
                sample_index = int(
                    row[c]
                )
                break

            except Exception:
                pass

    if sample_index is None:

        # For validation, use original dataframe
        # position as fallback.

        if split == "val":

            sample_index = int(
                val_df.index.get_loc(
                    row.name
                )
            )

        else:

            sample_index = int(
                train_df.index.get_loc(
                    row.name
                )
            )

    print(
        f"[{split.upper():5s}] "
        f"sample={sample_index:4d}"
    )

    result = {
        "split": split,
        "sample_index": sample_index,

        "speaker_id": str(
            row.get(
                "speaker_id",
                ""
            )
        ),

        "source": str(
            row.get(
                "source",
                ""
            )
        ),

        "lang": str(
            row.get(
                "lang",
                ""
            )
        ),

        "text": str(
            row.get(
                "text",
                ""
            )
        ),

        "normalized": str(
            row.get(
                "normalized",
                ""
            )
        ),

        "verbatim": str(
            row.get(
                "verbatim",
                ""
            )
        ),

        "parquet_file": str(
            row.get(
                "parquet_file",
                ""
            )
        ),

        "parquet_row": row.get(
            "parquet_row",
            np.nan
        ),
    }

    result["text_length"] = len(
        result["text"]
    )

    result["normalized_length"] = len(
        result["normalized"]
    )

    # --------------------------------------------------------
    # Blank attribution information
    # --------------------------------------------------------

    if sample_index in blank_lookup:

        b = blank_lookup[
            sample_index
        ]

        result[
            "blank_class"
        ] = b.get(
            "blank_class",
            b.get(
                "collapse_class",
                ""
            )
        )

        result[
            "T"
        ] = b.get(
            "T",
            np.nan
        )

        result[
            "reference_length"
        ] = b.get(
            "reference_length",
            np.nan
        )

        result[
            "predicted_length"
        ] = b.get(
            "predicted_length",
            np.nan
        )

        result[
            "pred_ref_length_ratio"
        ] = b.get(
            "pred_ref_length_ratio",
            np.nan
        )

        result[
            "blank_fraction"
        ] = b.get(
            "blank_fraction",
            np.nan
        )

        result[
            "mean_blank_probability"
        ] = b.get(
            "mean_blank_probability",
            np.nan
        )

    else:

        result[
            "blank_class"
        ] = ""

    # --------------------------------------------------------
    # Load actual audio
    # --------------------------------------------------------

    try:

        audio, sr, channels = resolve_audio(
            row
        )

        result["channels"] = channels

        features = calculate_audio_features(
            audio,
            sr
        )

        result.update(
            features
        )

        result["audio_status"] = "OK"

    except Exception as e:

        print(
            f"    ERROR: {e}"
        )

        result[
            "audio_status"
        ] = f"ERROR: {e}"

        return result

    # --------------------------------------------------------
    # Save validation WAV
    # --------------------------------------------------------

    if split == "val":

        cls = str(
            result.get(
                "blank_class",
                ""
            )
        ).upper()

        if cls == "":
            cls = "UNKNOWN"

        wav_name = (
            f"{sample_index:04d}_"
            f"{cls}.wav"
        )

        wav_path = (
            WAV_DIR / wav_name
        )

        try:

            write_wav(
                wav_path,
                audio,
                sr
            )

            result[
                "exported_wav"
            ] = str(
                wav_path
            )

        except Exception as e:

            result[
                "exported_wav"
            ] = (
                f"ERROR: {e}"
            )

    return result


# ============================================================
# RUN ANALYSIS
# ============================================================

print("=" * 80)
print("ANALYZING TRAINING AUDIO")
print("=" * 80)

results = []

for _, row in train_spk.iterrows():

    results.append(
        analyze_row(
            row,
            "train"
        )
    )


print()
print("=" * 80)
print("ANALYZING VALIDATION AUDIO")
print("=" * 80)

for _, row in val_spk.iterrows():

    results.append(
        analyze_row(
            row,
            "val"
        )
    )


# ============================================================
# DATAFRAME
# ============================================================

features_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE ALL FEATURES
# ============================================================

all_features_path = (
    OUTPUT_DIR
    / "all_61_audio_features.csv"
)

features_df.to_csv(
    all_features_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# TRAIN SUMMARY
# ============================================================

train_features = features_df[
    features_df["split"] == "train"
].copy()

val_features = features_df[
    features_df["split"] == "val"
].copy()


numeric_columns = [
    "duration_audio",
    "rms",
    "rms_db",
    "peak",
    "peak_db",
    "clipping_pct",
    "zero_pct",
    "active_ratio",
    "silence_ratio",
    "zero_crossing_rate",
    "dynamic_range_db",
    "abs_p01",
    "abs_p50",
    "abs_p95",
    "abs_p99",
    "frame_rms_mean",
    "frame_rms_std",
    "frame_rms_min",
    "frame_rms_max",
    "text_length",
    "normalized_length",
]


def make_summary(df):

    rows = []

    for col in numeric_columns:

        if col not in df.columns:
            continue

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        rows.append({
            "feature": col,
            "count": len(values),
            "mean": values.mean(),
            "median": values.median(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "p25": values.quantile(0.25),
            "p75": values.quantile(0.75),
        })

    return pd.DataFrame(rows)


train_summary = make_summary(
    train_features
)

val_summary = make_summary(
    val_features
)


train_summary.to_csv(
    OUTPUT_DIR / "train_audio_summary.csv",
    index=False
)

val_summary.to_csv(
    OUTPUT_DIR / "validation_audio_summary.csv",
    index=False
)


# ============================================================
# SEVERE VS NORMAL
# ============================================================

comparison_classes = [
    "SEVERE",
    "MODERATE",
    "NORMAL",
]

comparison_rows = []

for cls in comparison_classes:

    subset = val_features[
        val_features[
            "blank_class"
        ].astype(str).str.upper()
        == cls
    ]

    if len(subset) == 0:
        continue

    for col in numeric_columns:

        if col not in subset.columns:
            continue

        vals = pd.to_numeric(
            subset[col],
            errors="coerce"
        )

        comparison_rows.append({
            "class": cls,
            "n": len(subset),
            "feature": col,
            "mean": vals.mean(),
            "median": vals.median(),
            "std": vals.std(),
            "min": vals.min(),
            "max": vals.max(),
        })


comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    OUTPUT_DIR / "severe_vs_normal.csv",
    index=False
)


# ============================================================
# PRINT VALIDATION TABLE
# ============================================================

print()
print("=" * 80)
print("VALIDATION AUDIO RESULTS")
print("=" * 80)

display_columns = [
    "sample_index",
    "blank_class",
    "duration_audio",
    "text_length",
    "rms_db",
    "peak_db",
    "clipping_pct",
    "silence_ratio",
    "active_ratio",
    "zero_crossing_rate",
    "dynamic_range_db",
    "blank_fraction",
    "pred_ref_length_ratio",
]

available_display = [
    c for c in display_columns
    if c in val_features.columns
]

print(
    val_features[
        available_display
    ].to_string(
        index=False
    )
)


# ============================================================
# CLASS SUMMARY
# ============================================================

print()
print("=" * 80)
print("VALIDATION COLLAPSE CLASS SUMMARY")
print("=" * 80)

if "blank_class" in val_features.columns:

    class_counts = (
        val_features[
            "blank_class"
        ]
        .astype(str)
        .str.upper()
        .value_counts()
    )

    print(
        class_counts.to_string()
    )


# ============================================================
# TRAIN vs VALIDATION COMPARISON
# ============================================================

print()
print("=" * 80)
print("TRAIN vs VALIDATION AUDIO COMPARISON")
print("=" * 80)

comparison_features = [
    "duration_audio",
    "rms_db",
    "peak_db",
    "clipping_pct",
    "zero_pct",
    "active_ratio",
    "silence_ratio",
    "zero_crossing_rate",
    "dynamic_range_db",
    "text_length",
]


comparison_output = []

for col in comparison_features:

    if col not in train_features.columns:
        continue

    train_vals = pd.to_numeric(
        train_features[col],
        errors="coerce"
    ).dropna()

    val_vals = pd.to_numeric(
        val_features[col],
        errors="coerce"
    ).dropna()

    if len(train_vals) == 0:
        continue

    if len(val_vals) == 0:
        continue

    train_mean = train_vals.mean()
    val_mean = val_vals.mean()

    difference = (
        val_mean - train_mean
    )

    if abs(train_mean) > 1e-12:

        pct_difference = (
            difference
            / abs(train_mean)
            * 100.0
        )

    else:

        pct_difference = np.nan

    comparison_output.append({
        "feature": col,
        "train_mean": train_mean,
        "val_mean": val_mean,
        "difference": difference,
        "percent_difference": pct_difference,
    })


comparison_summary = pd.DataFrame(
    comparison_output
)

print(
    comparison_summary.to_string(
        index=False
    )
)


comparison_summary.to_csv(
    OUTPUT_DIR
    / "train_vs_validation_comparison.csv",
    index=False
)


# ============================================================
# OUTLIERS
# ============================================================

print()
print("=" * 80)
print("VALIDATION AUDIO OUTLIERS")
print("=" * 80)


def print_extreme(
    df,
    column,
    ascending=False,
    n=10
):

    if column not in df.columns:
        return

    tmp = df.copy()

    tmp[column] = pd.to_numeric(
        tmp[column],
        errors="coerce"
    )

    tmp = tmp.dropna(
        subset=[column]
    )

    if len(tmp) == 0:
        return

    print()
    print(
        f"--- {column} ---"
    )

    print(
        tmp.sort_values(
            column,
            ascending=ascending
        )[
            [
                "sample_index",
                "blank_class",
                column,
                "duration_audio",
                "text_length",
            ]
        ]
        .head(n)
        .to_string(
            index=False
        )
    )


print_extreme(
    val_features,
    "rms_db",
    ascending=True
)

print_extreme(
    val_features,
    "silence_ratio",
    ascending=False
)

print_extreme(
    val_features,
    "clipping_pct",
    ascending=False
)

print_extreme(
    val_features,
    "zero_crossing_rate",
    ascending=False
)

print_extreme(
    val_features,
    "dynamic_range_db",
    ascending=False
)


# ============================================================
# REPORT
# ============================================================

report_lines = []

report_lines.append(
    "B3 SPEAKER AUDIO FORENSICS REPORT"
)

report_lines.append(
    "=" * 70
)

report_lines.append(
    f"Target speaker: {TARGET_SPEAKER}"
)

report_lines.append(
    f"Training samples: {len(train_features)}"
)

report_lines.append(
    f"Validation samples: {len(val_features)}"
)

report_lines.append("")


# ------------------------------------------------------------
# Collapse distribution
# ------------------------------------------------------------

report_lines.append(
    "VALIDATION COLLAPSE DISTRIBUTION"
)

if "blank_class" in val_features.columns:

    counts = (
        val_features[
            "blank_class"
        ]
        .astype(str)
        .str.upper()
        .value_counts()
    )

    for cls, count in counts.items():

        report_lines.append(
            f"{cls}: {count}"
        )

report_lines.append("")


# ------------------------------------------------------------
# Severe samples
# ------------------------------------------------------------

report_lines.append(
    "SEVERE VALIDATION SAMPLES"
)

severe = val_features[
    val_features[
        "blank_class"
    ].astype(str).str.upper()
    == "SEVERE"
]

for _, r in severe.iterrows():

    report_lines.append(
        f"sample={r['sample_index']} | "
        f"duration={r.get('duration_audio', np.nan):.3f}s | "
        f"text_len={r.get('text_length', np.nan)} | "
        f"RMS_dB={r.get('rms_db', np.nan):.3f} | "
        f"peak_dB={r.get('peak_db', np.nan):.3f} | "
        f"silence={r.get('silence_ratio', np.nan):.4f} | "
        f"active={r.get('active_ratio', np.nan):.4f} | "
        f"clip={r.get('clipping_pct', np.nan):.4f}% | "
        f"blank={r.get('blank_fraction', np.nan):.4f} | "
        f"pred/ref={r.get('pred_ref_length_ratio', np.nan):.4f}"
    )

report_lines.append("")


# ------------------------------------------------------------
# Normal samples
# ------------------------------------------------------------

report_lines.append(
    "NORMAL VALIDATION SAMPLES"
)

normal = val_features[
    val_features[
        "blank_class"
    ].astype(str).str.upper()
    == "NORMAL"
]

for _, r in normal.iterrows():

    report_lines.append(
        f"sample={r['sample_index']} | "
        f"duration={r.get('duration_audio', np.nan):.3f}s | "
        f"text_len={r.get('text_length', np.nan)} | "
        f"RMS_dB={r.get('rms_db', np.nan):.3f} | "
        f"peak_dB={r.get('peak_db', np.nan):.3f} | "
        f"silence={r.get('silence_ratio', np.nan):.4f} | "
        f"active={r.get('active_ratio', np.nan):.4f} | "
        f"clip={r.get('clipping_pct', np.nan):.4f}% | "
        f"blank={r.get('blank_fraction', np.nan):.4f} | "
        f"pred/ref={r.get('pred_ref_length_ratio', np.nan):.4f}"
    )

report_lines.append("")


# ------------------------------------------------------------
# Important interpretation
# ------------------------------------------------------------

report_lines.append(
    "IMPORTANT:"
)

report_lines.append(
    "This script reports acoustic differences only."
)

report_lines.append(
    "It does NOT establish causality."
)

report_lines.append(
    "Severe samples should be manually listened to "
    "before concluding that audio quality caused the "
    "CTC collapse."
)


report_path = (
    OUTPUT_DIR / "report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            report_lines
        )
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    f"Output directory:\n{OUTPUT_DIR}"
)

print()
print("Generated:")

for p in sorted(
    OUTPUT_DIR.rglob("*")
):

    if p.is_file():

        print(
            f"  {p.relative_to(OUTPUT_DIR)}"
        )

print()
print(
    "IMPORTANT: listen to the six WAV files in:"
)

print(
    WAV_DIR
)

print()
print(
    "Do NOT retrain B3 yet."
)

print("=" * 80)