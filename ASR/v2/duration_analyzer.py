from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================================
# PROJECT PATHS
# ============================================================================

# This file:
#
#   <project>\manipuri_stt\ASR\v2\duration_analyzer.py
#
# Therefore:
#
#   parent        = v2
#   parent.parent = ASR
#
ASR_DIR = Path(__file__).resolve().parent.parent

# Actual project root containing:
#
#   ASR/
#   benchmark/
#   ...
#
PROJECT_DIR = ASR_DIR.parent

OUTPUT_DIR = Path(__file__).resolve().parent / "logs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATASET FILES
# ============================================================================

TRAIN_CSV = ASR_DIR / "train.csv"
VAL_CSV = ASR_DIR / "val.csv"
TEST_CSV = ASR_DIR / "test.csv"

OUTPUT_JSON = OUTPUT_DIR / "duration_analysis.json"


# ============================================================================
# AUDIO / FEATURE CONFIGURATION
# ============================================================================

# Approximate Wav2Vec2-BERT feature rate.
FEATURE_RATE = 50.0

# Audio Projector uses stride=2.
#
# Therefore:
#
#   50 encoder frames/sec / 2
#   = 25 projected tokens/sec
#
PROJECTED_RATE = FEATURE_RATE / 2.0


# ============================================================================
# COLUMN DETECTION
# ============================================================================

def find_audio_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find the audio path column.

    Supports several common naming conventions.
    """

    candidates = [
        "audio_path",
        "audio",
        "path",
        "file",
        "file_path",
        "wav",
        "wav_path",
        "audio_file",
    ]

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in lower_map:
            return str(lower_map[candidate])

    return None


def find_duration_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find an existing duration column if the CSV already provides one.
    """

    candidates = [
        "duration",
        "duration_sec",
        "duration_seconds",
        "audio_duration",
        "audio_seconds",
        "seconds",
    ]

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        if candidate in lower_map:
            return str(lower_map[candidate])

    return None


# ============================================================================
# PATH HELPERS
# ============================================================================

def _is_valid_file(path: Path) -> bool:
    """
    Return True only if path exists and is a regular file.
    """

    try:
        return path.exists() and path.is_file()
    except Exception:
        return False


def _candidate_from_marker(
    normalized_path: str,
    marker: str,
) -> Optional[Path]:
    """
    Rebuild a path using PROJECT_DIR.

    Example:

        old:
        C:\\Users\\OldUser\\...\\manipuri_stt\\benchmark\\sample_053.wav

        marker:
        \\manipuri_stt\\

        result:

        <current project>\\benchmark\\sample_053.wav
    """

    lower_path = normalized_path.lower()
    lower_marker = marker.lower()

    marker_index = lower_path.find(
        lower_marker
    )

    if marker_index < 0:
        return None

    relative_part = normalized_path[
        marker_index + len(marker):
    ]

    relative_part = relative_part.lstrip(
        "\\/"
    )

    candidate = PROJECT_DIR / Path(
        relative_part
    )

    if _is_valid_file(candidate):
        return candidate

    return None


# ============================================================================
# AUDIO PATH RESOLUTION
# ============================================================================

def resolve_audio_path(
    raw_path,
) -> Optional[Path]:
    """
    Resolve an audio path without modifying the CSV.

    Resolution order:

        1. Exact path
        2. Relative to ASR directory
        3. Relative to project directory
        4. Legacy \\manipuri_stt\\ path
        5. Known dataset directories
        6. Filename fallback

    This handles old absolute Windows paths generated on another
    machine/user account.
    """

    if raw_path is None:
        return None

    raw = str(raw_path).strip().strip('"')

    if not raw:
        return None

    # ------------------------------------------------------------------------
    # Normalize Windows separators.
    # ------------------------------------------------------------------------

    normalized = raw.replace(
        "/",
        "\\"
    )

    # ------------------------------------------------------------------------
    # Case 1:
    #
    # Path exists exactly as recorded.
    # ------------------------------------------------------------------------

    direct = Path(raw)

    if _is_valid_file(direct):
        return direct.resolve()

    # ------------------------------------------------------------------------
    # Case 2:
    #
    # Relative to ASR directory.
    #
    # Example:
    #
    #   audio/sample.wav
    #
    # becomes:
    #
    #   <project>\ASR\audio\sample.wav
    # ------------------------------------------------------------------------

    if not Path(raw).is_absolute():

        candidate = ASR_DIR / Path(raw)

        if _is_valid_file(candidate):
            return candidate.resolve()

        # --------------------------------------------------------------------
        # Case 3:
        #
        # Relative to project directory.
        # --------------------------------------------------------------------

        candidate = PROJECT_DIR / Path(raw)

        if _is_valid_file(candidate):
            return candidate.resolve()

    # ------------------------------------------------------------------------
    # Case 4:
    #
    # Legacy absolute path containing:
    #
    #   \manipuri_stt\
    #
    # This is the important fix for:
    #
    #   ...\manipuri_stt\benchmark\sample_053.wav
    #
    # and:
    #
    #   ...\manipuri_stt\ASR\...
    #
    # ------------------------------------------------------------------------

    candidate = _candidate_from_marker(
        normalized,
        "\\manipuri_stt\\",
    )

    if candidate is not None:
        return candidate.resolve()

    # ------------------------------------------------------------------------
    # Case 5:
    #
    # Explicitly recognize known dataset directories.
    #
    # This is useful if a path was generated without the expected
    # project-root marker.
    # ------------------------------------------------------------------------

    known_dirs = [
        "benchmark",
        "manipuri_benchmark_extracted",
        "manipuri_1k_project",
    ]

    lower_normalized = normalized.lower()

    for known_dir in known_dirs:

        marker = (
            "\\"
            + known_dir
            + "\\"
        )

        index = lower_normalized.find(
            marker.lower()
        )

        if index < 0:
            continue

        relative_part = normalized[
            index + 1:
        ]

        candidate = PROJECT_DIR / Path(
            relative_part
        )

        if _is_valid_file(candidate):
            return candidate.resolve()

    # ------------------------------------------------------------------------
    # Case 6:
    #
    # Filename fallback.
    #
    # Only accept the result when there is exactly one match.
    #
    # This prevents accidentally selecting the wrong file if duplicate
    # filenames exist in multiple directories.
    # ------------------------------------------------------------------------

    filename = Path(normalized).name

    if filename:

        try:
            matches = list(
                PROJECT_DIR.rglob(filename)
            )

            valid_matches = [
                path
                for path in matches
                if _is_valid_file(path)
            ]

            if len(valid_matches) == 1:
                return valid_matches[0].resolve()

        except Exception:
            pass

    return None


# ============================================================================
# AUDIO DURATION
# ============================================================================

def get_duration(
    audio_path: Path,
) -> Optional[float]:
    """
    Read duration directly from the audio file.
    """

    try:

        import soundfile as sf

        info = sf.info(
            str(audio_path)
        )

        if info.samplerate <= 0:
            return None

        duration = (
            float(info.frames)
            / float(info.samplerate)
        )

        if duration <= 0:
            return None

        return duration

    except Exception:
        return None


# ============================================================================
# DATASET ANALYSIS
# ============================================================================

def analyze_dataset(
    csv_path: Path,
) -> dict:

    print()
    print("=" * 78)
    print(f"FILE: {csv_path.name}")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # Verify CSV
    # ------------------------------------------------------------------------

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Dataset CSV not found: {csv_path}"
        )

    # ------------------------------------------------------------------------
    # Load CSV
    # ------------------------------------------------------------------------

    df = pd.read_csv(
        csv_path
    )

    print(
        f"Rows: {len(df):,}"
    )

    print("Columns:")

    for column in df.columns:
        print(
            f"  - {column}"
        )

    # ------------------------------------------------------------------------
    # Detect columns
    # ------------------------------------------------------------------------

    duration_col = find_duration_column(
        df
    )

    audio_col = find_audio_column(
        df
    )

    durations = []

    resolved_paths = []

    missing_paths = []

    # ------------------------------------------------------------------------
    # Existing duration column
    # ------------------------------------------------------------------------

    if duration_col:

        print(
            f"Duration column detected: "
            f"{duration_col}"
        )

        raw_durations = pd.to_numeric(
            df[duration_col],
            errors="coerce",
        )

        durations = raw_durations.tolist()

    # ------------------------------------------------------------------------
    # Audio path column
    # ------------------------------------------------------------------------

    elif audio_col:

        print(
            f"Audio column detected: "
            f"{audio_col}"
        )

        print(
            "Resolving audio paths against "
            "CURRENT project..."
        )

        print()

        total_rows = len(df)

        for i, raw_path in enumerate(
            df[audio_col]
        ):

            if (
                i % 500 == 0
                or i == total_rows - 1
            ):

                print(
                    f"  Processing: "
                    f"{i + 1:,}/"
                    f"{total_rows:,}"
                )

            resolved = resolve_audio_path(
                raw_path
            )

            if resolved is None:

                durations.append(
                    None
                )

                missing_paths.append(
                    {
                        "row": int(i),
                        "original_path": str(
                            raw_path
                        ),
                    }
                )

                continue

            duration = get_duration(
                resolved
            )

            if duration is None:

                durations.append(
                    None
                )

                missing_paths.append(
                    {
                        "row": int(i),
                        "original_path": str(
                            raw_path
                        ),
                        "resolved_path": str(
                            resolved
                        ),
                        "error": (
                            "Could not read "
                            "audio duration"
                        ),
                    }
                )

                continue

            durations.append(
                duration
            )

            resolved_paths.append(
                str(resolved)
            )

    else:

        raise RuntimeError(
            "No audio_path or duration "
            f"column found in {csv_path}"
        )

    # ------------------------------------------------------------------------
    # Valid durations
    # ------------------------------------------------------------------------

    durations_series = pd.Series(
        durations,
        dtype="float64",
    )

    valid = durations_series.dropna()

    valid = valid[
        valid > 0
    ]

    if len(valid) == 0:

        print()
        print(
            "ERROR: No valid audio "
            "durations found."
        )

        if (
            audio_col
            and len(df) > 0
        ):

            print()
            print(
                "Example original path:"
            )

            print(
                df[audio_col].iloc[0]
            )

        raise RuntimeError(
            "No valid durations found "
            f"in {csv_path.name}."
        )

    # ------------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------------

    stats = {

        "rows": int(
            len(df)
        ),

        "valid_duration_rows": int(
            len(valid)
        ),

        "missing_duration_rows": int(
            len(df) - len(valid)
        ),

        "min_seconds": float(
            valid.min()
        ),

        "max_seconds": float(
            valid.max()
        ),

        "mean_seconds": float(
            valid.mean()
        ),

        "median_seconds": float(
            valid.median()
        ),

        "p50_seconds": float(
            valid.quantile(0.50)
        ),

        "p75_seconds": float(
            valid.quantile(0.75)
        ),

        "p90_seconds": float(
            valid.quantile(0.90)
        ),

        "p95_seconds": float(
            valid.quantile(0.95)
        ),

        "p99_seconds": float(
            valid.quantile(0.99)
        ),
    }

    # ------------------------------------------------------------------------
    # Duration buckets
    # ------------------------------------------------------------------------

    buckets = [

        ("0-3 sec", 0, 3),

        ("3-5 sec", 3, 5),

        ("5-8 sec", 5, 8),

        ("8-12 sec", 8, 12),

        ("12-15 sec", 12, 15),

        (">15 sec", 15, float("inf")),
    ]

    bucket_stats = {}

    for name, low, high in buckets:

        mask = (
            (valid >= low)
            & (valid < high)
        )

        count = int(
            mask.sum()
        )

        bucket_stats[name] = {

            "count": count,

            "percentage": float(
                count
                / len(valid)
                * 100
            ),
        }

    # ------------------------------------------------------------------------
    # Projected sequence lengths
    # ------------------------------------------------------------------------

    projected = (
        valid
        * PROJECTED_RATE
    )

    projected_stats = {

        "estimated_min_tokens": float(
            projected.min()
        ),

        "estimated_max_tokens": float(
            projected.max()
        ),

        "estimated_mean_tokens": float(
            projected.mean()
        ),

        "estimated_p50_tokens": float(
            projected.quantile(0.50)
        ),

        "estimated_p90_tokens": float(
            projected.quantile(0.90)
        ),

        "estimated_p95_tokens": float(
            projected.quantile(0.95)
        ),

        "estimated_p99_tokens": float(
            projected.quantile(0.99)
        ),
    }

    # ------------------------------------------------------------------------
    # Long-duration thresholds
    # ------------------------------------------------------------------------

    threshold_stats = {}

    for threshold in [
        5,
        8,
        10,
        12,
        15,
    ]:

        count = int(
            (valid > threshold).sum()
        )

        threshold_stats[
            f">{threshold}_sec"
        ] = {

            "count": count,

            "percentage": float(
                count
                / len(valid)
                * 100
            ),
        }

    # ------------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------------

    print()
    print(
        "Duration statistics"
    )

    print(
        "-" * 50
    )

    print(
        f"  Min:       "
        f"{stats['min_seconds']:.2f} sec"
    )

    print(
        f"  Mean:      "
        f"{stats['mean_seconds']:.2f} sec"
    )

    print(
        f"  Median:    "
        f"{stats['median_seconds']:.2f} sec"
    )

    print(
        f"  P75:       "
        f"{stats['p75_seconds']:.2f} sec"
    )

    print(
        f"  P90:       "
        f"{stats['p90_seconds']:.2f} sec"
    )

    print(
        f"  P95:       "
        f"{stats['p95_seconds']:.2f} sec"
    )

    print(
        f"  P99:       "
        f"{stats['p99_seconds']:.2f} sec"
    )

    print(
        f"  Max:       "
        f"{stats['max_seconds']:.2f} sec"
    )

    # ------------------------------------------------------------------------
    # Buckets
    # ------------------------------------------------------------------------

    print()
    print(
        "Duration buckets"
    )

    print(
        "-" * 50
    )

    for name, data in bucket_stats.items():

        print(
            f"  {name:10s} "
            f"{data['count']:6,d} "
            f"({data['percentage']:6.2f}%)"
        )

    # ------------------------------------------------------------------------
    # Projected sequence lengths
    # ------------------------------------------------------------------------

    print()
    print(
        "Estimated projected sequence lengths"
    )

    print(
        "-" * 50
    )

    print(
        f"  Mean: "
        f"{projected_stats['estimated_mean_tokens']:.0f}"
    )

    print(
        f"  P90:  "
        f"{projected_stats['estimated_p90_tokens']:.0f}"
    )

    print(
        f"  P95:  "
        f"{projected_stats['estimated_p95_tokens']:.0f}"
    )

    print(
        f"  P99:  "
        f"{projected_stats['estimated_p99_tokens']:.0f}"
    )

    print(
        f"  Max:  "
        f"{projected_stats['estimated_max_tokens']:.0f}"
    )

    # ------------------------------------------------------------------------
    # Long-duration distribution
    # ------------------------------------------------------------------------

    print()
    print(
        "Long-duration distribution"
    )

    print(
        "-" * 50
    )

    for name, data in threshold_stats.items():

        print(
            f"  {name:8s} "
            f"{data['count']:6,d} "
            f"({data['percentage']:6.2f}%)"
        )

    # ------------------------------------------------------------------------
    # Resolution status
    # ------------------------------------------------------------------------

    print()
    print(
        f"Resolved audio files: "
        f"{len(resolved_paths):,}"
    )

    print(
        f"Missing audio files:  "
        f"{len(missing_paths):,}"
    )

    # ------------------------------------------------------------------------
    # Return result
    # ------------------------------------------------------------------------

    return {

        "file": str(
            csv_path
        ),

        "columns": [
            str(column)
            for column in df.columns
        ],

        "duration_column": (
            duration_col
            if duration_col
            else None
        ),

        "audio_column": (
            audio_col
            if audio_col
            else None
        ),

        "statistics": stats,

        "buckets": bucket_stats,

        "estimated_projected_tokens": (
            projected_stats
        ),

        "thresholds": threshold_stats,

        "resolved_audio_count": len(
            resolved_paths
        ),

        "missing_audio_count": len(
            missing_paths
        ),

        "missing_audio_examples": (
            missing_paths[:20]
        ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "V2 DATASET DURATION ANALYZER"
    )
    print("=" * 78)

    print()
    print(
        f"Project directory: "
        f"{PROJECT_DIR}"
    )

    print(
        f"ASR directory:     "
        f"{ASR_DIR}"
    )

    print(
        f"Feature rate:      "
        f"{FEATURE_RATE:.1f} frames/sec"
    )

    print(
        f"Projected rate:    "
        f"{PROJECTED_RATE:.1f} tokens/sec"
    )

    # ------------------------------------------------------------------------
    # Verify dataset files
    # ------------------------------------------------------------------------

    print()
    print(
        "Dataset files"
    )

    print(
        "-" * 50
    )

    for csv_path in [
        TRAIN_CSV,
        VAL_CSV,
        TEST_CSV,
    ]:

        print(
            f"{csv_path.name:12s} "
            f"{'FOUND' if csv_path.exists() else 'MISSING'}"
        )

    # ------------------------------------------------------------------------
    # Analyze datasets
    # ------------------------------------------------------------------------

    results = {}

    for name, csv_path in [

        ("train", TRAIN_CSV),

        ("val", VAL_CSV),

        ("test", TEST_CSV),
    ]:

        results[name] = analyze_dataset(
            csv_path
        )

    # ------------------------------------------------------------------------
    # Global summary
    # ------------------------------------------------------------------------

    total_rows = sum(
        result["statistics"]["rows"]
        for result in results.values()
    )

    total_resolved = sum(
        result["resolved_audio_count"]
        for result in results.values()
    )

    total_missing = sum(
        result["missing_audio_count"]
        for result in results.values()
    )

    output = {

        "feature_rate": FEATURE_RATE,

        "projected_rate": PROJECTED_RATE,

        "project_directory": str(
            PROJECT_DIR
        ),

        "asr_directory": str(
            ASR_DIR
        ),

        "total_rows": total_rows,

        "total_resolved_audio": (
            total_resolved
        ),

        "total_missing_audio": (
            total_missing
        ),

        "datasets": results,
    }

    # ------------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------------

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "ANALYSIS COMPLETE"
    )
    print("=" * 78)

    print()
    print(
        f"Total rows:          "
        f"{total_rows:,}"
    )

    print(
        f"Resolved audio:      "
        f"{total_resolved:,}"
    )

    print(
        f"Missing audio:       "
        f"{total_missing:,}"
    )

    print()
    print(
        f"Saved: {OUTPUT_JSON}"
    )

    print()

    # ------------------------------------------------------------------------
    # Explicit dataset integrity check
    # ------------------------------------------------------------------------

    expected = {
        "train": 10199,
        "val": 1207,
        "test": 601,
    }

    print(
        "DATASET INTEGRITY CHECK"
    )

    print(
        "-" * 50
    )

    integrity_ok = True

    for name, expected_count in expected.items():

        actual_count = results[
            name
        ]["statistics"]["rows"]

        status = (
            "OK"
            if actual_count == expected_count
            else "MISMATCH"
        )

        if status != "OK":
            integrity_ok = False

        print(
            f"{name.upper():8s} "
            f"{actual_count:6,d} / "
            f"{expected_count:6,d} "
            f"[{status}]"
        )

    print()

    if integrity_ok:
        print(
            "Dataset row counts are correct."
        )
    else:
        print(
            "WARNING: Dataset row counts "
            "do not match expected values."
        )

    print()

    # ------------------------------------------------------------------------
    # Resolution check
    # ------------------------------------------------------------------------

    if total_missing == 0:

        print(
            "ALL AUDIO FILES RESOLVED."
        )

    else:

        print(
            f"WARNING: {total_missing:,} "
            f"audio files could not be resolved."
        )

        print(
            "Missing examples are stored in:"
        )

        print(
            OUTPUT_JSON
        )

    print()


if __name__ == "__main__":
    main()