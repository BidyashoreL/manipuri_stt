from pathlib import Path
import csv
import sys
import traceback

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly

# ============================================================
# CONFIGURATION: FINAL CLEANED DATASET ONLY
# Run this script from your existing xlit directory.
# ============================================================

DATASETS = [
    {
        "name": "final_train_16k",
        "manifest": Path("audio_audit/final_candidate/train_16k.csv"),
        "expected_rows": 23234,
    },
    {
        "name": "final_validation_16k",
        "manifest": Path("audio_audit/final_candidate/validation_16k.csv"),
        "expected_rows": 1693,
    },
]

OUTPUT_DIR = Path("audio_audit/silence_statistics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VAD_THRESHOLD = 0.5
MIN_SPEECH_MS = 100
MIN_SILENCE_MS = 300
PROGRESS_EVERY = 500


def find_column(df, candidates, required=True):
    """Find a manifest column without assuming exact capitalization."""
    lookup = {str(c).strip().lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    # Allow common column names with prefixes/suffixes.
    for candidate in candidates:
        for normalized, original in lookup.items():
            if normalized.endswith("_" + candidate.lower()):
                return original

    if required:
        raise ValueError(
            f"Could not find any of {candidates}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def resolve_audio_path(value, manifest_path):
    """Resolve absolute paths and common relative-path layouts."""
    raw = str(value).strip().strip('"')
    p = Path(raw)

    if p.is_absolute():
        return p

    candidates = [
        Path.cwd() / p,
        manifest_path.parent / p,
        manifest_path.parent.parent / p,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    # Return the most likely location for a useful error message.
    return candidates[0].resolve()


def load_vad():
    print("\nLoading Silero VAD...")
    model, utilities = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )

    get_speech_timestamps = utilities[0]
    model.eval()

    return model, get_speech_timestamps


def analyze_audio(audio_path, model, get_speech_timestamps):
    info = sf.info(str(audio_path))

    # Read audio without modifying the original file.
    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )

    # Convert to mono for VAD.
    audio = audio.mean(axis=1)

    if audio.size == 0:
        raise ValueError("Audio contains no samples")

    original_duration = len(audio) / sample_rate

    # Silero VAD supports 8 kHz or 16 kHz.
    if sample_rate not in (8000, 16000):
        gcd = np.gcd(sample_rate, 16000)
        audio = resample_poly(
            audio,
            16000 // gcd,
            sample_rate // gcd,
        ).astype(np.float32)
        vad_sample_rate = 16000
    else:
        vad_sample_rate = sample_rate

    waveform = torch.from_numpy(
        np.ascontiguousarray(audio, dtype=np.float32)
    )

    with torch.inference_mode():
        segments = get_speech_timestamps(
            waveform,
            model,
            threshold=VAD_THRESHOLD,
            sampling_rate=vad_sample_rate,
            min_speech_duration_ms=MIN_SPEECH_MS,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=0,
            return_seconds=False,
        )

    duration = len(waveform) / vad_sample_rate

    if not segments:
        return {
            "sample_rate_original": info.samplerate,
            "channels_original": info.channels,
            "duration_sec": round(original_duration, 4),
            "speech_duration_est_sec": 0.0,
            "speech_ratio": 0.0,
            "leading_silence_sec": round(duration, 4),
            "trailing_silence_sec": round(duration, 4),
            "longest_internal_gap_sec": 0.0,
            "speech_segments": 0,
            "flags": "no_speech_detected",
        }

    intervals = []
    for segment in segments:
        start = max(0, int(segment["start"])) / vad_sample_rate
        end = min(len(waveform), int(segment["end"])) / vad_sample_rate
        if end > start:
            intervals.append((start, end))

    if not intervals:
        raise ValueError("VAD returned invalid speech intervals")

    speech_duration = sum(end - start for start, end in intervals)
    leading = intervals[0][0]
    trailing = max(0.0, duration - intervals[-1][1])

    internal_gaps = [
        max(0.0, intervals[i + 1][0] - intervals[i][1])
        for i in range(len(intervals) - 1)
    ]
    longest_gap = max(internal_gaps, default=0.0)
    ratio = speech_duration / duration if duration else 0.0

    flags = []
    if ratio < 0.20:
        flags.append("low_speech_ratio")
    if leading > 0.5:
        flags.append("long_leading_silence")
    if trailing > 0.5:
        flags.append("long_trailing_silence")
    if longest_gap > 2.0:
        flags.append("long_internal_pause")
    if duration < 0.5:
        flags.append("very_short_clip")

    return {
        "sample_rate_original": info.samplerate,
        "channels_original": info.channels,
        "duration_sec": round(original_duration, 4),
        "speech_duration_est_sec": round(speech_duration, 4),
        "speech_ratio": round(ratio, 5),
        "leading_silence_sec": round(leading, 4),
        "trailing_silence_sec": round(trailing, 4),
        "longest_internal_gap_sec": round(longest_gap, 4),
        "speech_segments": len(intervals),
        "flags": ";".join(flags) if flags else "none",
    }


def main():
    print("=" * 68)
    print("MANIPURI ASR — FINAL 16 kHz DATASET SILENCE STATISTICS")
    print("=" * 68)
    print("Training manifest:   audio_audit/final_candidate/train_16k.csv")
    print("Validation manifest: audio_audit/final_candidate/validation_16k.csv")
    print("Test set:            NOT LOADED")
    print("Audio modification:  NONE")
    print(f"Output directory:    {OUTPUT_DIR.resolve()}")

    # Validate manifests before downloading/loading the VAD model.
    loaded = []

    for config in DATASETS:
        manifest = config["manifest"]

        if not manifest.is_file():
            raise FileNotFoundError(
                f"\nManifest not found:\n{manifest.resolve()}\n"
                "Run this command from your xlit project directory."
            )

        df = pd.read_csv(manifest)
        if len(df) != config["expected_rows"]:
            raise ValueError(
                f"{config['name']}: expected "
                f"{config['expected_rows']:,} rows, found {len(df):,}. "
                "Stopping to avoid auditing the wrong dataset."
            )

        audio_col = find_column(
            df,
            ["audio_path", "audio_filepath", "filepath",
             "file_path", "path", "audio", "file", "filename"],
        )
        text_col = find_column(
            df,
            ["text", "transcript", "sentence", "transcription",
             "normalized_text", "target"],
            required=False,
        )
        speaker_col = find_column(
            df,
            ["speaker_id", "speaker", "speakerid"],
            required=False,
        )
        source_col = find_column(
            df,
            ["source", "dataset", "corpus"],
            required=False,
        )

        loaded.append(
            (config, df, audio_col, text_col, speaker_col, source_col)
        )
        print(
            f"\nManifest verified: {config['name']} "
            f"— {len(df):,} rows"
        )

    model, get_speech_timestamps = load_vad()
    summaries = []

    for config, df, audio_col, text_col, speaker_col, source_col in loaded:
        name = config["name"]
        manifest = config["manifest"]
        output_csv = OUTPUT_DIR / f"{name}.csv"

        print(f"\n{'=' * 60}")
        print(f"Auditing {name}: {len(df):,} entries")
        print(f"Writing: {output_csv}")
        print(f"{'=' * 60}")

        results = []
        missing = 0
        errors = 0
        no_speech = 0

        for index, row in df.iterrows():
            audio_path = resolve_audio_path(row[audio_col], manifest)

            result = {
                "split": name,
                "row_number": index + 1,
                "audio_path": str(audio_path),
                "text": (
                    str(row[text_col])
                    if text_col is not None and pd.notna(row[text_col])
                    else ""
                ),
                "speaker_id": (
                    str(row[speaker_col])
                    if speaker_col is not None and pd.notna(row[speaker_col])
                    else ""
                ),
                "source": (
                    str(row[source_col])
                    if source_col is not None and pd.notna(row[source_col])
                    else ""
                ),
            }

            if not audio_path.is_file():
                missing += 1
                result.update({
                    "sample_rate_original": None,
                    "channels_original": None,
                    "duration_sec": None,
                    "speech_duration_est_sec": None,
                    "speech_ratio": None,
                    "leading_silence_sec": None,
                    "trailing_silence_sec": None,
                    "longest_internal_gap_sec": None,
                    "speech_segments": None,
                    "flags": "missing_audio",
                })
            else:
                try:
                    result.update(
                        analyze_audio(
                            audio_path,
                            model,
                            get_speech_timestamps,
                        )
                    )
                    if "no_speech_detected" in result["flags"]:
                        no_speech += 1
                except Exception as exc:
                    errors += 1
                    result.update({
                        "sample_rate_original": None,
                        "channels_original": None,
                        "duration_sec": None,
                        "speech_duration_est_sec": None,
                        "speech_ratio": None,
                        "leading_silence_sec": None,
                        "trailing_silence_sec": None,
                        "longest_internal_gap_sec": None,
                        "speech_segments": None,
                        "flags": f"analysis_error: {type(exc).__name__}: {exc}",
                    })

            results.append(result)

            completed = index + 1
            if completed % PROGRESS_EVERY == 0 or completed == len(df):
                print(
                    f"  {completed:,}/{len(df):,} | "
                    f"missing={missing} | errors={errors} | "
                    f"no_speech={no_speech}"
                )

        out_df = pd.DataFrame(results)
        out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

        valid = out_df[
            pd.to_numeric(out_df["speech_ratio"], errors="coerce").notna()
        ]

        summary = {
            "dataset": name,
            "manifest_rows": len(df),
            "audited_rows": len(out_df),
            "missing_audio": missing,
            "analysis_errors": errors,
            "no_speech_detected": no_speech,
            "low_speech_ratio_lt_0.20": int(
                (pd.to_numeric(valid["speech_ratio"]) < 0.20).sum()
            ) if len(valid) else 0,
            "leading_silence_gt_0.5s": int(
                (pd.to_numeric(valid["leading_silence_sec"]) > 0.5).sum()
            ) if len(valid) else 0,
            "trailing_silence_gt_0.5s": int(
                (pd.to_numeric(valid["trailing_silence_sec"]) > 0.5).sum()
            ) if len(valid) else 0,
            "internal_pause_gt_2s": int(
                (pd.to_numeric(valid["longest_internal_gap_sec"]) > 2.0).sum()
            ) if len(valid) else 0,
            "output_csv": str(output_csv.resolve()),
        }
        summaries.append(summary)

        print(f"\nCompleted {name}")
        print(f"  Missing audio:       {missing}")
        print(f"  Analysis errors:     {errors}")
        print(f"  No speech detected:  {no_speech}")
        print(f"  Results:             {output_csv.resolve()}")

    summary_path = OUTPUT_DIR / "summary.csv"
    pd.DataFrame(summaries).to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )

    print("\n" + "=" * 68)
    print("AUDIT FINISHED")
    print("=" * 68)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nSummary saved to: {summary_path.resolve()}")
    print("\nNo audio was trimmed or modified.")
    print("The test benchmark was not opened or evaluated.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL ERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
