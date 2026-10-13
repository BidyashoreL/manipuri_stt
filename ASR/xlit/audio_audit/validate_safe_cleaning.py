from pathlib import Path
import pandas as pd
import soundfile as sf

base = Path("audio_audit/safe_cleaned")
reports = [
    base / "train_before_after.csv",
    base / "validation_before_after.csv",
    base / "test_before_after.csv",
]

all_rows = []

for report_path in reports:
    df = pd.read_csv(report_path)
    all_rows.append(df)

df = pd.concat(all_rows, ignore_index=True)

missing = []
bad_audio = []
very_short = []
large_trim = []

for _, row in df.iterrows():
    path = Path(str(row["cleaned_path"]))
    if not path.is_file():
        missing.append(str(path))
        continue

    try:
        info = sf.info(str(path))
        if info.frames == 0 or info.samplerate != 16000:
            bad_audio.append({
                "path": str(path),
                "sample_rate": info.samplerate,
                "frames": info.frames,
            })

        duration = info.frames / info.samplerate
        if duration < 0.5:
            very_short.append({
                "path": str(path),
                "duration_sec": round(duration, 3),
                "text": row.get("text", ""),
                "split": row.get("split", ""),
            })

        removed = (
            float(row["duration_before_sec"])
            - float(row["duration_after_sec"])
        )
        if removed > 2:
            large_trim.append({
                "path": str(path),
                "split": row.get("split", ""),
                "seconds_removed": round(removed, 3),
                "text": row.get("text", ""),
            })

    except Exception as exc:
        bad_audio.append({
            "path": str(path),
            "error": str(exc),
        })

df["seconds_removed"] = (
    df["duration_before_sec"] - df["duration_after_sec"]
).clip(lower=0)

out = base / "validation_checks"
out.mkdir(exist_ok=True)

pd.DataFrame(very_short).to_csv(
    out / "very_short_cleaned_audio.csv", index=False
)
pd.DataFrame(large_trim).to_csv(
    out / "large_trim_review.csv", index=False
)
pd.DataFrame(bad_audio).to_csv(
    out / "invalid_audio.csv", index=False
)
pd.DataFrame({"missing_path": missing}).to_csv(
    out / "missing_audio.csv", index=False
)

print("\n========== CLEANED AUDIO VALIDATION ==========")
print("Manifest entries:", len(df))
print("Missing cleaned files:", len(missing))
print("Invalid audio files:", len(bad_audio))
print("Files shorter than 0.5 seconds:", len(very_short))
print("Files with over 2 seconds removed:", len(large_trim))
print("Total seconds removed:", round(df["seconds_removed"].sum(), 2))
print("\nReview reports:", out.resolve())
