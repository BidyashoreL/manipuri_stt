from pathlib import Path
import pandas as pd

source = Path("audio_audit/all_splits_silence_report.csv")
outdir = Path("audio_audit/cleaning_plan")
outdir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source).fillna({
    "flags": "",
    "error": "",
    "active_speech_ratio": 0,
    "speech_sec_vad": 0,
    "duration_sec": 0,
    "leading_silence_sec": 0,
    "trailing_silence_sec": 0,
    "longest_internal_gap_sec": 0,
})

def decide(row):
    flags = set(str(row["flags"]).split(";"))
    ratio = float(row["active_speech_ratio"])
    speech = float(row["speech_sec_vad"])
    error = str(row["error"]).strip()

    if error:
        return "CHECK_AUDIO_ERROR"

    if "no_speech_detected" in flags or speech == 0:
        return "LISTEN_FIRST_NO_SPEECH"

    if "very_low_speech_ratio" in flags or ratio < 0.25:
        return "LISTEN_FIRST_LOW_SPEECH"

    if "long_internal_pause" in flags:
        return "LISTEN_FIRST_INTERNAL_PAUSE"

    if flags.intersection({
        "long_leading_silence",
        "long_trailing_silence"
    }):
        return "CHECK_SILENCE_TRIMMING"

    return "NO_SILENCE_FLAG_REVIEW_TRANSCRIPT"

df["recommended_action"] = df.apply(decide, axis=1)

# Preserve the source report; this writes new files only.
df.to_csv(outdir / "all_entries_cleaning_plan.csv", index=False)

print("\n========== CLEANING PLAN SUMMARY ==========")
print("Total entries:", len(df))
print("\nRecommended action counts:")
print(df["recommended_action"].value_counts().to_string())

print("\nBy split and action:")
if "split" in df.columns:
    print(pd.crosstab(
        df["split"], df["recommended_action"]
    ).to_string())

print("\nBy source and action:")
print(pd.crosstab(
    df["source"], df["recommended_action"]
).to_string())

print("\nReport saved to:")
print(outdir / "all_entries_cleaning_plan.csv")
print("\nNo audio or original manifests were modified.")
