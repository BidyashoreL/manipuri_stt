from pathlib import Path
import pandas as pd

base = Path("audio_audit/safe_cleaned/validation_checks")

short = pd.read_csv(base / "very_short_cleaned_audio.csv")
long = pd.read_csv(base / "large_trim_review.csv")

short_paths = set(short["path"].astype(str))
long_paths = set(long["path"].astype(str))

print("\n========== REVIEW OVERLAP ==========")
print("Short clips:", len(short_paths))
print("Heavily trimmed clips:", len(long_paths))
print("Both short and heavily trimmed:",
      len(short_paths & long_paths))

short["review_reason"] = "cleaned_duration_under_0.5s"
long["review_reason"] = "more_than_2s_removed"

combined = pd.concat([short, long], ignore_index=True, sort=False)
combined = combined.drop_duplicates(subset=["path"], keep="first")

# Merge reasons so overlapping clips receive both flags.
reasons = {}
for p in short_paths:
    reasons.setdefault(p, []).append("short_after_cleaning")
for p in long_paths:
    reasons.setdefault(p, []).append("more_than_2s_removed")

combined["review_reason"] = combined["path"].astype(str).map(
    lambda p: ";".join(reasons.get(p, []))
)

combined.to_csv(base / "combined_manual_review.csv", index=False)

print("Unique clips needing review:", len(combined))
print("Saved:", base / "combined_manual_review.csv")
print("\nCounts by split:")
if "split" in combined.columns:
    print(combined["split"].value_counts(dropna=False).to_string())
