from pathlib import Path
import pandas as pd

base = Path("audio_audit/silence_statistics")
files = [
    base / "final_train_16k.csv",
    base / "final_validation_16k.csv",
]

frames = []
for file in files:
    df = pd.read_csv(file)
    df["review_reasons"] = ""

    conditions = [
        ("no_speech_detected", df["flags"].fillna("").str.contains("no_speech_detected")),
        ("low_speech_ratio", pd.to_numeric(df["speech_ratio"], errors="coerce") < 0.20),
        ("long_leading_silence", pd.to_numeric(df["leading_silence_sec"], errors="coerce") > 0.5),
        ("long_trailing_silence", pd.to_numeric(df["trailing_silence_sec"], errors="coerce") > 0.5),
        ("long_internal_pause", pd.to_numeric(df["longest_internal_gap_sec"], errors="coerce") > 2.0),
    ]

    for reason, mask in conditions:
        df.loc[mask.fillna(False), "review_reasons"] += reason + ";"

    flagged = df[df["review_reasons"] != ""].copy()
    frames.append(flagged)

review = pd.concat(frames, ignore_index=True)
out = base / "prioritized_review.csv"
review.to_csv(out, index=False, encoding="utf-8-sig")

print("Flagged rows (categories overlap):", len(review))
print("Unique audio paths:", review["audio_path"].nunique())
print("\nCounts by reason:")
for reason, _ in conditions:
    print(f"{reason}: {(review['review_reasons'].str.contains(reason)).sum()}")

print("\nSaved:", out.resolve())
print("No audio was changed.")
