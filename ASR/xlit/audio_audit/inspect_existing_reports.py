from pathlib import Path
import pandas as pd

base = Path("audio_audit")
reports = [
    "all_splits_silence_report.csv",
    "train_silence_report.csv",
    "validation_silence_report.csv",
    "test_silence_report.csv",
    "clips_for_review.csv",
]

for name in reports:
    path = base / name
    if not path.exists():
        print(f"\n{name}: NOT FOUND")
        continue

    try:
        df = pd.read_csv(path)
        print(f"\n--- {name} ---")
        print("Rows:", len(df))
        print("Columns:", ", ".join(df.columns))

        if "source" in df.columns:
            print("Sources:")
            print(df["source"].value_counts(dropna=False).to_string())

        if "flags" in df.columns:
            print("Flag counts:")
            print(
                df["flags"].fillna("").str.split(";").explode()
                  .loc[lambda s: s.ne("")]
                  .value_counts().to_string()
            )

        if "audio_path" in df.columns:
            exists = df["audio_path"].dropna().map(
                lambda p: Path(str(p)).is_file()
            )
            print("Existing audio paths:", int(exists.sum()))
            print("Missing audio paths:", int((~exists).sum()))

    except Exception as e:
        print("Could not read report:", e)
