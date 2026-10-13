import pandas as pd
from pathlib import Path

folder = Path("combined_dataset")

for name in ["train", "validation", "test"]:
    df = pd.read_csv(folder / f"{name}.csv", encoding="utf-8-sig")

    missing = sum(
        not Path(str(path)).is_file()
        for path in df["audio_path"]
    )
    empty = df["text"].fillna("").str.strip().eq("").sum()

    print(f"\n{name.upper()}")
    print(f"Rows: {len(df)}")
    print(f"Missing audio: {missing}")
    print(f"Empty transcripts: {empty}")
    print("Sources:")
    print(df["source"].value_counts().to_string())
