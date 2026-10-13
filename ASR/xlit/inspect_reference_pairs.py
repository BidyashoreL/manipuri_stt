from pathlib import Path
import pandas as pd

path = Path("data/transcribed.txt")

df = pd.read_csv(
    path,
    sep="\t",
    header=None,
    names=["bengali", "reference_meitei_mayek", "category"],
    dtype=str,
    encoding="utf-8"
)

print("Total reference pairs:", len(df))
print("Missing Bengali words:", df["bengali"].isna().sum())
print("Missing Meitei Mayek references:",
      df["reference_meitei_mayek"].isna().sum())

print("\nCategory distribution:")
print(df["category"].value_counts(dropna=False).sort_index())

print("\nFirst five reference pairs:")
for _, row in df.head().iterrows():
    print("Bengali:", row["bengali"])
    print("Reference:", row["reference_meitei_mayek"])
    print("Category:", row["category"])
    print()

