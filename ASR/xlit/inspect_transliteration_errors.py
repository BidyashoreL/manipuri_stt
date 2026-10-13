import pandas as pd
from pathlib import Path

base = Path(
    r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt\Manipuri_Datasets\02_IndicTTS"
)

metadata = pd.read_csv(base / "metadata.csv", dtype=str).fillna("")
errors = pd.read_csv(base / "transliteration_errors.csv", dtype=str).fillna("")

print("Affected sentences:\n")

for row_number in sorted(set(errors["row"].astype(int))):
    # Error row numbers are CSV line numbers: header is line 1.
    index = row_number - 2

    if 0 <= index < len(metadata):
        row = metadata.iloc[index]
        print(f"CSV row: {row_number}")
        print("Audio:", row["file_name"])
        print("Original:", row["text"])
        print()

