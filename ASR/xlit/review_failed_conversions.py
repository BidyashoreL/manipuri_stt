import pandas as pd
from pathlib import Path

base = Path(
    r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt\Manipuri_Datasets\02_IndicTTS"
)

metadata = pd.read_csv(
    base / "metadata_meitei_mayek_preserved.csv",
    dtype=str
).fillna("")

errors = pd.read_csv(
    base / "transliteration_errors.csv",
    dtype=str
).fillna("")

for row_number in sorted(set(errors["row"].astype(int))):
    i = row_number - 2
    if 0 <= i < len(metadata):
        r = metadata.iloc[i]
        print(f"\nCSV row: {row_number} | Audio: {r['file_name']}")
        print("Original :", r["text"])
        print("Converted:", r["text_meitei_mayek"])

