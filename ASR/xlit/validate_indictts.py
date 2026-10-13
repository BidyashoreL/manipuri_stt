
from pathlib import Path
import pandas as pd
import re

DATA = Path(
    r"E:\manipuri_stt1\manipuri_stt4"
    r"\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt\Manipuri_Datasets\02_IndicTTS"
)

original_path = DATA / "metadata.csv"
converted_path = DATA / "metadata_meitei_mayek_preserved.csv"

original = pd.read_csv(
    original_path, encoding="utf-8-sig", keep_default_na=False
)
converted = pd.read_csv(
    converted_path, encoding="utf-8-sig", keep_default_na=False
)

# Validate row counts and audio filename alignment.
if len(original) != len(converted):
    raise ValueError("Row counts differ.")

if original["file_name"].tolist() != converted["file_name"].tolist():
    raise ValueError("Filename alignment differs.")

# IMPORTANT: explicitly select the converted transcript column.
if "text_meitei_mayek" not in converted.columns:
    raise ValueError("Missing text_meitei_mayek column.")

report = pd.DataFrame({
    "file_name": original["file_name"],
    "original_text": original["text"],
    "converted_text": converted["text_meitei_mayek"],
})

def has_bengali_letters(text):
    return bool(re.search(r"[\u0980-\u09FF]", str(text)))

def has_meitei_mayek(text):
    return bool(re.search(r"[\uABC0-\uABFF]", str(text)))

report["empty_original"] = report["original_text"].str.strip().eq("")
report["empty_converted"] = report["converted_text"].str.strip().eq("")
report["bengali_letters_remaining"] = report["converted_text"].map(
    has_bengali_letters
)
report["has_meitei_mayek"] = report["converted_text"].map(
    has_meitei_mayek
)
report["unchanged_text"] = (
    report["original_text"] == report["converted_text"]
)

flagged = report[
    report["empty_original"]
    | report["empty_converted"]
    | report["bengali_letters_remaining"]
    | ~report["has_meitei_mayek"]
]

output = DATA / "indictts_transcript_validation_report.csv"
flagged.to_csv(output, index=False, encoding="utf-8-sig")

print("=== CORRECTED VALIDATION ===")
print("Rows checked:", len(report))
print("Empty original transcripts:", int(report["empty_original"].sum()))
print("Empty converted transcripts:", int(report["empty_converted"].sum()))
print(
    "Converted rows retaining Bengali letters:",
    int(report["bengali_letters_remaining"].sum())
)
print(
    "Converted rows containing Meitei Mayek:",
    int(report["has_meitei_mayek"].sum())
)
print("Unchanged transcripts:", int(report["unchanged_text"].sum()))
print("Flagged rows:", len(flagged))
print("Report saved to:", output)

print("\n=== FIRST FIVE CONVERSIONS ===")
print(
    report[
        ["original_text", "converted_text"]
    ].head(5).to_string(index=False)
)

print("\n=== FIRST TEN FLAGGED ROWS ===")
if flagged.empty:
    print("No rows flagged by these structural checks.")
else:
    print(flagged.head(10).to_string(index=False))
