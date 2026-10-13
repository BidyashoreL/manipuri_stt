import pandas as pd
from pathlib import Path

p = Path(r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt\Manipuri_Datasets\02_IndicTTS\metadata_meitei_mayek.csv")
df = pd.read_csv(p, dtype=str).fillna("")

print("Rows:", len(df))
print("Original transcripts with spaces:",
      df["text"].str.contains(r"\s", regex=True).sum())
print("Converted transcripts with spaces:",
      df["text_meitei_mayek"].str.contains(r"\s", regex=True).sum())
print("Original transcripts with punctuation:",
      df["text"].str.contains(r"[,.;!?।]", regex=True).sum())
print("Converted transcripts with punctuation:",
      df["text_meitei_mayek"].str.contains(r"[,.;!?।]", regex=True).sum())

print("\nExamples:")
for _, row in df.head(10).iterrows():
    print("ORIGINAL :", row["text"])
    print("CONVERTED:", row["text_meitei_mayek"])
    print()
