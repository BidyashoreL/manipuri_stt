from pathlib import Path
import pandas as pd

base = Path(
    r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt\Manipuri_Datasets\02_IndicTTS"
)

metadata_path = base / "metadata_meitei_mayek_preserved.csv"
audio_dir = base / "audio"

df = pd.read_csv(metadata_path, dtype=str).fillna("")

missing = []
for filename in df["file_name"]:
    path = audio_dir / filename
    if not path.is_file():
        missing.append(filename)

print("Total metadata rows:", len(df))
print("Unique audio filenames:", df["file_name"].nunique())
print("Audio files found:", len(df) - len(missing))
print("Missing audio files:", len(missing))

if missing:
    print("\nFirst missing files:")
    for filename in missing[:20]:
        print(filename)
else:
    print("\nPASS: Every metadata filename has a matching WAV file.")

print("\nOriginal transcript column present:", "text" in df.columns)
print("Meitei Mayek column present:", "text_meitei_mayek" in df.columns)
