
from pathlib import Path
import pandas as pd
import random

ROOT = Path(
    r"E:\manipuri_stt1\manipuri_stt4"
    r"\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt"
)

DATA = ROOT / "Manipuri_Datasets"
IV = DATA / "01_IndicVoices" / "recovered_audio"
TTS = DATA / "02_IndicTTS"
OUT = ROOT / "ASR" / "xlit" / "combined_dataset"

OUT.mkdir(parents=True, exist_ok=True)
random.seed(42)


def load_indicvoices(filename, split):
    df = pd.read_csv(IV / filename, dtype=str).fillna("")
    rows = []

    for _, r in df.iterrows():
        audio = Path(r.get("audio_path", ""))
        if not audio.is_file():
            continue

        text = (r.get("normalized", "") or r.get("text", "")).strip()
        if not text:
            continue

        rows.append({
            "audio_path": str(audio.resolve()),
            "text": text,
            "source": "IndicVoices",
            "speaker_id": r.get("speaker_id", ""),
        })

    print(f"IndicVoices {split}: {len(rows)}")
    return rows


def load_indictts():
    df = pd.read_csv(
        TTS / "metadata_meitei_mayek_preserved.csv",
        dtype=str
    ).fillna("")
    rows = []

    for _, r in df.iterrows():
        filename = (r.get("file_name", "")).strip()
        text = (r.get("text_meitei_mayek", "")).strip()
        audio = TTS / "audio" / filename

        if audio.is_file() and text:
            rows.append({
                "audio_path": str(audio.resolve()),
                "text": text,
                "source": "IndicTTS",
                "speaker_id": "",
            })

    print(f"IndicTTS usable: {len(rows)}")
    return rows


# Preserve the original IndicVoices assignments.
iv_train = load_indicvoices("train.csv", "train")
iv_val = load_indicvoices("val.csv", "validation")
iv_test = load_indicvoices("test.csv", "test")

tts = load_indictts()

# Prevent IndicVoices test audio from entering training or validation.
test_paths = {r["audio_path"].casefold() for r in iv_test}
val_paths = {r["audio_path"].casefold() for r in iv_val}
train_paths = {r["audio_path"].casefold() for r in iv_train}

tts_train = []
tts_val = []

random.shuffle(tts)

for row in tts:
    key = row["audio_path"].casefold()

    if key in test_paths or key in val_paths or key in train_paths:
        continue

    if random.random() < 0.05:
        tts_val.append(row)
    else:
        tts_train.append(row)

# Combine the two training sources.
train = iv_train + tts_train
validation = iv_val + tts_val
test = iv_test

# Verify that no audio path occurs across splits.
def paths(rows):
    return {r["audio_path"].casefold() for r in rows}

assert not (paths(train) & paths(validation)), "Train/validation overlap!"
assert not (paths(train) & paths(test)), "Train/test overlap!"
assert not (paths(validation) & paths(test)), "Validation/test overlap!"

for name, rows in [
    ("train", train),
    ("validation", validation),
    ("test", test),
]:
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"{name}: {len(df)} rows")
    print(df["source"].value_counts().to_dict())

print(f"\nSaved manifests to: {OUT}")
print("Original datasets were not modified.")
