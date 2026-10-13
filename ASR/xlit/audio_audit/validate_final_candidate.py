from pathlib import Path
import pandas as pd
import soundfile as sf

root = Path.cwd()
folder = root / "audio_audit" / "final_candidate"
manifests = {}
errors = []

print("\n========== FINAL DATASET INTEGRITY AUDIT ==========")

for split in ["train", "validation"]:
    path = folder / f"{split}.csv"
    df = pd.read_csv(path).fillna("")
    manifests[split] = df

    missing = 0
    unreadable = 0
    wrong_rate = 0
    empty_text = int(df["text"].astype(str).str.strip().eq("").sum())
    duplicate_paths = int(df["audio_path"].astype(str).duplicated().sum())
    sample_rates = {}

    for audio_path in df["audio_path"].astype(str):
        p = Path(audio_path)
        if not p.is_file():
            missing += 1
            errors.append((split, audio_path, "missing"))
            continue

        try:
            info = sf.info(str(p))
            sample_rates[info.samplerate] = sample_rates.get(info.samplerate, 0) + 1
            if info.frames <= 0 or info.duration <= 0:
                unreadable += 1
                errors.append((split, audio_path, "empty_audio"))
            if info.samplerate != 16000:
                wrong_rate += 1
        except Exception as e:
            unreadable += 1
            errors.append((split, audio_path, f"unreadable: {e}"))

    print(f"\n{split.upper()}")
    print("Entries:", len(df))
    print("Missing files:", missing)
    print("Unreadable/empty audio:", unreadable)
    print("Non-16-kHz files:", wrong_rate)
    print("Sample-rate counts:", sample_rates)
    print("Empty transcripts:", empty_text)
    print("Duplicate paths:", duplicate_paths)

train_paths = set(manifests["train"]["audio_path"].astype(str))
val_paths = set(manifests["validation"]["audio_path"].astype(str))
overlap = train_paths & val_paths

print("\n========== SPLIT CHECK ==========")
print("Train-validation path overlap:", len(overlap))

if errors:
    pd.DataFrame(errors, columns=["split", "audio_path", "issue"]).to_csv(
        folder / "integrity_errors.csv", index=False
    )
    print("Error report:", folder / "integrity_errors.csv")

if overlap:
    print("WARNING: paths appear in both training and validation.")

if errors or overlap:
    print("\nAUDIT RESULT: REVIEW REQUIRED")
else:
    print("\nAUDIT RESULT: PATHS AND AUDIO FILES PASSED BASIC CHECKS")
    print("Non-16-kHz files, if any, must be resampled during preprocessing.")
