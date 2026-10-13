from pathlib import Path
import pandas as pd
import soundfile as sf

base = Path("audio_audit/final_candidate")
errors = []

for split in ["train", "validation"]:
    csv_path = base / f"{split}_16k.csv"
    df = pd.read_csv(csv_path).fillna("")

    missing = 0
    bad_audio = 0
    wrong_format = 0
    empty_text = int(df["text"].astype(str).str.strip().eq("").sum())

    for p in df["audio_path"].astype(str):
        if not Path(p).is_file():
            missing += 1
            errors.append((split, p, "missing"))
            continue

        try:
            info = sf.info(p)
            if info.frames <= 0 or info.duration <= 0:
                bad_audio += 1
                errors.append((split, p, "empty_audio"))
            if info.samplerate != 16000 or info.channels != 1:
                wrong_format += 1
                errors.append((split, p, "wrong_format"))
        except Exception as e:
            bad_audio += 1
            errors.append((split, p, str(e)))

    print(f"\n{split.upper()}")
    print("Entries:", len(df))
    print("Missing audio:", missing)
    print("Unreadable/empty audio:", bad_audio)
    print("Wrong sample rate/channels:", wrong_format)
    print("Empty transcripts:", empty_text)

    if missing or bad_audio or wrong_format or empty_text:
        print("STATUS: REVIEW REQUIRED")
    else:
        print("STATUS: PASS")

train = pd.read_csv(base / "train_16k.csv")
val = pd.read_csv(base / "validation_16k.csv")
overlap = set(train["audio_path"]) & set(val["audio_path"])

print("\nTrain-validation path overlap:", len(overlap))

if errors:
    pd.DataFrame(errors, columns=["split", "audio_path", "error"]).to_csv(
        base / "audit_16k_errors.csv", index=False
    )
    print("Error report saved:", base / "audit_16k_errors.csv")

if not errors and not overlap:
    print("\nFINAL RESULT: 16 kHz DATASET PASSED")
