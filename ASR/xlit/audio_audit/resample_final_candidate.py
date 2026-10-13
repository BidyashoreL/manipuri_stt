from pathlib import Path
from math import gcd
import pandas as pd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path.cwd()
BASE = ROOT / "audio_audit" / "final_candidate"
OUT = BASE / "audio_16k"
OUT.mkdir(parents=True, exist_ok=True)

total = 0
errors = []

for split in ["train", "validation"]:
    manifest_path = BASE / f"{split}.csv"
    df = pd.read_csv(manifest_path).fillna("")
    new_paths = []
    split_out = OUT / split
    split_out.mkdir(parents=True, exist_ok=True)

    print(f"\nProcessing {split}: {len(df)} clips")

    for i, source in enumerate(df["audio_path"].astype(str), start=1):
        src = Path(source)
        dst = split_out / f"{i:06d}_{src.stem}.wav"

        try:
            audio, sr = sf.read(str(src), dtype="float32", always_2d=True)

            if audio.shape[0] == 0:
                raise ValueError("Audio contains zero frames")

            # Convert multichannel audio to mono.
            audio = audio.mean(axis=1)

            # Resample using a rational polyphase filter.
            if sr != 16000:
                common = gcd(int(sr), 16000)
                up = 16000 // common
                down = int(sr) // common
                audio = resample_poly(audio, up, down).astype(np.float32)

            if len(audio) == 0 or not np.isfinite(audio).all():
                raise ValueError("Empty or non-finite resampled audio")

            sf.write(str(dst), audio, 16000, subtype="PCM_16")

            # Verify the saved output.
            info = sf.info(str(dst))
            if info.samplerate != 16000 or info.channels != 1 or info.frames == 0:
                raise ValueError("Output verification failed")

            new_paths.append(str(dst.resolve()))
            total += 1

        except Exception as e:
            errors.append({
                "split": split,
                "source_path": source,
                "error": str(e)
            })
            new_paths.append("")

        if i % 500 == 0 or i == len(df):
            print(f"  {i}/{len(df)} processed")

    result = df.copy()
    result["audio_path"] = new_paths

    # Save only if every clip in this split converted successfully.
    split_errors = [e for e in errors if e["split"] == split]
    if split_errors:
        print(f"WARNING: {len(split_errors)} conversion errors in {split}.")
        print("Manifest not saved for this split.")
    else:
        out_manifest = BASE / f"{split}_16k.csv"
        result.to_csv(out_manifest, index=False)
        print("Saved manifest:", out_manifest)

if errors:
    error_path = BASE / "resampling_errors.csv"
    pd.DataFrame(errors).to_csv(error_path, index=False)
    print("\nConversion errors:", len(errors))
    print("Error report:", error_path)
else:
    print("\nSUCCESS: All clips converted and verified.")
    print("Converted files:", total)
    print("Output folder:", OUT)
    print("New manifests: train_16k.csv and validation_16k.csv")
