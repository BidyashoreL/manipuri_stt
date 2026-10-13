from pathlib import Path
import math
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, get_speech_timestamps

SR = 16000
PAD = int(0.25 * SR)
MIN_REMAINING_SEC = 0.30

ROOT = Path.cwd()
MANIFEST_DIR = ROOT / "combined_dataset"
OUTPUT_DIR = ROOT / "audio_audit" / "safe_cleaned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_audio(path):
    x, original_sr = sf.read(
        str(path), dtype="float32", always_2d=True
    )
    x = x.mean(axis=1)
    if len(x) and original_sr != SR:
        g = math.gcd(int(original_sr), SR)
        x = resample_poly(
            x, SR // g, int(original_sr) // g
        ).astype(np.float32)
    return x.astype(np.float32), int(original_sr)

def get_segments(x, model):
    if len(x) == 0:
        return []
    timestamps = get_speech_timestamps(
        torch.from_numpy(x),
        model,
        sampling_rate=SR,
        threshold=0.5,
        min_speech_duration_ms=100,
        min_silence_duration_ms=300,
        speech_pad_ms=0,
        return_seconds=False,
    )
    return [(int(t["start"]), int(t["end"])) for t in timestamps]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

print("Loading Silero VAD...")
model = load_silero_vad()
model.eval()

all_reports = []

for split in ("train", "validation", "test"):
    manifest = MANIFEST_DIR / f"{split}.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")

    df = pd.read_csv(manifest)
    if "audio_path" not in df.columns or "text" not in df.columns:
        raise ValueError(f"{manifest} must contain audio_path and text")

    split_out = OUTPUT_DIR / "audio" / split
    split_out.mkdir(parents=True, exist_ok=True)

    new_paths, rows = [], []
    print(f"\nProcessing {split}: {len(df)} entries")

    for i, row in df.iterrows():
        original_path = Path(str(row["audio_path"]))
        action = "unchanged"
        error = ""
        before_duration = 0.0
        after_duration = 0.0
        lead_trim = 0.0
        tail_trim = 0.0

        try:
            if not original_path.is_file():
                raise FileNotFoundError(str(original_path))

            x, original_sr = load_audio(original_path)
            before_duration = len(x) / SR
            segments = get_segments(x, model)

            # Never auto-trim clips when speech wasn't detected.
            if not segments:
                cleaned = x
                action = "manual_review_no_speech"
            else:
                first_start = segments[0][0]
                last_end = segments[-1][1]

                start = max(0, first_start - PAD)
                end = min(len(x), last_end + PAD)

                # Only trim outer silence; never remove internal gaps.
                candidate = x[start:end]

                # Reject a proposed trim that would leave too little audio.
                if len(candidate) / SR < MIN_REMAINING_SEC:
                    cleaned = x
                    action = "manual_review_short_after_trim"
                elif start == 0 and end == len(x):
                    cleaned = x
                    action = "unchanged_no_outer_trim"
                else:
                    cleaned = candidate
                    action = "trimmed_outer_silence"
                    lead_trim = start / SR
                    tail_trim = (len(x) - end) / SR

            after_duration = len(cleaned) / SR

            out_path = (
                split_out / f"{i:06d}_{original_path.stem}.wav"
            ).resolve()

            # Keep all entries represented in the new manifest.
            sf.write(str(out_path), cleaned, SR, subtype="PCM_16")
            new_paths.append(str(out_path))

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            action = "error_original_path_retained"
            new_paths.append(str(original_path))

        rows.append({
            "row_index": i,
            "original_path": str(original_path),
            "cleaned_path": new_paths[-1],
            "source": row.get("source", ""),
            "split": split,
            "text": row.get("text", ""),
            "duration_before_sec": round(before_duration, 3),
            "duration_after_sec": round(after_duration, 3),
            "leading_trim_sec": round(lead_trim, 3),
            "trailing_trim_sec": round(tail_trim, 3),
            "action": action,
            "error": error,
        })

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(df)}")

    cleaned_manifest = df.copy()
    cleaned_manifest["audio_path"] = new_paths
    cleaned_manifest.to_csv(
        OUTPUT_DIR / f"{split}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    report = pd.DataFrame(rows)
    report.to_csv(
        OUTPUT_DIR / f"{split}_before_after.csv",
        index=False,
        encoding="utf-8-sig",
    )
    all_reports.append(report)

    print("Actions:")
    print(report["action"].value_counts().to_string())
    print("Saved manifest:", OUTPUT_DIR / f"{split}.csv")

combined = pd.concat(all_reports, ignore_index=True)
combined.to_csv(
    OUTPUT_DIR / "all_splits_before_after.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\n========== SAFE CLEANING SUMMARY ==========")
print("Entries:", len(combined))
print(combined["action"].value_counts().to_string())
print("Errors:", int((combined["error"] != "").sum()))
print("Output:", OUTPUT_DIR.resolve())
print("Original audio and manifests were not modified.")
