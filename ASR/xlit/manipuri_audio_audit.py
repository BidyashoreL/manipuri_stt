import argparse, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from silero_vad import load_silero_vad, get_speech_timestamps

SR = 16000
PAD = int(0.20 * SR)
MAX_GAP = int(1.00 * SR)

def load_audio(path):
    x, original_sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if len(x) and original_sr != SR:
        g = math.gcd(int(original_sr), SR)
        x = resample_poly(x, SR // g, int(original_sr) // g).astype(np.float32)
    return x.astype(np.float32), int(original_sr)

def intervals_for(x, model):
    if not len(x):
        return []
    ts = get_speech_timestamps(
        torch.from_numpy(x), model, sampling_rate=SR, threshold=0.5,
        min_speech_duration_ms=100, min_silence_duration_ms=300,
        speech_pad_ms=0, return_seconds=False
    )
    result = []
    for t in ts:
        a, b = int(t["start"]), int(t["end"])
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result

def clean_audio(x, intervals):
    if not intervals:
        return x
    pieces, previous_end = [], None
    for a, b in intervals:
        a, b = max(0, a-PAD), min(len(x), b+PAD)
        if previous_end is not None and a > previous_end:
            gap = x[previous_end:a]
            pieces.append(gap[:MAX_GAP])
        pieces.append(x[a:b])
        previous_end = b
    return np.concatenate(pieces).astype(np.float32) if pieces else x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", default="combined_dataset")
    ap.add_argument("--output-dir", default="audio_audit")
    ap.add_argument("--trim", action="store_true",
                    help="Create separate cleaned WAV copies and cleaned manifests")
    args = ap.parse_args()
    src, out = Path(args.dataset_dir).resolve(), Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = ["train.csv", "validation.csv", "test.csv"]
    for n in names:
        if not (src/n).is_file():
            raise FileNotFoundError(f"Missing manifest: {src/n}")

    print("Loading Silero VAD...")
    model = load_silero_vad()
    model.eval()
    print("Dataset:", src)
    print("Mode:", "AUDIT + CLEANED COPIES" if args.trim else "AUDIT ONLY")
    reports = []

    for name in names:
        split = Path(name).stem
        df = pd.read_csv(src/name)
        for col in ("audio_path", "text", "source"):
            if col not in df.columns:
                raise ValueError(f"{name} is missing required column: {col}")
        rows, new_paths = [], []
        print(f"\nProcessing {split}: {len(df)} rows")
        for i, row in df.iterrows():
            p = Path(str(row["audio_path"]))
            base = {"row_index": i, "audio_path": str(p),
                    "source": row.get("source", ""), "text": row.get("text", "")}
            if not p.is_file():
                rows.append({**base, "error": "audio_file_missing"})
                new_paths.append(str(p))
                continue
            try:
                x, original_sr = load_audio(p)
                duration = len(x)/SR
                segs = intervals_for(x, model)
                speech = sum(b-a for a,b in segs)/SR
                ratio = speech/duration if duration else 0.0
                if segs:
                    lead = segs[0][0]/SR
                    tail = (len(x)-segs[-1][1])/SR
                    longest_gap = max((segs[j+1][0]-segs[j][1])/SR
                                      for j in range(len(segs)-1)) if len(segs)>1 else 0.0
                else:
                    lead = tail = longest_gap = duration
                flags = []
                if not segs: flags.append("no_speech_detected")
                if duration >= 1 and ratio < .10: flags.append("very_low_speech_ratio")
                if lead > 1: flags.append("long_leading_silence")
                if tail > 1: flags.append("long_trailing_silence")
                if longest_gap > 3: flags.append("long_internal_pause")
                if duration > 30: flags.append("long_clip_review")
                if duration < .25: flags.append("very_short")
                rows.append({**base, "error": "", "duration_sec": round(duration,3),
                    "original_sample_rate": original_sr, "speech_sec_vad": round(speech,3),
                    "active_speech_ratio": round(ratio,4), "leading_silence_sec": round(lead,3),
                    "trailing_silence_sec": round(tail,3),
                    "longest_internal_gap_sec": round(longest_gap,3),
                    "vad_segments": len(segs), "flags": ";".join(flags)})
                if args.trim:
                    cleaned = clean_audio(x, segs)
                    clean_dir = out/"cleaned_audio"/split
                    clean_dir.mkdir(parents=True, exist_ok=True)
                    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", p.stem)
                    cp = (clean_dir/f"{i:06d}_{stem}.wav").resolve()
                    sf.write(str(cp), cleaned, SR, subtype="PCM_16")
                    new_paths.append(str(cp))
                else:
                    new_paths.append(str(p))
            except Exception as e:
                rows.append({**base, "error": f"{type(e).__name__}: {e}"})
                new_paths.append(str(p))
                print(f"Error on row {i}: {e}")
            if (i+1) % 500 == 0: print(f"  {i+1}/{len(df)}")
        report = pd.DataFrame(rows)
        rp = out/f"{split}_silence_report.csv"
        report.to_csv(rp, index=False, encoding="utf-8-sig")
        reports.append(report.assign(split=split))
        print("Saved report:", rp)
        if args.trim:
            cleaned_df = df.copy()
            cleaned_df["audio_path"] = new_paths
            cp = out/f"{split}_cleaned.csv"
            cleaned_df.to_csv(cp, index=False, encoding="utf-8-sig")
            print("Saved cleaned manifest:", cp)
        if "active_speech_ratio" in report:
            valid = report["active_speech_ratio"].dropna()
            if len(valid):
                flags = report["flags"].fillna("")
                print(f"  Median active-speech ratio: {valid.median():.3f}")
                print(f"  No speech detected: {flags.str.contains('no_speech_detected').sum()}")
                print(f"  Long internal pauses: {flags.str.contains('long_internal_pause').sum()}")
    all_df = pd.concat(reports, ignore_index=True)
    all_path = out/"all_splits_silence_report.csv"
    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    print("\nSaved combined report:", all_path)
    print("Original files and manifests were not modified.")
    if not args.trim:
        print("Audit only. Review reports before rerunning with --trim.")

if __name__ == "__main__":
    main()
