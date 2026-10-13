from pathlib import Path
import io
import json
import wave

import pandas as pd
import torch
import torchaudio
import soundfile as sf

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent

VAL_CSV = ASR_DIR / "combined_dataset" / "val.csv"
CACHE_DIR = ASR_DIR / "val_audio_cache"

TARGET_SR = 16000


def decode_pcm(audio_bytes, sample_width, channels, sample_rate):
    if sample_width == 1:
        import numpy as np
        data = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        import numpy as np
        data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        import numpy as np
        raw = np.frombuffer(audio_bytes, dtype=np.uint8)
        raw = raw.reshape(-1, 3)
        vals = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        vals[vals & 0x800000 != 0] -= 1 << 24
        data = vals.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        import numpy as np
        data = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    audio = torch.from_numpy(data)

    if channels > 1:
        audio = audio.view(-1, channels).mean(dim=1)

    audio = audio.float()

    if sample_rate != TARGET_SR:
        audio = torchaudio.functional.resample(
            audio,
            sample_rate,
            TARGET_SR,
        )

    return audio.contiguous()


def load_parquet_audio(row):
    parquet_file = str(row["parquet_file"])
    parquet_row = int(row["parquet_row"])

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_file)
    audio_obj = table.column("audio")[parquet_row].as_py()

    if isinstance(audio_obj, dict):
        audio_bytes = audio_obj.get("bytes")
        if audio_bytes is None:
            raise ValueError("Audio bytes missing")

        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            return decode_pcm(
                frames,
                wf.getsampwidth(),
                wf.getnchannels(),
                wf.getframerate(),
            )

    raise ValueError("Unsupported parquet audio object")


def load_audio(row):
    source = str(row["source"])

    if source == "indicvoices":
        return load_parquet_audio(row)

    audio_path = Path(str(row["audio_path"]))

    audio, sr = sf.read(str(audio_path), dtype="float32")

    audio = torch.from_numpy(audio)

    if audio.ndim > 1:
        audio = audio.mean(dim=1)

    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(
            audio,
            sr,
            TARGET_SR,
        )

    return audio.contiguous()


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(VAL_CSV)

    print("=" * 70)
    print("BUILDING VALIDATION AUDIO CACHE")
    print("=" * 70)
    print(f"[VAL] Samples: {len(df)}")
    print(f"[CACHE] {CACHE_DIR}")

    success = 0
    failed = 0

    failures = []

    for idx, row in df.iterrows():
        out = CACHE_DIR / f"{idx:06d}.pt"

        if out.exists():
            success += 1
            continue

        try:
            audio = load_audio(row)

            torch.save(
                {
                    "audio": audio,
                    "val_index": int(idx),
                },
                out,
            )

            success += 1

        except Exception as e:
            failed += 1
            failures.append(
                {
                    "index": int(idx),
                    "error": repr(e),
                    "audio_path": str(row.get("audio_path", "")),
                }
            )

        if (idx + 1) % 100 == 0:
            print(
                f"[{idx + 1}/{len(df)}] "
                f"success={success} failed={failed}"
            )

    metadata = {
        "samples": len(df),
        "success": success,
        "failed": failed,
        "sample_rate": TARGET_SR,
    }

    (CACHE_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if failures:
        (CACHE_DIR / "failures.json").write_text(
            json.dumps(failures, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print()
    print("=" * 70)
    print("VALIDATION CACHE COMPLETE")
    print("=" * 70)
    print(f"Samples : {len(df)}")
    print(f"Success : {success}")
    print(f"Failed  : {failed}")
    print("=" * 70)


if __name__ == "__main__":
    main()