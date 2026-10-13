from pathlib import Path
import io
import wave
import json
import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio


ROOT = Path.cwd()
ASR = ROOT / "ASR"

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"

CACHE_DIR = ASR / "audio_cache"

SAMPLE_RATE = 16000


def decode_pcm(raw, sample_width):

    if sample_width == 1:

        audio = np.frombuffer(
            raw,
            dtype=np.uint8,
        ).astype(np.float32)

        audio = (audio - 128.0) / 128.0

    elif sample_width == 2:

        audio = np.frombuffer(
            raw,
            dtype=np.int16,
        ).astype(np.float32) / 32768.0

    elif sample_width == 3:

        raw_bytes = np.frombuffer(
            raw,
            dtype=np.uint8,
        )

        if len(raw_bytes) % 3 != 0:
            raise ValueError(
                "Invalid 24-bit PCM byte length"
            )

        raw_bytes = raw_bytes.reshape(-1, 3)

        audio = (
            raw_bytes[:, 0].astype(np.int32)
            | (raw_bytes[:, 1].astype(np.int32) << 8)
            | (raw_bytes[:, 2].astype(np.int32) << 16)
        )

        mask = audio & 0x800000
        audio = audio - (mask << 1)

        audio = (
            audio.astype(np.float32)
            / 8388608.0
        )

    elif sample_width == 4:

        audio = np.frombuffer(
            raw,
            dtype=np.int32,
        ).astype(np.float32) / 2147483648.0

    else:

        raise ValueError(
            f"Unsupported sample width: {sample_width}"
        )

    return audio


def load_parquet_audio(
    parquet_file,
    row_index,
):

    df = pd.read_parquet(
        parquet_file,
        columns=["audio"],
    )

    audio_obj = df.iloc[int(row_index)]["audio"]

    if not isinstance(audio_obj, dict):
        raise ValueError(
            f"Unexpected audio type: {type(audio_obj)}"
        )

    audio_bytes = audio_obj.get("bytes")

    if audio_bytes is None:
        raise ValueError(
            "Audio object has no bytes"
        )

    with wave.open(
        io.BytesIO(audio_bytes),
        "rb",
    ) as wf:

        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()

        raw = wf.readframes(frames)

    audio = decode_pcm(
        raw,
        sample_width,
    )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels,
        )

        audio = audio.mean(axis=1)

    return audio, sample_rate


def normalize_audio(
    audio,
    sample_rate,
):

    audio = torch.as_tensor(
        audio,
        dtype=torch.float32,
    )

    if audio.ndim == 2:

        if audio.shape[0] <= 8:

            audio = audio.mean(dim=0)

        else:

            audio = audio.mean(dim=1)

    elif audio.ndim > 2:

        audio = audio.squeeze()

    if sample_rate != SAMPLE_RATE:

        audio = torchaudio.functional.resample(
            audio,
            sample_rate,
            SAMPLE_RATE,
        )

    audio = audio.contiguous()

    return audio


def resolve_audio(row):

    source = str(
        row.get("source", "")
    ).strip().lower()

    if source == "indicvoices":

        audio, sr = load_parquet_audio(
            str(row["parquet_file"]),
            int(row["parquet_row"]),
        )

        return normalize_audio(
            audio,
            sr,
        )

    audio_path = row.get(
        "audio_path",
        "",
    )

    if pd.isna(audio_path):
        raise RuntimeError(
            "audio_path is NaN"
        )

    audio_path = str(
        audio_path
    ).strip()

    if not audio_path:
        raise RuntimeError(
            "audio_path is empty"
        )

    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            audio_path
        )

    array, sr = sf.read(
        audio_path,
        dtype="float32",
        always_2d=False,
    )

    return normalize_audio(
        array,
        sr,
    )


def process_manifest(csv_path):

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
    )

    print(
        f"\n[DATASET] {csv_path}"
    )

    print(
        f"[DATASET] samples={len(df):,}"
    )

    for idx, row in df.iterrows():

        out = CACHE_DIR / f"{int(idx):06d}.pt"

        if out.exists():

            if idx % 500 == 0:
                print(
                    f"[CACHE] existing {idx:,}/{len(df):,}",
                    flush=True,
                )

            continue

        try:

            audio = resolve_audio(row)

            if audio.numel() == 0:
                raise RuntimeError(
                    "Empty audio"
                )

            if not torch.isfinite(audio).all():
                raise RuntimeError(
                    "Audio contains NaN/Inf"
                )

            payload = {
                "audio": audio,
                "sample_rate": SAMPLE_RATE,
                "source": str(
                    row.get("source", "")
                ),
                "original_index": int(idx),
            }

            tmp = out.with_suffix(".tmp")

            torch.save(
                payload,
                tmp,
            )

            tmp.replace(out)

            if idx % 100 == 0:

                print(
                    f"[CACHE] {idx:,}/{len(df):,} "
                    f"frames={audio.numel():,}",
                    flush=True,
                )

        except Exception as e:

            print(
                f"[ERROR] index={idx}",
                flush=True,
            )

            print(
                f"        source={row.get('source')}",
                flush=True,
            )

            print(
                f"        error={type(e).__name__}: {e}",
                flush=True,
            )

            raise


def main():

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("BUILDING 16-kHz AUDIO CACHE")
    print("=" * 72)

    process_manifest(
        TRAIN_CSV
    )

    process_manifest(
        VAL_CSV
    )

    metadata = {
        "sample_rate": SAMPLE_RATE,
        "train_csv": str(TRAIN_CSV),
        "val_csv": str(VAL_CSV),
    }

    (
        CACHE_DIR / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    count = len(
        list(
            CACHE_DIR.glob("*.pt")
        )
    )

    print()
    print("=" * 72)
    print(
        f"AUDIO CACHE COMPLETE: {count:,} files"
    )
    print(
        "CACHE:",
        CACHE_DIR,
    )
    print("=" * 72)


if __name__ == "__main__":
    main()