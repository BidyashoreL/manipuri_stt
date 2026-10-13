import os
import io
import csv
import wave
import time
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

# ============================================================
# CONFIG
# ============================================================

ROOT = os.getcwd()

MANIFEST = os.path.join(
    ROOT,
    "ASR",
    "combined_dataset",
    "all.csv"
)

CACHE_ROOT = os.path.join(
    ROOT,
    "ASR",
    "feature_cache"
)

ERROR_FILE = os.path.join(
    CACHE_ROOT,
    "errors.csv"
)

W2V_PATH = r"E:\hf_models\w2v-bert-2.0"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_SR = 16000
HIDDEN_SIZE = 1024

# ============================================================
# SETUP
# ============================================================

os.makedirs(CACHE_ROOT, exist_ok=True)

for source in [
    "indicvoices",
    "1k_project",
    "benchmark"
]:
    os.makedirs(
        os.path.join(CACHE_ROOT, source),
        exist_ok=True
    )

print("=" * 75)
print("PRODUCTION W2V-BERT FEATURE CACHE")
print("=" * 75)

print("Device:", DEVICE)
print("Manifest:", MANIFEST)
print("Cache:", CACHE_ROOT)

# ============================================================
# LOAD MODEL
# ============================================================

print("\n[1/6] Loading W2V-BERT...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    W2V_PATH,
    local_files_only=True
)

encoder = Wav2Vec2BertModel.from_pretrained(
    W2V_PATH,
    dtype=torch.float32,
    local_files_only=True
)

encoder.to(DEVICE)
encoder.eval()

for p in encoder.parameters():
    p.requires_grad_(False)

print("W2V-BERT loaded.")

# ============================================================
# LOAD MANIFEST
# ============================================================

print("\n[2/6] Loading manifest...")

df = pd.read_csv(MANIFEST)

print("Total rows:", len(df))

print("\nSources:")
print(df["source"].value_counts())

# ============================================================
# WAV DECODER
# ============================================================

def decode_wav_bytes(audio_bytes):

    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:

        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()

        raw = wf.readframes(n_frames)

    if sample_width == 2:

        audio = np.frombuffer(
            raw,
            dtype=np.int16
        ).astype(np.float32)

        audio /= 32768.0

    elif sample_width == 4:

        audio = np.frombuffer(
            raw,
            dtype=np.int32
        ).astype(np.float32)

        audio /= 2147483648.0

    else:

        raise RuntimeError(
            f"Unsupported sample width: {sample_width}"
        )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        )

        audio = audio.mean(axis=1)

    return torch.from_numpy(
        audio.copy()
    ), sample_rate


# ============================================================
# AUDIO RESOLUTION
# ============================================================

def decode_audio(row):

    source = str(row["source"])

    # --------------------------------------------------------
    # 1K / BENCHMARK
    # --------------------------------------------------------

    if source in [
        "1k_project",
        "benchmark"
    ]:

        path = str(row["audio_path"])

        if not os.path.exists(path):

            raise FileNotFoundError(
                path
            )

        with open(path, "rb") as f:

            audio_bytes = f.read()

        return decode_wav_bytes(
            audio_bytes
        )

    # --------------------------------------------------------
    # INDICVOICES
    # --------------------------------------------------------

    if source == "indicvoices":

        parquet_file = str(
            row["parquet_file"]
        )

        parquet_row = int(
            row["parquet_row"]
        )

        if not os.path.exists(
            parquet_file
        ):

            raise FileNotFoundError(
                parquet_file
            )

        table = pd.read_parquet(
            parquet_file,
            engine="pyarrow"
        )

        audio_obj = table.iloc[
            parquet_row
        ]["audio"]

        if not isinstance(
            audio_obj,
            dict
        ):

            raise RuntimeError(
                f"Unexpected audio object type: "
                f"{type(audio_obj)}"
            )

        audio_bytes = audio_obj["bytes"]

        if audio_bytes is None:

            raise RuntimeError(
                "Audio bytes are None"
            )

        return decode_wav_bytes(
            audio_bytes
        )

    raise RuntimeError(
        f"Unknown source: {source}"
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_audio(
    waveform,
    sr
):

    waveform = waveform.float()

    if waveform.ndim != 1:

        raise RuntimeError(
            f"Expected mono waveform, "
            f"got {waveform.shape}"
        )

    waveform_np = waveform.numpy()

    original_sr = sr

    if sr != TARGET_SR:

        gcd = np.gcd(
            sr,
            TARGET_SR
        )

        up = TARGET_SR // gcd
        down = sr // gcd

        waveform_np = resample_poly(
            waveform_np,
            up,
            down
        ).astype(np.float32)

        sr = TARGET_SR

    return (
        torch.from_numpy(
            waveform_np.astype(
                np.float32
            )
        ),
        original_sr,
        sr
    )


# ============================================================
# CACHE PATH
# ============================================================

def cache_path(
    source,
    manifest_index
):

    return os.path.join(
        CACHE_ROOT,
        source,
        f"{manifest_index:06d}.pt"
    )


# ============================================================
# ERROR LOG
# ============================================================

def write_error(
    manifest_index,
    source,
    error
):

    exists = os.path.exists(
        ERROR_FILE
    )

    with open(
        ERROR_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        if not exists:

            writer.writerow([
                "manifest_index",
                "source",
                "error"
            ])

        writer.writerow([
            manifest_index,
            source,
            str(error)
        ])


# ============================================================
# BUILD CACHE
# ============================================================

print("\n[3/6] Starting feature extraction...")

total = len(df)

completed = 0
skipped = 0
failed = 0

start_time = time.time()

for manifest_index, row in df.iterrows():

    source = str(row["source"])

    output_file = cache_path(
        source,
        manifest_index
    )

    # --------------------------------------------------------
    # RESUME SUPPORT
    # --------------------------------------------------------

    if os.path.exists(
        output_file
    ):

        skipped += 1

        if (
            (skipped + completed + failed) % 100
            == 0
        ):

            print(
                f"[RESUME] "
                f"{skipped + completed + failed}/{total}"
            )

        continue

    try:

        # ----------------------------------------------------
        # AUDIO
        # ----------------------------------------------------

        waveform, sr = decode_audio(
            row
        )

        waveform, original_sr, final_sr = (
            normalize_audio(
                waveform,
                sr
            )
        )

        # ----------------------------------------------------
        # W2V-BERT FEATURE EXTRACTION
        # ----------------------------------------------------

        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        )

        input_features = (
            inputs.input_features
            .to(
                DEVICE,
                dtype=torch.float32
            )
        )

        with torch.inference_mode():

            outputs = encoder(
                input_features=input_features
            )

            features = (
                outputs
                .last_hidden_state
                .float()
                .squeeze(0)
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if features.ndim != 2:

            raise RuntimeError(
                f"Expected (T,1024), "
                f"got {features.shape}"
            )

        if features.shape[1] != HIDDEN_SIZE:

            raise RuntimeError(
                f"Expected hidden size "
                f"{HIDDEN_SIZE}, "
                f"got {features.shape[1]}"
            )

        if not torch.isfinite(
            features
        ).all():

            raise RuntimeError(
                "NaN/Inf detected"
            )

        # ----------------------------------------------------
        # SAVE AS FLOAT16
        # ----------------------------------------------------

        payload = {
            "features": features.cpu().half(),
            "manifest_index": int(
                manifest_index
            ),
            "source": source,
        }

        torch.save(
            payload,
            output_file
        )

        completed += 1

    except Exception as e:

        failed += 1

        write_error(
            manifest_index,
            source,
            e
        )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    processed = (
        completed
        + skipped
        + failed
    )

    if (
        processed % 25 == 0
        or processed == total
    ):

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            processed / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total - processed
        )

        eta = (
            remaining / rate
            if rate > 0
            else 0
        )

        print(
            f"[{processed:5d}/{total}] "
            f"{processed / total * 100:6.2f}% | "
            f"new={completed} "
            f"skip={skipped} "
            f"fail={failed} | "
            f"{rate:.2f}/s | "
            f"ETA={eta/60:.1f} min"
        )

# ============================================================
# FINAL REPORT
# ============================================================

print("\n[4/6] Cache generation finished.")

print("\n[5/6] Final verification...")

actual_files = 0

for source in [
    "indicvoices",
    "1k_project",
    "benchmark"
]:

    source_dir = os.path.join(
        CACHE_ROOT,
        source
    )

    count = len([
        f for f in os.listdir(
            source_dir
        )
        if f.endswith(".pt")
    ])

    actual_files += count

    print(
        f"{source:15s}: {count}"
    )

print(
    "\nTotal cache files:",
    actual_files
)

print(
    "Expected:",
    total
)

print(
    "Newly generated:",
    completed
)

print(
    "Skipped existing:",
    skipped
)

print(
    "Failed:",
    failed
)

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "manifest": MANIFEST,
    "total_manifest_rows": int(total),
    "cache_files": int(actual_files),
    "newly_generated": int(completed),
    "skipped_existing": int(skipped),
    "failed": int(failed),
    "hidden_size": HIDDEN_SIZE,
    "dtype": "float16",
    "sample_rate": TARGET_SR,
    "sources": {
        source: int(
            (df["source"] == source).sum()
        )
        for source in [
            "indicvoices",
            "1k_project",
            "benchmark"
        ]
    }
}

summary_file = os.path.join(
    CACHE_ROOT,
    "metadata.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    import json

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2
    )

print("\n[6/6] Summary saved:")
print(summary_file)

print("\n" + "=" * 75)
print("PRODUCTION FEATURE CACHE COMPLETE")
print("=" * 75)
