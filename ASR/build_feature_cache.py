import os
import io
import json
import time
import wave
import traceback

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel


# ============================================================
# CONFIG
# ============================================================

ROOT = r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt"

MODEL_PATH = r"E:\hf_models\w2v-bert-2.0"

COMBINED_DIR = os.path.join(ROOT, "ASR", "combined_dataset")

TRAIN_CSV = os.path.join(COMBINED_DIR, "train.csv")
VAL_CSV = os.path.join(COMBINED_DIR, "val.csv")

CACHE_ROOT = os.path.join(COMBINED_DIR, "feature_cache")

TRAIN_CACHE = os.path.join(CACHE_ROOT, "train")
VAL_CACHE = os.path.join(CACHE_ROOT, "val")

PROGRESS_FILE = os.path.join(CACHE_ROOT, "progress.json")

TARGET_SR = 16000

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Keep this conservative.
# W2V-BERT is run one sample at a time.
MAX_AUDIO_SECONDS = 30.0


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(TRAIN_CACHE, exist_ok=True)
os.makedirs(VAL_CACHE, exist_ok=True)


# ============================================================
# PRINT CONFIG
# ============================================================

print("=" * 75)
print("MANIPURI W2V-BERT FEATURE CACHE")
print("=" * 75)

print("ROOT:")
print(ROOT)

print("\nMODEL:")
print(MODEL_PATH)

print("\nDEVICE:")
print(DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2,
        ),
        "GB",
    )

print("\nCACHE:")
print(CACHE_ROOT)


# ============================================================
# LOAD W2V-BERT
# ============================================================

print("\n" + "=" * 75)
print("[1/6] Loading W2V-BERT")
print("=" * 75)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

encoder = Wav2Vec2BertModel.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True,
)

encoder.to(DEVICE)
encoder.eval()

for param in encoder.parameters():
    param.requires_grad_(False)

print("W2V-BERT loaded successfully.")


# ============================================================
# AUDIO LOADERS
# ============================================================

def decode_pcm(raw, sample_width):

    if sample_width == 2:

        audio = np.frombuffer(
            raw,
            dtype=np.int16,
        ).astype(np.float32)

        audio /= 32768.0

    elif sample_width == 4:

        audio = np.frombuffer(
            raw,
            dtype=np.int32,
        ).astype(np.float32)

        audio /= 2147483648.0

    else:

        raise ValueError(
            f"Unsupported PCM sample width: {sample_width}"
        )

    return audio


def load_wav(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"WAV does not exist: {path}"
        )

    with wave.open(path, "rb") as wf:

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


def load_parquet_audio(parquet_file, row_index):

    if not os.path.exists(parquet_file):

        raise FileNotFoundError(
            f"Parquet does not exist: {parquet_file}"
        )

    df = pd.read_parquet(
        parquet_file,
        columns=["audio"],
    )

    audio_obj = df.iloc[int(row_index)]["audio"]

    if not isinstance(audio_obj, dict):

        raise ValueError(
            f"Unexpected audio object type: {type(audio_obj)}"
        )

    audio_bytes = audio_obj.get("bytes")

    if audio_bytes is None:

        raise ValueError(
            "IndicVoices audio object has no bytes"
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


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_audio(audio, sample_rate):

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # Mono
    if audio.ndim > 1:

        audio = audio.mean(axis=-1)

    # Resample
    if sample_rate != TARGET_SR:

        gcd = np.gcd(
            int(sample_rate),
            TARGET_SR,
        )

        up = TARGET_SR // gcd
        down = int(sample_rate) // gcd

        audio = resample_poly(
            audio,
            up,
            down,
        ).astype(np.float32)

    return audio


# ============================================================
# RESOLVE MANIFEST ROW
# ============================================================

def resolve_audio(row):

    source = str(
        row["source"]
    ).strip()

    # --------------------------------------------------------
    # IndicVoices
    # --------------------------------------------------------

    if source == "indicvoices":

        parquet_file = str(
            row["parquet_file"]
        )

        row_index = int(
            row["parquet_row"]
        )

        audio, sr = load_parquet_audio(
            parquet_file,
            row_index,
        )

        return normalize_audio(
            audio,
            sr,
        )

    # --------------------------------------------------------
    # 1K project
    # --------------------------------------------------------

    if source == "1k_project":

        path = str(
            row["audio_path"]
        )

        if not os.path.isabs(path):

            path = os.path.join(
                ROOT,
                path,
            )

        audio, sr = load_wav(path)

        return normalize_audio(
            audio,
            sr,
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    if source == "benchmark":

        path = str(
            row["audio_path"]
        )

        if not os.path.isabs(path):

            path = os.path.join(
                ROOT,
                path,
            )

        audio, sr = load_wav(path)

        return normalize_audio(
            audio,
            sr,
        )

    raise ValueError(
        f"Unknown source: {source}"
    )


# ============================================================
# W2V-BERT ENCODING
# ============================================================

@torch.inference_mode()
def encode_audio(audio):

    duration = len(audio) / TARGET_SR

    if duration <= 0:

        raise ValueError(
            "Audio is empty"
        )

    if duration > MAX_AUDIO_SECONDS:

        raise ValueError(
            f"Audio exceeds {MAX_AUDIO_SECONDS}s: "
            f"{duration:.2f}s"
        )

    inputs = feature_extractor(
        audio,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )

    input_features = inputs[
        "input_features"
    ].to(DEVICE)

    attention_mask = inputs.get(
        "attention_mask"
    )

    if attention_mask is not None:

        attention_mask = attention_mask.to(
            DEVICE
        )

    outputs = encoder(
        input_features=input_features,
        attention_mask=attention_mask,
    )

    features = outputs.last_hidden_state[
        0
    ].float().cpu()

    return features


# ============================================================
# PROGRESS
# ============================================================

def load_progress():

    if not os.path.exists(PROGRESS_FILE):

        return {
            "train_completed": 0,
            "val_completed": 0,
            "train_errors": [],
            "val_errors": [],
        }

    try:

        with open(
            PROGRESS_FILE,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    except Exception:

        return {
            "train_completed": 0,
            "val_completed": 0,
            "train_errors": [],
            "val_errors": [],
        }


def save_progress(progress):

    temp = PROGRESS_FILE + ".tmp"

    with open(
        temp,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            progress,
            f,
            indent=2,
        )

    os.replace(
        temp,
        PROGRESS_FILE,
    )


# ============================================================
# SAVE FEATURE
# ============================================================

def save_feature(
    feature,
    output_path,
):

    temp_path = output_path + ".tmp"

    torch.save(
        feature,
        temp_path,
    )

    os.replace(
        temp_path,
        output_path,
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    df,
    split_name,
    cache_dir,
    progress,
):

    total = len(df)

    if split_name == "train":

        completed = int(
            progress["train_completed"]
        )

        error_key = "train_errors"

    else:

        completed = int(
            progress["val_completed"]
        )

        error_key = "val_errors"

    print("\n" + "=" * 75)
    print(
        f"[{split_name.upper()}] "
        f"{completed}/{total} already completed"
    )
    print("=" * 75)

    start_time = time.time()

    for position in range(
        completed,
        total,
    ):

        row = df.iloc[position]

        # The filename uses the manifest position.
        output_path = os.path.join(
            cache_dir,
            f"{position:08d}.pt",
        )

        # Extra safety for interrupted runs.
        if os.path.exists(output_path):

            if split_name == "train":

                progress["train_completed"] = position + 1

            else:

                progress["val_completed"] = position + 1

            save_progress(progress)

            continue

        try:

            source = str(
                row["source"]
            )

            audio = resolve_audio(
                row
            )

            feature = encode_audio(
                audio
            )

            # ------------------------------------------------
            # Validate output
            # ------------------------------------------------

            if feature.ndim != 2:

                raise ValueError(
                    f"Expected 2D feature, got "
                    f"{feature.shape}"
                )

            if feature.shape[1] != 1024:

                raise ValueError(
                    f"Expected hidden size 1024, "
                    f"got {feature.shape}"
                )

            save_feature(
                feature,
                output_path,
            )

            if split_name == "train":

                progress[
                    "train_completed"
                ] = position + 1

            else:

                progress[
                    "val_completed"
                ] = position + 1

            save_progress(
                progress
            )

            # ------------------------------------------------
            # Progress display
            # ------------------------------------------------

            elapsed = (
                time.time()
                - start_time
            )

            current = position + 1

            rate = (
                current - completed
            ) / max(
                elapsed,
                1e-6,
            )

            remaining = (
                total - current
            )

            eta_seconds = (
                remaining / rate
                if rate > 0
                else 0
            )

            eta_minutes = (
                eta_seconds / 60
            )

            audio_seconds = (
                len(audio)
                / TARGET_SR
            )

            print(
                f"[{split_name}] "
                f"{current:5d}/{total} "
                f"({100*current/total:6.2f}%) "
                f"source={source:15s} "
                f"audio={audio_seconds:6.2f}s "
                f"feat={tuple(feature.shape)} "
                f"ETA={eta_minutes:7.1f} min"
            )

        except Exception as e:

            error_record = {
                "position": position,
                "source": str(
                    row.get(
                        "source",
                        "",
                    )
                ),
                "error": str(e),
            }

            progress[
                error_key
            ].append(
                error_record
            )

            # Move forward so one bad file
            # does not stop the entire cache.
            if split_name == "train":

                progress[
                    "train_completed"
                ] = position + 1

            else:

                progress[
                    "val_completed"
                ] = position + 1

            save_progress(
                progress
            )

            print(
                f"\nERROR at {split_name} "
                f"position {position}:"
            )

            print(
                traceback.format_exc()
            )

    print(
        f"\n{split_name.upper()} COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("[2/6] Loading manifests")
    print("=" * 75)

    train_df = pd.read_csv(
        TRAIN_CSV
    )

    val_df = pd.read_csv(
        VAL_CSV
    )

    print(
        "Train samples:",
        len(train_df),
    )

    print(
        "Validation samples:",
        len(val_df),
    )

    print("\nTrain sources:")
    print(
        train_df["source"].value_counts()
    )

    print("\nValidation sources:")
    print(
        val_df["source"].value_counts()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Preserve the exact manifest row index.
    # --------------------------------------------------------

    train_df["_cache_record_index"] = (
        train_df.index
    )

    val_df["_cache_record_index"] = (
        val_df.index
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("[3/6] Loading progress")
    print("=" * 75)

    progress = load_progress()

    print(
        "Train completed:",
        progress["train_completed"],
    )

    print(
        "Validation completed:",
        progress["val_completed"],
    )

    print(
        "Train errors:",
        len(progress["train_errors"]),
    )

    print(
        "Validation errors:",
        len(progress["val_errors"]),
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("[4/6] Processing TRAIN")
    print("=" * 75)

    process_split(
        train_df,
        "train",
        TRAIN_CACHE,
        progress,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("[5/6] Processing VALIDATION")
    print("=" * 75)

    process_split(
        val_df,
        "val",
        VAL_CACHE,
        progress,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("[6/6] Writing metadata")
    print("=" * 75)

    metadata = {
        "model": MODEL_PATH,
        "hidden_size": 1024,
        "sample_rate": TARGET_SR,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "train_completed": progress[
            "train_completed"
        ],
        "val_completed": progress[
            "val_completed"
        ],
        "train_errors": len(
            progress["train_errors"]
        ),
        "val_errors": len(
            progress["val_errors"]
        ),
        "feature_dtype": "float32",
        "feature_shape": "(T, 1024)",
    }

    metadata_path = os.path.join(
        CACHE_ROOT,
        "metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print("\n" + "=" * 75)
    print("FEATURE CACHE COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()