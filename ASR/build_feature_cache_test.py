import os
import io
import json
import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

ROOT = r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt"

MODEL_PATH = r"E:\hf_models\w2v-bert-2.0"

TRAIN_CSV = os.path.join(ROOT, "ASR", "combined_dataset", "train.csv")
VAL_CSV = os.path.join(ROOT, "ASR", "combined_dataset", "val.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("W2V-BERT FEATURE CACHE TEST")
print("=" * 70)
print("Device:", device)

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("\n[1/4] Loading W2V-BERT...")

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

encoder = Wav2Vec2BertModel.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    local_files_only=True,
)

encoder.to(device)
encoder.eval()

for p in encoder.parameters():
    p.requires_grad_(False)

print("W2V-BERT loaded.")

# ---------------------------------------------------------
# Audio loading
# ---------------------------------------------------------

def load_wav(path):
    import wave

    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        audio /= 2147483648.0
    else:
        raise ValueError(
            f"Unsupported sample width {sample_width}: {path}"
        )

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def load_parquet_audio(parquet_file, row_index):
    df = pd.read_parquet(parquet_file)

    audio_obj = df.iloc[int(row_index)]["audio"]

    if isinstance(audio_obj, dict):
        audio_bytes = audio_obj["bytes"]
    else:
        raise ValueError(
            f"Unexpected audio object type: {type(audio_obj)}"
        )

    import wave

    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        audio /= 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        audio /= 2147483648.0
    else:
        raise ValueError(
            f"Unsupported sample width: {sample_width}"
        )

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def normalize_audio(audio, sample_rate):

    audio = np.asarray(audio, dtype=np.float32)

    # Remove NaN / Inf
    audio = np.nan_to_num(audio)

    # Convert to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)

    # Resample to 16 kHz
    if sample_rate != 16000:
        gcd = np.gcd(sample_rate, 16000)

        up = 16000 // gcd
        down = sample_rate // gcd

        audio = resample_poly(
            audio,
            up,
            down,
        ).astype(np.float32)

    return audio


# ---------------------------------------------------------
# Resolve manifest row
# ---------------------------------------------------------

def resolve_audio(row):

    source = str(row["source"])

    # -----------------------------------------
    # 1K project
    # -----------------------------------------

    if source == "1k_project":

        path = row["audio_path"]

        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)

        audio, sr = load_wav(path)

        return normalize_audio(audio, sr)

    # -----------------------------------------
    # Benchmark
    # -----------------------------------------

    if source == "benchmark":

        path = row["audio_path"]

        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)

        audio, sr = load_wav(path)

        return normalize_audio(audio, sr)

    # -----------------------------------------
    # IndicVoices
    # -----------------------------------------

    if source == "indicvoices":

        parquet_file = row["parquet_file"]
        row_index = int(row["parquet_row"])

        audio, sr = load_parquet_audio(
            parquet_file,
            row_index,
        )

        return normalize_audio(audio, sr)

    raise ValueError(
        f"Unknown source: {source}"
    )


# ---------------------------------------------------------
# Encode
# ---------------------------------------------------------

def encode(audio):

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    input_features = inputs["input_features"].to(device)

    attention_mask = inputs.get("attention_mask")

    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():

        outputs = encoder(
            input_features=input_features,
            attention_mask=attention_mask,
        )

    features = outputs.last_hidden_state[0]

    return features.float().cpu()


# ---------------------------------------------------------
# Select test samples
# ---------------------------------------------------------

print("\n[2/4] Loading manifests...")

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

print("Train:", len(train_df))
print("Val:", len(val_df))

print("\nSources:")
print(train_df["source"].value_counts())

# Pick one example from each source
samples = []

for source in [
    "indicvoices",
    "1k_project",
    "benchmark",
]:

    subset = train_df[
        train_df["source"] == source
    ]

    if len(subset) == 0:
        print("WARNING: no sample for", source)
        continue

    samples.append(
        subset.iloc[0]
    )

# ---------------------------------------------------------
# Test
# ---------------------------------------------------------

print("\n[3/4] Testing audio resolution + W2V-BERT...")

results = []

for i, row in enumerate(samples, 1):

    source = row["source"]

    print("\n" + "-" * 70)
    print(f"TEST {i}: {source}")
    print("-" * 70)

    print("Text:", str(row["text"])[:100])

    audio = resolve_audio(row)

    print(
        "Normalized samples:",
        len(audio),
    )

    print(
        "Duration:",
        round(len(audio) / 16000, 3),
        "seconds",
    )

    print(
        "Min:",
        float(audio.min()),
        "Max:",
        float(audio.max()),
    )

    features = encode(audio)

    print(
        "W2V-BERT features:",
        tuple(features.shape),
    )

    assert features.ndim == 2
    assert features.shape[1] == 1024

    results.append({
        "source": source,
        "audio_seconds": len(audio) / 16000,
        "feature_frames": features.shape[0],
        "feature_dim": features.shape[1],
    })

# ---------------------------------------------------------
# Final
# ---------------------------------------------------------

print("\n[4/4] RESULTS")
print("=" * 70)

for r in results:
    print(
        f"{r['source']:20s} "
        f"audio={r['audio_seconds']:.2f}s "
        f"features=({r['feature_frames']}, {r['feature_dim']})"
    )

print("\nTEST: PASS")
print("All three dataset sources successfully resolved.")
print("W2V-BERT produced 1024-dimensional features.")
