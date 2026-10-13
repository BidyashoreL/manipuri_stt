from pathlib import Path
import os
import io
import wave
import json
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()
ASR = ROOT.parent

TRAINER_PATH = ROOT / "train_b3_narrow_E11_E12.py"

VAL_CSV = ASR / "combined_dataset" / "val.csv"

E11_CHECKPOINT = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "best.pt"
)

VOCAB_PATH = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

B2_CHECKPOINT = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "regularized"
    / "FINAL_B2_MODEL.pt"
)

ENCODER_MODEL = Path(r"E:\hf_models\w2v-bert-2.0")

OUT_DIR = ROOT / "diagnostics" / "representation_forensics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAMPLE_RATE = 16000
SPECIAL_BLANK = "<blank>"
SPECIAL_UNK = "<unk>"

TRAINABLE_LAYERS = [22, 23]


# ============================================================
# TEXT
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# AUDIO
# ============================================================

def decode_pcm(raw, sample_width):

    if sample_width == 1:
        audio = np.frombuffer(
            raw,
            dtype=np.uint8
        ).astype(np.float32)

        return (audio - 128.0) / 128.0

    if sample_width == 2:
        return (
            np.frombuffer(
                raw,
                dtype=np.int16
            ).astype(np.float32)
            / 32768.0
        )

    if sample_width == 3:

        b = np.frombuffer(
            raw,
            dtype=np.uint8
        )

        if len(b) % 3:
            raise ValueError(
                "Invalid 24-bit PCM byte length"
            )

        b = b.reshape(-1, 3)

        audio = (
            b[:, 0].astype(np.int32)
            | (b[:, 1].astype(np.int32) << 8)
            | (b[:, 2].astype(np.int32) << 16)
        )

        mask = audio & 0x800000
        audio = audio - (mask << 1)

        return (
            audio.astype(np.float32)
            / 8388608.0
        )

    if sample_width == 4:
        return (
            np.frombuffer(
                raw,
                dtype=np.int32
            ).astype(np.float32)
            / 2147483648.0
        )

    raise ValueError(
        f"Unsupported PCM width: {sample_width}"
    )


@lru_cache(maxsize=4)
def read_parquet_audio_table(parquet_file):

    return pd.read_parquet(
        str(parquet_file),
        columns=["audio"]
    )


def load_parquet_audio(
    parquet_file,
    row_index
):

    parquet_file = str(parquet_file)

    df = read_parquet_audio_table(
        parquet_file
    )

    obj = df.iloc[int(row_index)]["audio"]

    if not isinstance(obj, dict):
        raise ValueError(
            f"Unexpected audio object: {type(obj)}"
        )

    raw_bytes = obj.get("bytes")

    if raw_bytes is None:
        raise ValueError(
            "IndicVoices audio object has no bytes"
        )

    with wave.open(
        io.BytesIO(raw_bytes),
        "rb"
    ) as wf:

        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()

        raw = wf.readframes(frames)

    audio = decode_pcm(
        raw,
        width
    )

    if channels > 1:
        audio = audio.reshape(
            -1,
            channels
        ).mean(axis=1)

    return audio, sr


def normalize_audio(
    audio,
    sample_rate
):

    audio = torch.as_tensor(
        audio,
        dtype=torch.float32
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
            SAMPLE_RATE
        )

    return audio.contiguous()


def resolve_audio(row):

    source = str(
        row.get("source", "")
    ).strip().lower()

    if source == "indicvoices":

        audio, sr = load_parquet_audio(
            row["parquet_file"],
            int(row["parquet_row"])
        )

        return normalize_audio(
            audio,
            sr
        )

    path = row.get(
        "audio_path",
        ""
    )

    if pd.isna(path):
        raise RuntimeError(
            "audio_path is NaN"
        )

    path = str(path).strip()

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    audio, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False
    )

    return normalize_audio(
        audio,
        sr
    )


# ============================================================
# VOCAB
# ============================================================

def load_vocab():

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        vocab = json.load(f)

    if len(vocab) != 60:
        raise RuntimeError(
            f"Expected vocab size 60, got {len(vocab)}"
        )

    if vocab.get(SPECIAL_BLANK) != 0:
        raise RuntimeError(
            "Blank token is not ID 0"
        )

    return vocab


# ============================================================
# IMPORT EXACT B3 MODEL
# ============================================================

print("=" * 72)
print("B3-NARROW REPRESENTATION FORENSICS")
print("=" * 72)

print("[DEVICE]", DEVICE)
print("[TRAINER]", TRAINER_PATH)
print("[E11]", E11_CHECKPOINT)

if not TRAINER_PATH.exists():
    raise FileNotFoundError(
        TRAINER_PATH
    )

if not E11_CHECKPOINT.exists():
    raise FileNotFoundError(
        E11_CHECKPOINT
    )

if not VAL_CSV.exists():
    raise FileNotFoundError(
        VAL_CSV
    )

# Load the exact working trainer module.
import importlib.util

spec = importlib.util.spec_from_file_location(
    "b3_trainer",
    TRAINER_PATH
)

b3 = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(b3)

print("[IMPORT] Exact B3 trainer imported.")


# ============================================================
# LOAD MODEL
# ============================================================

vocab = b3.load_vocab()

model = b3.B2Model(
    len(vocab)
)

b3.load_b2_checkpoint(
    model
)

b3.configure_b3(
    model
)

checkpoint = torch.load(
    E11_CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]

    elif "model" in checkpoint:
        state = checkpoint["model"]

    else:
        state = checkpoint

else:

    state = checkpoint


missing, unexpected = model.load_state_dict(
    state,
    strict=False
)

if missing:

    raise RuntimeError(
        "Missing E11 keys:\n"
        + "\n".join(
            map(str, missing)
        )
    )

if unexpected:

    print(
        "[WARNING] Unexpected keys:",
        unexpected
    )


model.eval()

print("[MODEL] E11 best.pt loaded.")


# ============================================================
# VALIDATION DATA
# ============================================================

val_df = pd.read_csv(
    VAL_CSV,
    encoding="utf-8-sig"
)

if len(val_df) != 864:
    raise RuntimeError(
        f"Expected 864 validation rows, got {len(val_df)}"
    )

print(
    f"[DATA] Validation rows: {len(val_df)}"
)


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def make_features(
    processor,
    audio
):

    arr = [
        audio.cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False
        )
    ]

    result = processor(
        arr,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    return result.input_features


# ============================================================
# ENTROPY
# ============================================================

def entropy_from_probs(
    probs
):

    return -(
        probs
        * torch.log(
            probs.clamp_min(1e-9)
        )
    ).sum(dim=-1)


# ============================================================
# FORWARD HOOKS
# ============================================================

representation_records = []

severe_records = []


# ============================================================
# MAIN LOOP
# ============================================================

print()
print("[RUN] Starting representation analysis...")
print()

with torch.no_grad():

    for idx in range(
        len(val_df)
    ):

        row = val_df.iloc[idx]

        audio = resolve_audio(
            row
        )

        features = make_features(
            model.processor,
            audio
        ).to(DEVICE)

        # ----------------------------------------------------
        # W2V-BERT encoder output
        # ----------------------------------------------------

        encoder_out = model.encoder(
            input_features=features
        )

        hidden = (
            encoder_out.last_hidden_state
        )

        # ----------------------------------------------------
        # Adapter output
        # ----------------------------------------------------

        adapted = model.adapter(
            hidden
        )

        logits = model.ctc_head(
            adapted
        )

        probs = torch.softmax(
            logits.float(),
            dim=-1
        )

        # ----------------------------------------------------
        # CTC statistics
        # ----------------------------------------------------

        blank_prob = probs[..., 0]

        nonblank_probs = probs[..., 1:]

        max_nonblank_prob = (
            nonblank_probs.max(
                dim=-1
            ).values
        )

        predicted_ids = probs.argmax(
            dim=-1
        )

        active_mask = (
            predicted_ids != 0
        )

        active_fraction = (
            active_mask.float()
            .mean()
            .item()
        )

        blank_fraction = (
            blank_prob > 0.5
        ).float().mean().item()

        mean_blank_probability = (
            blank_prob.mean().item()
        )

        max_nonblank_mean = (
            max_nonblank_prob.mean().item()
        )

        max_nonblank_max = (
            max_nonblank_prob.max().item()
        )

        entropy = entropy_from_probs(
            probs
        )

        mean_entropy = (
            entropy.mean().item()
        )

        entropy_std = (
            entropy.std().item()
            if entropy.numel() > 1
            else 0.0
        )

        # ----------------------------------------------------
        # Hidden representation statistics
        # ----------------------------------------------------

        hidden_norm = (
            hidden.norm(
                dim=-1
            )
        )

        adapted_norm = (
            adapted.norm(
                dim=-1
            )
        )

        hidden_frame_std = (
            hidden.std(
                dim=-1
            )
        )

        adapted_frame_std = (
            adapted.std(
                dim=-1
            )
        )

        # temporal differences
        if hidden.shape[1] > 1:

            hidden_delta = (
                hidden[:, 1:, :]
                - hidden[:, :-1, :]
            )

            temporal_change = (
                hidden_delta.norm(
                    dim=-1
                ).mean().item()
            )

        else:

            temporal_change = 0.0

        # ----------------------------------------------------
        # Temporal speech concentration
        # ----------------------------------------------------

        T = hidden.shape[1]

        active_positions = (
            active_mask[0]
            .nonzero(
                as_tuple=False
            )
            .flatten()
        )

        if active_positions.numel() > 0:

            first_active = (
                active_positions[0]
                .item()
                / max(T, 1)
            )

            last_active = (
                active_positions[-1]
                .item()
                / max(T, 1)
            )

            active_span = (
                last_active
                - first_active
            )

        else:

            first_active = 1.0
            last_active = 0.0
            active_span = 0.0

        # ----------------------------------------------------
        # Reference text
        # ----------------------------------------------------

        text = normalize_text(
            row["text"]
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        record = {

            "sample_index": idx,

            "source": str(
                row.get(
                    "source",
                    ""
                )
            ),

            "speaker_id": str(
                row.get(
                    "speaker_id",
                    ""
                )
            ),

            "duration": float(
                row.get(
                    "duration",
                    np.nan
                )
            ),

            "ref_len": len(text),

            "T": int(T),

            "hidden_norm_mean":
                hidden_norm.mean().item(),

            "hidden_norm_std":
                hidden_norm.std().item(),

            "hidden_norm_min":
                hidden_norm.min().item(),

            "hidden_norm_max":
                hidden_norm.max().item(),

            "hidden_frame_std_mean":
                hidden_frame_std.mean().item(),

            "hidden_frame_std_std":
                hidden_frame_std.std().item(),

            "adapted_norm_mean":
                adapted_norm.mean().item(),

            "adapted_norm_std":
                adapted_norm.std().item(),

            "adapted_norm_min":
                adapted_norm.min().item(),

            "adapted_norm_max":
                adapted_norm.max().item(),

            "adapted_frame_std_mean":
                adapted_frame_std.mean().item(),

            "adapted_frame_std_std":
                adapted_frame_std.std().item(),

            "temporal_change":
                temporal_change,

            "mean_blank_probability":
                mean_blank_probability,

            "blank_fraction":
                blank_fraction,

            "active_fraction":
                active_fraction,

            "max_nonblank_mean":
                max_nonblank_mean,

            "max_nonblank_max":
                max_nonblank_max,

            "entropy_mean":
                mean_entropy,

            "entropy_std":
                entropy_std,

            "first_active_fraction":
                first_active,

            "last_active_fraction":
                last_active,

            "active_span_fraction":
                active_span,
        }

        representation_records.append(
            record
        )

        if idx % 50 == 0:

            print(
                f"[{idx:03d}/864] "
                f"T={T:4d} "
                f"blank={blank_fraction:.3f} "
                f"active={active_fraction:.3f} "
                f"entropy={mean_entropy:.3f}"
            )


# ============================================================
# DATAFRAME
# ============================================================

df = pd.DataFrame(
    representation_records
)

# ------------------------------------------------------------
# Merge previous blank/error information if available
# ------------------------------------------------------------

blank_dir = (
    ROOT
    / "diagnostics"
    / "blank_attribution"
)

candidate_blank_files = list(
    blank_dir.glob("*.csv")
)

blank_df = None

for p in candidate_blank_files:

    try:

        tmp = pd.read_csv(p)

        if (
            "sample_index" in tmp.columns
            and
            (
                "collapse_class"
                in tmp.columns
            )
        ):

            blank_df = tmp
            print(
                "[MERGE] Using blank file:",
                p
            )
            break

    except Exception:
        pass


if blank_df is not None:

    keep = [
        c for c in [
            "sample_index",
            "cer",
            "wer",
            "pred_len",
            "ref_len",
            "blank_fraction",
            "mean_blank_probability",
            "collapse_class"
        ]
        if c in blank_df.columns
    ]

    blank_df = blank_df[keep]

    df = df.merge(
        blank_df,
        on="sample_index",
        how="left",
        suffixes=(
            "",
            "_old"
        )
    )


# ============================================================
# GROUPING
# ============================================================

if "collapse_class" not in df.columns:

    def classify_blank(x):

        if x >= 0.95:
            return "SEVERE"

        if x >= 0.90:
            return "HIGH"

        if x >= 0.80:
            return "MODERATE"

        return "NORMAL"

    df["collapse_class"] = (
        df["blank_fraction"]
        .apply(classify_blank)
    )


# ============================================================
# SUMMARY
# ============================================================

numeric_columns = [

    "hidden_norm_mean",
    "hidden_norm_std",
    "hidden_frame_std_mean",
    "adapted_norm_mean",
    "adapted_norm_std",
    "adapted_frame_std_mean",
    "temporal_change",
    "mean_blank_probability",
    "blank_fraction",
    "active_fraction",
    "max_nonblank_mean",
    "max_nonblank_max",
    "entropy_mean",
    "entropy_std",
    "first_active_fraction",
    "last_active_fraction",
    "active_span_fraction",
]

summary = (
    df
    .groupby("collapse_class")[
        [
            c
            for c in numeric_columns
            if c in df.columns
        ]
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std"
        ]
    )
)

summary.to_csv(
    OUT_DIR
    / "representation_summary.csv"
)


# ============================================================
# NORMAL VS SEVERE EFFECT SIZES
# ============================================================

rows = []

normal = df[
    df["collapse_class"]
    == "NORMAL"
]

severe = df[
    df["collapse_class"]
    == "SEVERE"
]

for col in numeric_columns:

    if col not in df.columns:
        continue

    n = normal[col].dropna()
    s = severe[col].dropna()

    if len(n) == 0 or len(s) == 0:
        continue

    nmean = n.mean()
    smean = s.mean()

    pooled = np.sqrt(
        (
            n.var(ddof=1)
            +
            s.var(ddof=1)
        )
        / 2
    )

    d = (
        (smean - nmean)
        / pooled
        if pooled > 0
        else np.nan
    )

    rows.append({

        "feature": col,

        "normal_n": len(n),
        "severe_n": len(s),

        "normal_mean": nmean,
        "severe_mean": smean,

        "delta_severe_minus_normal":
            smean - nmean,

        "cohens_d":
            d,
    })


effect_df = pd.DataFrame(rows)

effect_df.to_csv(
    OUT_DIR
    / "normal_vs_severe_representation.csv",
    index=False
)


# ============================================================
# SEVERE CASES
# ============================================================

severe_df = (
    df[
        df["collapse_class"]
        == "SEVERE"
    ]
    .sort_values(
        "blank_fraction",
        ascending=False
    )
)

severe_df.to_csv(
    OUT_DIR
    / "severe_representation_cases.csv",
    index=False
)


# ============================================================
# HIGH / MODERATE / SEVERE
# ============================================================

collapse_df = (
    df[
        df["collapse_class"]
        .isin(
            [
                "MODERATE",
                "HIGH",
                "SEVERE"
            ]
        )
    ]
    .sort_values(
        [
            "collapse_class",
            "blank_fraction"
        ],
        ascending=[
            True,
            False
        ]
    )
)

collapse_df.to_csv(
    OUT_DIR
    / "collapse_representation_cases.csv",
    index=False
)


# ============================================================
# CORRELATIONS
# ============================================================

targets = [
    "cer",
    "wer",
    "blank_fraction",
    "mean_blank_probability",
    "active_fraction",
    "entropy_mean",
]

features = [
    c
    for c in numeric_columns
    if c in df.columns
]

corr_rows = []

for f in features:

    for t in targets:

        if t not in df.columns:
            continue

        x = df[f]
        y = df[t]

        mask = (
            x.notna()
            & y.notna()
        )

        if mask.sum() < 3:
            continue

        corr = np.corrcoef(
            x[mask],
            y[mask]
        )[0, 1]

        corr_rows.append({

            "feature": f,
            "target": t,
            "pearson_r": corr,
            "n": int(mask.sum()),
        })


pd.DataFrame(
    corr_rows
).to_csv(
    OUT_DIR
    / "representation_error_correlations.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

df.to_csv(
    OUT_DIR
    / "representation_all_864.csv",
    index=False
)

print()
print("=" * 72)
print("REPRESENTATION FORENSICS COMPLETE")
print("=" * 72)

print(
    "[OUTPUT]",
    OUT_DIR
)

print()
print(
    summary
)

print()
print(
    "Severe samples:",
    len(severe_df)
)

print(
    severe_df[
        [
            "sample_index",
            "speaker_id",
            "T",
            "ref_len",
            "hidden_norm_mean",
            "adapted_norm_mean",
            "blank_fraction",
            "active_fraction",
            "max_nonblank_mean",
            "entropy_mean"
        ]
    ].to_string(
        index=False
    )
)