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

OUT_DIR = (
    ROOT
    / "diagnostics"
    / "ctc_logit_forensics"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SAMPLE_RATE = 16000

SPECIAL_BLANK = "<blank>"
SPECIAL_UNK = "<unk>"


# ============================================================
# TEXT
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = unicodedata.normalize(
        "NFC",
        str(text)
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# AUDIO
# ============================================================

def decode_pcm(
    raw,
    sample_width
):

    if sample_width == 1:

        audio = np.frombuffer(
            raw,
            dtype=np.uint8
        ).astype(np.float32)

        return (
            audio - 128.0
        ) / 128.0

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
                "Invalid 24-bit PCM"
            )

        b = b.reshape(
            -1,
            3
        )

        audio = (
            b[:, 0].astype(np.int32)
            |
            (
                b[:, 1].astype(np.int32)
                << 8
            )
            |
            (
                b[:, 2].astype(np.int32)
                << 16
            )
        )

        mask = audio & 0x800000

        audio = (
            audio
            - (mask << 1)
        )

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
        f"Unsupported sample width: {sample_width}"
    )


@lru_cache(maxsize=4)
def read_parquet_audio_table(
    parquet_file
):

    return pd.read_parquet(
        str(parquet_file),
        columns=["audio"]
    )


def load_parquet_audio(
    parquet_file,
    row_index
):

    parquet_file = str(
        parquet_file
    )

    df = read_parquet_audio_table(
        parquet_file
    )

    obj = df.iloc[
        int(row_index)
    ]["audio"]

    if not isinstance(
        obj,
        dict
    ):
        raise ValueError(
            f"Unexpected audio object: {type(obj)}"
        )

    raw_bytes = obj.get(
        "bytes"
    )

    if raw_bytes is None:
        raise ValueError(
            "No audio bytes"
        )

    with wave.open(
        io.BytesIO(raw_bytes),
        "rb"
    ) as wf:

        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()

        raw = wf.readframes(
            frames
        )

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
            audio = audio.mean(
                dim=0
            )
        else:
            audio = audio.mean(
                dim=1
            )

    elif audio.ndim > 2:

        audio = audio.squeeze()

    if sample_rate != SAMPLE_RATE:

        audio = (
            torchaudio.functional.resample(
                audio,
                sample_rate,
                SAMPLE_RATE
            )
        )

    return audio.contiguous()


def resolve_audio(row):

    source = str(
        row.get(
            "source",
            ""
        )
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
        raise FileNotFoundError(
            path
        )

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
# FEATURES
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
# MODEL
# ============================================================

print("=" * 72)
print("B3-NARROW CTC LOGIT FORENSICS")
print("=" * 72)

print(
    "[DEVICE]",
    DEVICE
)

print(
    "[TRAINER]",
    TRAINER_PATH
)

print(
    "[CHECKPOINT]",
    E11_CHECKPOINT
)


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


# Import exact working model implementation.

import importlib.util

spec = (
    importlib.util.spec_from_file_location(
        "b3_trainer",
        TRAINER_PATH
    )
)

b3 = (
    importlib.util.module_from_spec(
        spec
    )
)

spec.loader.exec_module(
    b3
)

print(
    "[IMPORT] Exact B3 trainer loaded."
)


# Build exact B3 model.

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


# Load E11.

checkpoint = torch.load(
    E11_CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

if isinstance(
    checkpoint,
    dict
):

    if "model_state_dict" in checkpoint:

        state = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif "state_dict" in checkpoint:

        state = (
            checkpoint[
                "state_dict"
            ]
        )

    elif "model" in checkpoint:

        state = (
            checkpoint["model"]
        )

    else:

        state = checkpoint

else:

    state = checkpoint


missing, unexpected = (
    model.load_state_dict(
        state,
        strict=False
    )
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

print(
    "[MODEL] E11 loaded successfully."
)


# ============================================================
# DATA
# ============================================================

val_df = pd.read_csv(
    VAL_CSV,
    encoding="utf-8-sig"
)

if len(val_df) != 864:

    raise RuntimeError(
        f"Expected 864 rows, got {len(val_df)}"
    )

print(
    "[DATA] Validation:",
    len(val_df)
)


# ============================================================
# VOCAB REVERSE MAP
# ============================================================

inv_vocab = {
    int(v): k
    for k, v in vocab.items()
}


# ============================================================
# ANALYSIS
# ============================================================

records = []

frame_records = []


print()
print(
    "[RUN] Extracting CTC logits..."
)
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

        # Exact B3 forward,
        # but explicitly expose logits.

        hidden = (
            model.encoder(
                input_features=features
            ).last_hidden_state
        )

        adapted = model.adapter(
            hidden
        )

        logits = model.ctc_head(
            adapted
        )

        # Shape:
        # [1, T, 60]

        logits = logits[0].float()

        probs = torch.softmax(
            logits,
            dim=-1
        )

        blank_logits = (
            logits[:, 0]
        )

        nonblank_logits = (
            logits[:, 1:]
        )

        blank_probs = (
            probs[:, 0]
        )

        nonblank_probs = (
            probs[:, 1:]
        )

        # Best non-blank character.

        best_nonblank_logits, best_nonblank_ids = (
            nonblank_logits.max(
                dim=-1
            )
        )

        best_nonblank_probs, _ = (
            nonblank_probs.max(
                dim=-1
            )
        )

        # Critical quantity:
        #
        # blank logit - best character logit

        blank_margin = (
            blank_logits
            - best_nonblank_logits
        )

        # Probability margin.

        blank_probability_margin = (
            blank_probs
            - best_nonblank_probs
        )

        # Predicted class.

        predicted_ids = (
            logits.argmax(
                dim=-1
            )
        )

        blank_argmax = (
            predicted_ids == 0
        )

        # ----------------------------------------------------
        # Frame statistics
        # ----------------------------------------------------

        T = logits.shape[0]

        blank_argmax_fraction = (
            blank_argmax.float()
            .mean()
            .item()
        )

        blank_probability_mean = (
            blank_probs.mean()
            .item()
        )

        blank_probability_median = (
            blank_probs.median()
            .item()
        )

        blank_probability_max = (
            blank_probs.max()
            .item()
        )

        best_nonblank_probability_mean = (
            best_nonblank_probs.mean()
            .item()
        )

        best_nonblank_probability_max = (
            best_nonblank_probs.max()
            .item()
        )

        margin_mean = (
            blank_margin.mean()
            .item()
        )

        margin_median = (
            blank_margin.median()
            .item()
        )

        margin_std = (
            blank_margin.std()
            .item()
        )

        margin_min = (
            blank_margin.min()
            .item()
        )

        margin_max = (
            blank_margin.max()
            .item()
        )

        positive_margin_fraction = (
            blank_margin > 0
        ).float().mean().item()

        strong_blank_fraction = (
            blank_margin > 2.0
        ).float().mean().item()

        very_strong_blank_fraction = (
            blank_margin > 4.0
        ).float().mean().item()

        negative_margin_fraction = (
            blank_margin < 0
        ).float().mean().item()

        # ----------------------------------------------------
        # Entropy
        # ----------------------------------------------------

        entropy = -(
            probs
            * torch.log(
                probs.clamp_min(
                    1e-9
                )
            )
        ).sum(
            dim=-1
        )

        entropy_mean = (
            entropy.mean()
            .item()
        )

        # ----------------------------------------------------
        # Top character
        # ----------------------------------------------------

        top_char_frame = int(
            best_nonblank_ids[
                best_nonblank_probs.argmax()
            ].item()
        )

        top_char = inv_vocab.get(
            top_char_frame,
            SPECIAL_UNK
        )

        # ----------------------------------------------------
        # Reference
        # ----------------------------------------------------

        text = normalize_text(
            row["text"]
        )

        # ----------------------------------------------------
        # Store summary
        # ----------------------------------------------------

        records.append({

            "sample_index":
                idx,

            "speaker_id":
                str(
                    row.get(
                        "speaker_id",
                        ""
                    )
                ),

            "source":
                str(
                    row.get(
                        "source",
                        ""
                    )
                ),

            "duration":
                float(
                    row.get(
                        "duration",
                        np.nan
                    )
                ),

            "ref_len":
                len(text),

            "T":
                int(T),

            "blank_argmax_fraction":
                blank_argmax_fraction,

            "blank_probability_mean":
                blank_probability_mean,

            "blank_probability_median":
                blank_probability_median,

            "blank_probability_max":
                blank_probability_max,

            "best_nonblank_probability_mean":
                best_nonblank_probability_mean,

            "best_nonblank_probability_max":
                best_nonblank_probability_max,

            "blank_margin_mean":
                margin_mean,

            "blank_margin_median":
                margin_median,

            "blank_margin_std":
                margin_std,

            "blank_margin_min":
                margin_min,

            "blank_margin_max":
                margin_max,

            "positive_margin_fraction":
                positive_margin_fraction,

            "strong_blank_fraction":
                strong_blank_fraction,

            "very_strong_blank_fraction":
                very_strong_blank_fraction,

            "negative_margin_fraction":
                negative_margin_fraction,

            "entropy_mean":
                entropy_mean,

            "strongest_nonblank_char":
                top_char
        })


        # ----------------------------------------------------
        # Store frame-level data ONLY for important samples
        #
        # This avoids a gigantic CSV.
        # ----------------------------------------------------

        important = (
            idx in
            [
                187,
                491,
                568,
                814,
                803,
                349
            ]
        )

        if important:

            for t in range(T):

                frame_records.append({

                    "sample_index":
                        idx,

                    "frame":
                        t,

                    "blank_logit":
                        blank_logits[t].item(),

                    "best_nonblank_logit":
                        best_nonblank_logits[t].item(),

                    "blank_margin":
                        blank_margin[t].item(),

                    "blank_probability":
                        blank_probs[t].item(),

                    "best_nonblank_probability":
                        best_nonblank_probs[t].item(),

                    "entropy":
                        entropy[t].item(),

                    "best_nonblank_id":
                        int(
                            best_nonblank_ids[t]
                            .item()
                        ),

                    "best_nonblank_char":
                        inv_vocab.get(
                            int(
                                best_nonblank_ids[t]
                                .item()
                            ),
                            SPECIAL_UNK
                        ),

                    "predicted_id":
                        int(
                            predicted_ids[t]
                            .item()
                        ),

                    "predicted_char":
                        inv_vocab.get(
                            int(
                                predicted_ids[t]
                                .item()
                            ),
                            SPECIAL_UNK
                        )
                })


        if idx % 50 == 0:

            print(
                f"[{idx:03d}/864] "
                f"T={T:4d} "
                f"blank={blank_argmax_fraction:.3f} "
                f"margin={margin_mean:.3f} "
                f"strong={strong_blank_fraction:.3f}"
            )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(
    records
)

frames = pd.DataFrame(
    frame_records
)


df.to_csv(
    OUT_DIR
    / "ctc_logit_summary_864.csv",
    index=False
)

frames.to_csv(
    OUT_DIR
    / "frame_logits_important_samples.csv",
    index=False
)


# ============================================================
# MERGE CANONICAL BLANK / ERROR DATA
# ============================================================

blank_dir = (
    ROOT
    / "diagnostics"
    / "blank_attribution"
)

canonical_blank = None

for p in blank_dir.glob(
    "*.csv"
):

    try:

        tmp = pd.read_csv(
            p
        )

        required = {
            "sample_index",
            "collapse_class"
        }

        if required.issubset(
            tmp.columns
        ):

            canonical_blank = tmp

            print(
                "[MERGE] Blank file:",
                p
            )

            break

    except Exception:
        continue


if canonical_blank is not None:

    cols = [
        c
        for c in [
            "sample_index",
            "cer",
            "wer",
            "pred_len",
            "ref_len",
            "blank_fraction",
            "mean_blank_probability",
            "collapse_class"
        ]
        if c in canonical_blank.columns
    ]

    canonical_blank = (
        canonical_blank[
            cols
        ]
        .drop_duplicates(
            "sample_index"
        )
    )

    merged = df.merge(
        canonical_blank,
        on="sample_index",
        how="left",
        suffixes=(
            "",
            "_canonical"
        )
    )

else:

    merged = df.copy()


merged.to_csv(
    OUT_DIR
    / "ctc_logit_summary_with_errors.csv",
    index=False
)


# ============================================================
# GROUP SUMMARY
# ============================================================

if "collapse_class" in merged.columns:

    group_cols = [
        "blank_argmax_fraction",
        "blank_probability_mean",
        "best_nonblank_probability_mean",
        "blank_margin_mean",
        "blank_margin_median",
        "positive_margin_fraction",
        "strong_blank_fraction",
        "very_strong_blank_fraction",
        "negative_margin_fraction",
        "entropy_mean"
    ]

    group_cols = [
        c
        for c in group_cols
        if c in merged.columns
    ]

    summary = (
        merged
        .groupby(
            "collapse_class"
        )[group_cols]
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
        / "ctc_logit_group_summary.csv"
    )


# ============================================================
# SEVERE / HIGH CASES
# ============================================================

if "collapse_class" in merged.columns:

    critical = (
        merged[
            merged[
                "collapse_class"
            ].isin(
                [
                    "HIGH",
                    "SEVERE"
                ]
            )
        ]
        .sort_values(
            "blank_margin_mean",
            ascending=False
        )
    )

    critical.to_csv(
        OUT_DIR
        / "critical_blank_margin_cases.csv",
        index=False
    )


# ============================================================
# TOP BLANK-MARGIN CASES
# ============================================================

top_margin = (
    merged
    .sort_values(
        "blank_margin_mean",
        ascending=False
    )
    .head(50)
)

top_margin.to_csv(
    OUT_DIR
    / "top_50_blank_margin_cases.csv",
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("CTC LOGIT FORENSICS COMPLETE")
print("=" * 72)

print(
    "[OUTPUT]",
    OUT_DIR
)

print()

if "collapse_class" in merged.columns:

    print(
        merged
        .groupby(
            "collapse_class"
        )[
            [
                "blank_argmax_fraction",
                "blank_probability_mean",
                "best_nonblank_probability_mean",
                "blank_margin_mean",
                "positive_margin_fraction",
                "strong_blank_fraction",
                "very_strong_blank_fraction",
                "entropy_mean"
            ]
        ]
        .mean()
        .to_string()
    )

print()
print(
    "Important samples:"
)

important_indices = [
    187,
    491,
    568,
    814,
    803,
    349
]

print(
    merged[
        merged[
            "sample_index"
        ].isin(
            important_indices
        )
    ][
        [
            "sample_index",
            "T",
            "ref_len",
            "blank_argmax_fraction",
            "blank_probability_mean",
            "best_nonblank_probability_mean",
            "blank_margin_mean",
            "positive_margin_fraction",
            "strong_blank_fraction",
            "very_strong_blank_fraction",
            "entropy_mean"
        ]
    ].sort_values(
        "sample_index"
    ).to_string(
        index=False
    )
)