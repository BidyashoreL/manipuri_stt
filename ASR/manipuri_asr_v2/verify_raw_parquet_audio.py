"""
verify_raw_parquet_audio.py

Independent verification of the original IndicVoices parquet audio.

Purpose
-------
We previously found that validation samples:

    187
    491
    568
    814

are almost completely silent after our current audio loading path.

This script independently reads the ORIGINAL parquet audio and checks
whether the source itself is silent.

It deliberately does NOT import the B3 training script.

Main question:

    ORIGINAL PARQUET
          |
          v
    independent decoder
          |
          v
       WAV/audio
          |
          v
    Is the source actually silent?

Target speaker:
    S4259682900336548
"""


from pathlib import Path
import io
import json
import wave
import math
import warnings
import hashlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

TRAIN_CSV = (
    ASR
    / "combined_dataset"
    / "train.csv"
)

VAL_CSV = (
    ASR
    / "combined_dataset"
    / "val.csv"
)

OLD_FORENSIC_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
    / "speaker_forensics_audio"
)

OLD_WAV_DIR = (
    OLD_FORENSIC_DIR
    / "validation_audio"
)

OUTPUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
    / "raw_parquet_verification"
)

OUTPUT_WAV_DIR = (
    OUTPUT_DIR
    / "independent_wav"
)


TARGET_SPEAKER = "S4259682900336548"

TARGET_SAMPLES = [
    144,
    187,
    491,
    568,
    702,
    814,
]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("INDEPENDENT RAW PARQUET AUDIO VERIFICATION")
print("=" * 80)

print(f"Script directory : {SCRIPT_DIR}")
print(f"ASR directory    : {ASR}")
print(f"Train CSV        : {TRAIN_CSV}")
print(f"Val CSV          : {VAL_CSV}")
print(f"Target speaker   : {TARGET_SPEAKER}")
print(f"Target samples   : {TARGET_SAMPLES}")
print()

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_WAV_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CHECK DEPENDENCIES
# ============================================================

print("=" * 80)
print("CHECKING DEPENDENCIES")
print("=" * 80)

try:

    import pyarrow.parquet as pq

    print("PyArrow: OK")

except Exception as e:

    raise RuntimeError(
        "PyArrow is required.\n"
        "Install with:\n"
        "pip install pyarrow"
    ) from e


try:

    import soundfile as sf

    print(
        f"SoundFile: OK "
        f"(version {sf.__libsndfile_version__})"
    )

except Exception as e:

    sf = None

    print(
        "SoundFile: NOT AVAILABLE"
    )

    print(
        "We will use the built-in WAV decoder "
        "and scipy fallback where possible."
    )


try:

    import scipy

    from scipy.io import wavfile

    print(
        f"SciPy: OK "
        f"(version {scipy.__version__})"
    )

except Exception:

    scipy = None
    wavfile = None

    print(
        "SciPy: NOT AVAILABLE"
    )

print()


# ============================================================
# LOAD MANIFEST
# ============================================================

if not TRAIN_CSV.exists():

    raise FileNotFoundError(
        f"Missing:\n{TRAIN_CSV}"
    )

if not VAL_CSV.exists():

    raise FileNotFoundError(
        f"Missing:\n{VAL_CSV}"
    )


train_df = pd.read_csv(
    TRAIN_CSV
)

val_df = pd.read_csv(
    VAL_CSV
)

print("=" * 80)
print("MANIFEST")
print("=" * 80)

print(
    f"Train rows      : {len(train_df)}"
)

print(
    f"Validation rows : {len(val_df)}"
)

print()


# ============================================================
# VALIDATION SAMPLE INDEX
# ============================================================

# The B3 validation/cache pipeline uses the validation
# dataframe row order as its sample index.
#
# Therefore we explicitly attach:
#
#     sample_index = 0 ... len(val_df)-1
#
# This avoids confusion with the pandas dataframe index.

val_df = val_df.reset_index(
    drop=True
)

val_df[
    "sample_index"
] = np.arange(
    len(val_df)
)


target_df = val_df[
    val_df[
        "sample_index"
    ].isin(
        TARGET_SAMPLES
    )
].copy()


print("=" * 80)
print("TARGET ROWS")
print("=" * 80)

print(
    target_df[
        [
            "sample_index",
            "speaker_id",
            "source",
            "parquet_file",
            "parquet_row",
            "duration",
            "text",
        ]
    ].to_string(
        index=False
    )
)

print()


# ============================================================
# VERIFY SPEAKER
# ============================================================

bad_speaker = target_df[
    target_df[
        "speaker_id"
    ].astype(str)
    != TARGET_SPEAKER
]

if len(bad_speaker) > 0:

    raise RuntimeError(
        "At least one target sample does not belong "
        "to the expected speaker."
    )


# ============================================================
# PARQUET CACHE
# ============================================================

PARQUET_CACHE = {}


def load_parquet(
    parquet_path
):
    """
    Load a parquet file once and cache it.
    """

    parquet_path = str(
        Path(parquet_path)
    )

    if parquet_path in PARQUET_CACHE:

        return PARQUET_CACHE[
            parquet_path
        ]

    print()
    print(
        "Loading ORIGINAL parquet:"
    )

    print(
        parquet_path
    )

    table = pq.read_table(
        parquet_path
    )

    df = table.to_pandas()

    PARQUET_CACHE[
        parquet_path
    ] = df

    print(
        f"Rows loaded: {len(df)}"
    )

    print(
        f"Columns: {list(df.columns)}"
    )

    return df


# ============================================================
# AUDIO OBJECT DESCRIPTION
# ============================================================

def describe_audio_object(
    obj
):
    """
    Return a JSON-safe description of the raw parquet
    audio object.
    """

    description = {
        "python_type": str(
            type(obj)
        ),
    }

    if isinstance(
        obj,
        dict
    ):

        description[
            "dict_keys"
        ] = [
            str(k)
            for k in obj.keys()
        ]

        for key, value in obj.items():

            if key == "bytes":

                if value is None:

                    description[
                        "bytes"
                    ] = None

                else:

                    description[
                        "bytes_type"
                    ] = str(
                        type(value)
                    )

                    description[
                        "bytes_length"
                    ] = len(value)

                    description[
                        "bytes_sha256"
                    ] = hashlib.sha256(
                        bytes(value)
                    ).hexdigest()

            else:

                try:

                    json.dumps(
                        value
                    )

                    description[
                        str(key)
                    ] = value

                except Exception:

                    description[
                        str(key)
                    ] = str(
                        type(value)
                    )

    elif isinstance(
        obj,
        (bytes, bytearray)
    ):

        description[
            "bytes_length"
        ] = len(obj)

        description[
            "bytes_sha256"
        ] = hashlib.sha256(
            bytes(obj)
        ).hexdigest()

    else:

        description[
            "repr"
        ] = repr(obj)[:500]

    return description


# ============================================================
# RAW BYTES EXTRACTION
# ============================================================

def get_raw_audio_bytes(
    audio_obj
):
    """
    Extract actual encoded audio bytes from common
    HuggingFace Audio / IndicVoices parquet representations.
    """

    # --------------------------------------------------------
    # Direct bytes
    # --------------------------------------------------------

    if isinstance(
        audio_obj,
        (bytes, bytearray)
    ):

        return bytes(
            audio_obj
        )


    # --------------------------------------------------------
    # Dict
    # --------------------------------------------------------

    if isinstance(
        audio_obj,
        dict
    ):

        if "bytes" in audio_obj:

            data = audio_obj[
                "bytes"
            ]

            if data is not None:

                return bytes(
                    data
                )

        # If the parquet contains only a path,
        # resolve that separately.
        if "path" in audio_obj:

            path = audio_obj[
                "path"
            ]

            if path is not None:

                return Path(
                    str(path)
                )


    # --------------------------------------------------------
    # Path-like
    # --------------------------------------------------------

    if isinstance(
        audio_obj,
        (str, Path)
    ):

        return Path(
            str(audio_obj)
        )


    raise RuntimeError(
        "Unsupported parquet audio object:\n"
        f"{type(audio_obj)}\n"
        f"{repr(audio_obj)[:500]}"
    )


# ============================================================
# DECODE USING SOUNDFILE
# ============================================================

def decode_with_soundfile(
    raw
):
    """
    Decode using libsndfile.

    This is deliberately independent from the previous
    Python wave-only decoding path.
    """

    if sf is None:

        raise RuntimeError(
            "soundfile is unavailable"
        )

    # --------------------------------------------------------
    # Bytes
    # --------------------------------------------------------

    if isinstance(
        raw,
        (bytes, bytearray)
    ):

        audio, sr = sf.read(
            io.BytesIO(
                bytes(raw)
            ),
            always_2d=False,
            dtype="float32"
        )

    # --------------------------------------------------------
    # File
    # --------------------------------------------------------

    else:

        audio, sr = sf.read(
            str(raw),
            always_2d=False,
            dtype="float32"
        )

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    channels = 1

    if audio.ndim == 2:

        channels = audio.shape[1]

        audio = np.mean(
            audio,
            axis=1
        )

    return (
        audio,
        int(sr),
        int(channels),
        "soundfile"
    )


# ============================================================
# DECODE WAV USING BUILT-IN WAVE
# ============================================================

def decode_with_wave(
    raw_bytes
):
    """
    Independent fallback decoder.
    """

    if not isinstance(
        raw_bytes,
        (bytes, bytearray)
    ):

        raise RuntimeError(
            "Built-in wave fallback requires bytes."
        )

    with wave.open(
        io.BytesIO(
            bytes(raw_bytes)
        ),
        "rb"
    ) as wf:

        sr = wf.getframerate()

        channels = wf.getnchannels()

        sample_width = (
            wf.getsampwidth()
        )

        n_frames = (
            wf.getnframes()
        )

        raw_pcm = wf.readframes(
            n_frames
        )

    if sample_width == 1:

        audio = np.frombuffer(
            raw_pcm,
            dtype=np.uint8
        ).astype(
            np.float32
        )

        audio = (
            audio - 128.0
        ) / 128.0

    elif sample_width == 2:

        audio = np.frombuffer(
            raw_pcm,
            dtype=np.int16
        ).astype(
            np.float32
        )

        audio /= 32768.0

    elif sample_width == 3:

        # 24-bit PCM.
        data = np.frombuffer(
            raw_pcm,
            dtype=np.uint8
        )

        usable = (
            len(data) // 3
        ) * 3

        data = data[
            :usable
        ].reshape(
            -1,
            3
        )

        values = (
            data[:, 0].astype(
                np.int32
            )
            |
            (
                data[:, 1].astype(
                    np.int32
                ) << 8
            )
            |
            (
                data[:, 2].astype(
                    np.int32
                ) << 16
            )
        )

        sign_mask = (
            values & 0x800000
        ) != 0

        values[
            sign_mask
        ] -= 0x1000000

        audio = (
            values.astype(
                np.float32
            )
            / 8388608.0
        )

    elif sample_width == 4:

        audio = np.frombuffer(
            raw_pcm,
            dtype=np.int32
        ).astype(
            np.float32
        )

        audio /= 2147483648.0

    else:

        raise RuntimeError(
            f"Unsupported sample width: "
            f"{sample_width}"
        )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        )

        audio = audio.mean(
            axis=1
        )

    return (
        audio.astype(
            np.float32
        ),
        int(sr),
        int(channels),
        "wave"
    )


# ============================================================
# DECODE AUDIO
# ============================================================

def decode_independently(
    raw
):
    """
    Try SoundFile first.

    If SoundFile cannot decode the data, use wave
    when raw bytes are available.
    """

    soundfile_error = None

    if sf is not None:

        try:

            return decode_with_soundfile(
                raw
            )

        except Exception as e:

            soundfile_error = str(e)

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if isinstance(
        raw,
        (bytes, bytearray)
    ):

        try:

            return decode_with_wave(
                raw
            )

        except Exception as e:

            raise RuntimeError(
                "Both independent decoders failed.\n"
                f"SoundFile error: "
                f"{soundfile_error}\n"
                f"wave error: {e}"
            )

    raise RuntimeError(
        "Independent SoundFile decoder failed.\n"
        f"Error: {soundfile_error}"
    )


# ============================================================
# AUDIO METRICS
# ============================================================

def calculate_metrics(
    audio,
    sr
):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    n = len(audio)

    if n == 0:

        raise RuntimeError(
            "Decoded audio contains zero samples."
        )

    duration = (
        n / float(sr)
    )

    abs_audio = np.abs(
        audio
    )

    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = float(
        np.sqrt(
            np.mean(
                audio ** 2
            )
        )
    )

    rms_db = (
        20.0
        * math.log10(
            max(
                rms,
                1e-12
            )
        )
    )

    # --------------------------------------------------------
    # Peak
    # --------------------------------------------------------

    peak = float(
        np.max(
            abs_audio
        )
    )

    peak_db = (
        20.0
        * math.log10(
            max(
                peak,
                1e-12
            )
        )
    )

    # --------------------------------------------------------
    # Silence / activity
    #
    # Use TWO thresholds:
    #
    # 1. absolute threshold
    # 2. adaptive threshold
    #
    # This lets us distinguish actual silence from merely
    # low-energy speech.
    # --------------------------------------------------------

    absolute_threshold = 0.001

    absolute_active = (
        abs_audio
        > absolute_threshold
    )

    absolute_active_ratio = float(
        np.mean(
            absolute_active
        )
    )

    absolute_silence_ratio = (
        1.0
        - absolute_active_ratio
    )

    adaptive_threshold = max(
        rms * 0.10,
        1e-4
    )

    adaptive_active = (
        abs_audio
        > adaptive_threshold
    )

    adaptive_active_ratio = float(
        np.mean(
            adaptive_active
        )
    )

    adaptive_silence_ratio = (
        1.0
        - adaptive_active_ratio
    )

    # --------------------------------------------------------
    # Zero crossings
    # --------------------------------------------------------

    if n > 1:

        signs = np.signbit(
            audio
        )

        zc = np.sum(
            signs[1:]
            != signs[:-1]
        )

        zcr = float(
            zc / (n - 1)
        )

    else:

        zcr = 0.0

    # --------------------------------------------------------
    # Clipping
    # --------------------------------------------------------

    clipping_pct = float(
        np.mean(
            abs_audio >= 0.999
        )
        * 100.0
    )

    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    p01 = float(
        np.percentile(
            abs_audio,
            1
        )
    )

    p50 = float(
        np.percentile(
            abs_audio,
            50
        )
    )

    p95 = float(
        np.percentile(
            abs_audio,
            95
        )
    )

    p99 = float(
        np.percentile(
            abs_audio,
            99
        )
    )

    # --------------------------------------------------------
    # Dynamic range
    # --------------------------------------------------------

    dynamic_range_db = (
        peak_db
        - rms_db
    )

    # --------------------------------------------------------
    # Quantization / unique values
    # --------------------------------------------------------

    if len(audio) > 100000:

        sample_for_unique = (
            audio[
                :100000
            ]
        )

    else:

        sample_for_unique = audio

    unique_values = len(
        np.unique(
            sample_for_unique
        )
    )

    return {
        "num_samples": int(n),
        "sample_rate": int(sr),
        "duration": float(duration),

        "rms": rms,
        "rms_db": rms_db,

        "peak": peak,
        "peak_db": peak_db,

        "absolute_active_ratio":
            absolute_active_ratio,

        "absolute_silence_ratio":
            absolute_silence_ratio,

        "adaptive_active_ratio":
            adaptive_active_ratio,

        "adaptive_silence_ratio":
            adaptive_silence_ratio,

        "zero_crossing_rate":
            zcr,

        "clipping_pct":
            clipping_pct,

        "abs_p01": p01,
        "abs_p50": p50,
        "abs_p95": p95,
        "abs_p99": p99,

        "dynamic_range_db":
            dynamic_range_db,

        "unique_values_sample":
            unique_values,
    }


# ============================================================
# SAVE WAV
# ============================================================

def save_wav(
    path,
    audio,
    sr
):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    audio = np.clip(
        audio,
        -1.0,
        1.0
    )

    pcm = (
        audio * 32767.0
    ).astype(
        np.int16
    )

    with wave.open(
        str(path),
        "wb"
    ) as wf:

        wf.setnchannels(
            1
        )

        wf.setsampwidth(
            2
        )

        wf.setframerate(
            int(sr)
        )

        wf.writeframes(
            pcm.tobytes()
        )


# ============================================================
# HASH AUDIO
# ============================================================

def audio_hash(
    audio
):

    audio = np.asarray(
        audio,
        dtype=np.float32
    )

    return hashlib.sha256(
        audio.tobytes()
    ).hexdigest()


# ============================================================
# OLD FORENSIC WAV METRICS
# ============================================================

def analyze_old_wav(
    sample_index
):

    if not OLD_WAV_DIR.exists():

        return None

    candidates = list(
        OLD_WAV_DIR.glob(
            f"{sample_index:04d}_*.wav"
        )
    )

    if len(candidates) == 0:

        return None

    path = candidates[0]

    try:

        if sf is not None:

            audio, sr = sf.read(
                str(path),
                always_2d=False,
                dtype="float32"
            )

            audio = np.asarray(
                audio,
                dtype=np.float32
            )

            if audio.ndim == 2:

                audio = audio.mean(
                    axis=1
                )

        else:

            with wave.open(
                str(path),
                "rb"
            ) as wf:

                sr = wf.getframerate()
                channels = wf.getnchannels()
                width = wf.getsampwidth()
                frames = wf.getnframes()

                raw = wf.readframes(
                    frames
                )

            if width == 2:

                audio = (
                    np.frombuffer(
                        raw,
                        dtype=np.int16
                    ).astype(
                        np.float32
                    )
                    / 32768.0
                )

            else:

                return None

            if channels > 1:

                audio = audio.reshape(
                    -1,
                    channels
                ).mean(
                    axis=1
                )

        metrics = calculate_metrics(
            audio,
            sr
        )

        return {
            "path": str(path),
            "audio": audio,
            "sample_rate": sr,
            "metrics": metrics,
            "hash": audio_hash(
                audio
            ),
        }

    except Exception as e:

        print(
            f"Could not analyze old WAV "
            f"{path}: {e}"
        )

        return None


# ============================================================
# PROCESS ONE SAMPLE
# ============================================================

results = []

raw_descriptions = {}


for _, row in target_df.iterrows():

    sample_index = int(
        row[
            "sample_index"
        ]
    )

    print()
    print("=" * 80)

    print(
        f"SAMPLE {sample_index}"
    )

    print("=" * 80)

    parquet_path = Path(
        str(
            row[
                "parquet_file"
            ]
        )
    )

    parquet_row = int(
        row[
            "parquet_row"
        ]
    )

    print(
        f"Parquet : {parquet_path}"
    )

    print(
        f"Row     : {parquet_row}"
    )

    print(
        f"Source  : {row.get('source', '')}"
    )

    print(
        f"Text    : {str(row.get('text', ''))[:150]}"
    )

    # --------------------------------------------------------
    # Check parquet
    # --------------------------------------------------------

    if not parquet_path.exists():

        print(
            "ERROR: parquet file does not exist!"
        )

        results.append({
            "sample_index": sample_index,
            "status": "PARQUET_MISSING",
            "parquet_file": str(
                parquet_path
            ),
            "parquet_row": parquet_row,
        })

        continue

    # --------------------------------------------------------
    # Read original parquet row
    # --------------------------------------------------------

    parquet_df = load_parquet(
        parquet_path
    )

    if parquet_row >= len(
        parquet_df
    ):

        print(
            "ERROR: parquet row out of range!"
        )

        continue

    parquet_record = parquet_df.iloc[
        parquet_row
    ]

    if "audio" not in parquet_record.index:

        raise RuntimeError(
            "The parquet row has no 'audio' column."
        )

    audio_obj = parquet_record[
        "audio"
    ]

    # --------------------------------------------------------
    # Describe RAW object
    # --------------------------------------------------------

    raw_description = (
        describe_audio_object(
            audio_obj
        )
    )

    raw_descriptions[
        str(sample_index)
    ] = raw_description

    print()
    print(
        "RAW AUDIO OBJECT:"
    )

    print(
        json.dumps(
            raw_description,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # Extract actual encoded audio
    # --------------------------------------------------------

    raw = get_raw_audio_bytes(
        audio_obj
    )

    raw_type = type(
        raw
    ).__name__

    if isinstance(
        raw,
        (bytes, bytearray)
    ):

        raw_length = len(
            raw
        )

        raw_hash = hashlib.sha256(
            bytes(raw)
        ).hexdigest()

    else:

        raw_length = None

        raw_hash = None

    print()
    print(
        f"Extracted object type: "
        f"{raw_type}"
    )

    print(
        f"Encoded byte length  : "
        f"{raw_length}"
    )

    print(
        f"Encoded SHA256       : "
        f"{raw_hash}"
    )

    # --------------------------------------------------------
    # Independent decode
    # --------------------------------------------------------

    try:

        audio, sr, channels, decoder = (
            decode_independently(
                raw
            )
        )

    except Exception as e:

        print()
        print(
            "INDEPENDENT DECODE FAILED:"
        )

        print(
            e
        )

        results.append({
            "sample_index": sample_index,
            "status": "DECODE_FAILED",
            "parquet_file": str(
                parquet_path
            ),
            "parquet_row": parquet_row,
            "decoder_error": str(e),
        })

        continue

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        audio,
        sr
    )

    waveform_hash = audio_hash(
        audio
    )

    print()
    print(
        "INDEPENDENT DECODE:"
    )

    print(
        f"Decoder              : {decoder}"
    )

    print(
        f"Sample rate          : {sr}"
    )

    print(
        f"Channels             : {channels}"
    )

    print(
        f"Samples              : {len(audio)}"
    )

    print(
        f"Duration             : "
        f"{metrics['duration']:.6f}s"
    )

    print(
        f"RMS                  : "
        f"{metrics['rms_db']:.3f} dB"
    )

    print(
        f"Peak                 : "
        f"{metrics['peak_db']:.3f} dB"
    )

    print(
        f"Absolute active      : "
        f"{metrics['absolute_active_ratio']:.6f}"
    )

    print(
        f"Absolute silence     : "
        f"{metrics['absolute_silence_ratio']:.6f}"
    )

    print(
        f"Adaptive active      : "
        f"{metrics['adaptive_active_ratio']:.6f}"
    )

    print(
        f"Adaptive silence     : "
        f"{metrics['adaptive_silence_ratio']:.6f}"
    )

    print(
        f"Zero crossing rate   : "
        f"{metrics['zero_crossing_rate']:.6f}"
    )

    print(
        f"Clipping             : "
        f"{metrics['clipping_pct']:.6f}%"
    )

    print(
        f"Dynamic range        : "
        f"{metrics['dynamic_range_db']:.3f} dB"
    )

    print(
        f"Unique values        : "
        f"{metrics['unique_values_sample']}"
    )

    print(
        f"Waveform SHA256      : "
        f"{waveform_hash}"
    )

    # --------------------------------------------------------
    # Save independent WAV
    # --------------------------------------------------------

    independent_wav = (
        OUTPUT_WAV_DIR
        / f"{sample_index:04d}_RAW_INDEPENDENT.wav"
    )

    save_wav(
        independent_wav,
        audio,
        sr
    )

    print()
    print(
        f"Saved independent WAV:"
    )

    print(
        independent_wav
    )

    # --------------------------------------------------------
    # Compare with old forensic WAV
    # --------------------------------------------------------

    old = analyze_old_wav(
        sample_index
    )

    old_rms_db = np.nan
    old_duration = np.nan
    old_hash = None

    if old is not None:

        old_rms_db = old[
            "metrics"
        ][
            "rms_db"
        ]

        old_duration = old[
            "metrics"
        ][
            "duration"
        ]

        old_hash = old[
            "hash"
        ]

        rms_difference = (
            metrics[
                "rms_db"
            ]
            - old_rms_db
        )

        duration_difference = (
            metrics[
                "duration"
            ]
            - old_duration
        )

        print()
        print(
            "COMPARISON WITH PREVIOUS "
            "FORENSIC WAV:"
        )

        print(
            f"Old RMS             : "
            f"{old_rms_db:.3f} dB"
        )

        print(
            f"New RMS             : "
            f"{metrics['rms_db']:.3f} dB"
        )

        print(
            f"RMS difference      : "
            f"{rms_difference:.6f} dB"
        )

        print(
            f"Old duration        : "
            f"{old_duration:.6f}s"
        )

        print(
            f"New duration        : "
            f"{metrics['duration']:.6f}s"
        )

        print(
            f"Duration difference : "
            f"{duration_difference:.6f}s"
        )

        print(
            f"Old waveform hash   : "
            f"{old_hash}"
        )

        print(
            f"New waveform hash   : "
            f"{waveform_hash}"
        )

        if abs(
            rms_difference
        ) < 0.01 and abs(
            duration_difference
        ) < 0.01:

            comparison = (
                "MATCHES_PREVIOUS_DECODE"
            )

        else:

            comparison = (
                "DIFFERS_FROM_PREVIOUS_DECODE"
            )

    else:

        comparison = (
            "NO_PREVIOUS_WAV_FOUND"
        )

    # --------------------------------------------------------
    # Preliminary source verdict
    # --------------------------------------------------------

    abs_active = (
        metrics[
            "absolute_active_ratio"
        ]
    )

    rms_db = (
        metrics[
            "rms_db"
        ]
    )

    if (
        abs_active < 0.01
        and rms_db < -40
    ):

        source_verdict = (
            "SOURCE_APPEARS_NEAR_SILENT"
        )

    elif (
        abs_active < 0.05
        and rms_db < -30
    ):

        source_verdict = (
            "SOURCE_HAS_VERY_LOW_AUDIO_ACTIVITY"
        )

    else:

        source_verdict = (
            "SOURCE_CONTAINS_SUBSTANTIAL_SIGNAL"
        )

    print()
    print(
        "PRELIMINARY SOURCE VERDICT:"
    )

    print(
        source_verdict
    )

    results.append({
        "sample_index": sample_index,
        "speaker_id": str(
            row[
                "speaker_id"
            ]
        ),
        "source": str(
            row.get(
                "source",
                ""
            )
        ),
        "parquet_file": str(
            parquet_path
        ),
        "parquet_row": parquet_row,

        "manifest_duration": row.get(
            "duration",
            np.nan
        ),

        "text_length": len(
            str(
                row.get(
                    "text",
                    ""
                )
            )
        ),

        "decoder": decoder,

        "sample_rate": sr,
        "channels": channels,

        "encoded_bytes": raw_length,
        "encoded_sha256": raw_hash,

        "waveform_sha256": waveform_hash,

        "duration": metrics[
            "duration"
        ],

        "rms_db": metrics[
            "rms_db"
        ],

        "peak_db": metrics[
            "peak_db"
        ],

        "absolute_active_ratio":
            metrics[
                "absolute_active_ratio"
            ],

        "absolute_silence_ratio":
            metrics[
                "absolute_silence_ratio"
            ],

        "adaptive_active_ratio":
            metrics[
                "adaptive_active_ratio"
            ],

        "adaptive_silence_ratio":
            metrics[
                "adaptive_silence_ratio"
            ],

        "zero_crossing_rate":
            metrics[
                "zero_crossing_rate"
            ],

        "clipping_pct":
            metrics[
                "clipping_pct"
            ],

        "dynamic_range_db":
            metrics[
                "dynamic_range_db"
            ],

        "unique_values":
            metrics[
                "unique_values_sample"
            ],

        "old_wav_rms_db":
            old_rms_db,

        "old_wav_duration":
            old_duration,

        "old_wav_waveform_sha256":
            old_hash,

        "decode_comparison":
            comparison,

        "source_verdict":
            source_verdict,

        "independent_wav":
            str(
                independent_wav
            ),

        "status": "OK",
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_path = (
    OUTPUT_DIR
    / "raw_parquet_verification.csv"
)

results_df.to_csv(
    results_path,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# SAVE RAW OBJECT DESCRIPTIONS
# ============================================================

with open(
    OUTPUT_DIR
    / "raw_audio_objects.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        raw_descriptions,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# FINAL VERDICT
# ============================================================

print()
print("=" * 80)
print("FINAL VERIFICATION SUMMARY")
print("=" * 80)

if len(results_df) > 0:

    summary_columns = [
        "sample_index",
        "decoder",
        "sample_rate",
        "channels",
        "duration",
        "rms_db",
        "peak_db",
        "absolute_active_ratio",
        "absolute_silence_ratio",
        "adaptive_active_ratio",
        "adaptive_silence_ratio",
        "clipping_pct",
        "source_verdict",
        "decode_comparison",
    ]

    summary_columns = [
        c for c in summary_columns
        if c in results_df.columns
    ]

    print(
        results_df[
            summary_columns
        ].to_string(
            index=False
        )
    )


print()
print("=" * 80)
print("INTERPRETATION")
print("=" * 80)

ok_df = results_df[
    results_df[
        "status"
    ] == "OK"
].copy()


if len(ok_df) > 0:

    source_silent = ok_df[
        ok_df[
            "source_verdict"
        ].astype(str)
        .str.startswith(
            "SOURCE_APPEARS_NEAR_SILENT"
        )
    ]

    low_activity = ok_df[
        ok_df[
            "source_verdict"
        ].astype(str)
        .str.startswith(
            "SOURCE_HAS_VERY_LOW"
        )
    ]

    signal = ok_df[
        ok_df[
            "source_verdict"
        ].astype(str)
        .str.startswith(
            "SOURCE_CONTAINS"
        )
    ]

    print(
        f"Near-silent source samples : "
        f"{len(source_silent)}"
    )

    print(
        f"Very-low-activity samples  : "
        f"{len(low_activity)}"
    )

    print(
        f"Substantial-signal samples  : "
        f"{len(signal)}"
    )


print()
print(
    "IMPORTANT:"
)

print(
    "This script directly reads the ORIGINAL parquet audio."
)

print(
    "If the independently decoded audio is also silent, "
    "the evidence points to the SOURCE DATA."
)

print(
    "If the independently decoded audio contains speech "
    "while the previous WAV was silent, the problem is "
    "likely in our previous decoding path."
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "Independent WAV files:"
)

print(
    OUTPUT_WAV_DIR
)

print()
print("=" * 80)
print("DONE")
print("=" * 80)