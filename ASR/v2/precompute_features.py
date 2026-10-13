"""
V2 OFFLINE Wav2Vec2-BERT FEATURE PRECOMPUTATION
================================================

Purpose
-------
Precompute frozen Wav2Vec2-BERT 2.0 features for:

    TRAIN: 10,199 samples
    VAL:    1,207 samples

The TEST set (601 samples) is deliberately excluded.

Pipeline
--------
    WAV
      |
      v
    Audio loading / resampling
      |
      v
    AutoFeatureExtractor
      |
      v
    Wav2Vec2-BERT 2.0
      |
      v
    BF16 features
      |
      v
    Convert to FP16
      |
      v
    NVMe feature cache


Hardware
--------
    NVIDIA RTX 2050
    4 GB VRAM

Important
---------
    - Encoder is frozen.
    - One audio sample is processed at a time.
    - No CSV is modified.
    - No dataset sample is removed.
    - Test set is never processed.
    - Existing valid cache entries are reused.
    - Interrupted runs can resume.
    - Cache writes use the existing cache implementation.
    - Features are stored as FP16.
    - Encoder is unloaded when precomputation finishes.

Expected architecture after this stage
---------------------------------------

    NVMe FP16 Feature Cache
              |
              v
      Audio Projector
        1024 -> 2048
              |
              v
       Qwen2.5-3B QLoRA
              |
              v
         Meitei Mayek
"""

from __future__ import annotations

import gc
import inspect
import json
import time
from datetime import datetime
from pathlib import Path

import soundfile as sf
import torch

from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2BertModel,
)

try:
    import torchaudio
except ImportError as exc:
    raise RuntimeError(
        "torchaudio is required for proper anti-aliased "
        "audio resampling."
    ) from exc


# ============================================================================
# V2 LOCAL IMPORTS
# ============================================================================

try:
    from .data_manager import DataManager
    from .feature_cache import (
        FrozenEncoderFeatureCache,
        FeatureCacheConfig,
    )
except ImportError:
    from data_manager import DataManager
    from feature_cache import (
        FrozenEncoderFeatureCache,
        FeatureCacheConfig,
    )


# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_NAME = "facebook/w2v-bert-2.0"

TARGET_SAMPLE_RATE = 16000

EXPECTED_ENCODER_DIM = 1024

# Match the successful benchmark configuration.
ENCODER_DTYPE = torch.bfloat16

# Disk cache precision.
DISK_DTYPE = "float16"

# Process one sample at a time.
BATCH_SIZE = 1

# Print/save progress every N successfully processed samples.
PROGRESS_INTERVAL = 100

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

V2_DIR = Path(__file__).resolve().parent

CACHE_DIR = (
    V2_DIR
    / "cache"
    / "features"
)

LOG_DIR = (
    V2_DIR
    / "logs"
)

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# Main progress state.
STATE_FILE = (
    LOG_DIR
    / "precompute_state.json"
)

# Detailed per-sample telemetry.
TELEMETRY_FILE = (
    LOG_DIR
    / "precompute_telemetry.jsonl"
)

# Final diagnostic.
FINAL_REPORT_FILE = (
    LOG_DIR
    / "precompute_features_report.json"
)


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================================
# CUDA MEMORY
# ============================================================================

def gpu_memory() -> dict:
    """
    Return current CUDA memory statistics in MB.
    """

    if not torch.cuda.is_available():

        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_allocated_mb": 0.0,
            "peak_reserved_mb": 0.0,
            "free_mb": 0.0,
            "total_mb": 0.0,
        }

    torch.cuda.synchronize()

    free_bytes, total_bytes = (
        torch.cuda.mem_get_info()
    )

    return {
        "allocated_mb": (
            torch.cuda.memory_allocated()
            / (1024 ** 2)
        ),
        "reserved_mb": (
            torch.cuda.memory_reserved()
            / (1024 ** 2)
        ),
        "peak_allocated_mb": (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        ),
        "peak_reserved_mb": (
            torch.cuda.max_memory_reserved()
            / (1024 ** 2)
        ),
        "free_mb": (
            free_bytes
            / (1024 ** 2)
        ),
        "total_mb": (
            total_bytes
            / (1024 ** 2)
        ),
    }


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def cleanup_memory() -> None:
    """
    Release temporary Python/CUDA memory.

    This does NOT unload the encoder.
    """

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()


# ============================================================================
# AUDIO
# ============================================================================

def resample_audio(
    waveform: torch.Tensor,
    source_sample_rate: int,
) -> torch.Tensor:
    """
    Anti-aliased resampling using torchaudio.
    """

    if source_sample_rate == TARGET_SAMPLE_RATE:
        return waveform.contiguous()

    resampler = torchaudio.transforms.Resample(
        orig_freq=source_sample_rate,
        new_freq=TARGET_SAMPLE_RATE,
    )

    waveform = resampler(
        waveform.unsqueeze(0)
    ).squeeze(0)

    return waveform.contiguous()


def load_audio(path: Path) -> torch.Tensor:
    """
    Load audio as mono float32 at 16 kHz.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Audio file does not exist:\n{path}"
        )

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
    )

    waveform = torch.from_numpy(audio)

    # Stereo/multichannel -> mono.
    if waveform.ndim == 2:

        waveform = waveform.mean(
            dim=1
        )

    waveform = waveform.contiguous()

    if sample_rate != TARGET_SAMPLE_RATE:

        waveform = resample_audio(
            waveform,
            int(sample_rate),
        )

    return waveform.contiguous()


# ============================================================================
# ATOMIC JSON
# ============================================================================

def atomic_write_json(
    path: Path,
    data: dict,
) -> None:
    """
    Atomically write JSON.

    Prevents a partially written state file if the process
    is interrupted.
    """

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
        )

        f.flush()

    temporary.replace(path)


# ============================================================================
# TELEMETRY
# ============================================================================

def append_telemetry(record: dict) -> None:
    """
    Append one JSONL telemetry record immediately.
    """

    with open(
        TELEMETRY_FILE,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

        f.flush()


# ============================================================================
# MODEL INPUT
# ============================================================================

def prepare_model_inputs(
    feature_extractor,
    waveform: torch.Tensor,
    model,
) -> dict:
    """
    Convert CPU waveform into CUDA model inputs.

    Floating tensors use BF16.
    Integer tensors such as attention masks remain integer.
    """

    processed = feature_extractor(
        waveform.numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )

    model_signature = inspect.signature(
        model.forward
    )

    model_inputs = {}

    for key, value in processed.items():

        if (
            key
            not in model_signature.parameters
        ):
            continue

        if not isinstance(
            value,
            torch.Tensor,
        ):
            continue

        if value.dtype.is_floating_point:

            value = value.to(
                DEVICE,
                dtype=ENCODER_DTYPE,
                non_blocking=True,
            )

        else:

            value = value.to(
                DEVICE,
                non_blocking=True,
            )

        model_inputs[key] = value

    if not model_inputs:

        raise RuntimeError(
            "Feature extractor produced no "
            "model-compatible inputs.\n"
            f"Extractor keys: "
            f"{list(processed.keys())}\n"
            f"Model parameters: "
            f"{list(model_signature.parameters.keys())}"
        )

    return model_inputs


# ============================================================================
# ENCODER
# ============================================================================

def encode_sample(
    model,
    feature_extractor,
    waveform: torch.Tensor,
):
    """
    Run one frozen encoder inference pass.

    Returns:
        features
        encoder_time
        model_inputs
    """

    model_inputs = prepare_model_inputs(
        feature_extractor,
        waveform,
        model,
    )

    reset_peak_memory()

    synchronize()

    start = time.perf_counter()

    with torch.no_grad():

        outputs = model(
            **model_inputs
        )

    synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    if hasattr(
        outputs,
        "last_hidden_state",
    ):

        features = (
            outputs.last_hidden_state
        )

    elif isinstance(
        outputs,
        dict,
    ):

        features = outputs[
            "last_hidden_state"
        ]

    else:

        raise RuntimeError(
            "Could not obtain "
            "last_hidden_state."
        )

    features = features.detach()

    # ------------------------------------------------------------
    # Validate shape
    # ------------------------------------------------------------

    if features.ndim != 3:

        raise RuntimeError(
            "Unexpected encoder output rank.\n"
            f"Shape: {tuple(features.shape)}"
        )

    if features.shape[0] != 1:

        raise RuntimeError(
            "Expected batch size 1.\n"
            f"Shape: {tuple(features.shape)}"
        )

    if features.shape[-1] != EXPECTED_ENCODER_DIM:

        raise RuntimeError(
            "Unexpected encoder hidden dimension.\n"
            f"Expected: {EXPECTED_ENCODER_DIM}\n"
            f"Received: {features.shape[-1]}"
        )

    if not torch.isfinite(
        features
    ).all():

        raise RuntimeError(
            "Encoder produced NaN or Inf values."
        )

    return (
        features,
        elapsed,
        model_inputs,
    )


# ============================================================================
# CACHE VALIDATION
# ============================================================================

def get_cache_key(
    cache,
    record,
    fingerprint,
):
    return cache.build_key(
        record,
        fingerprint,
    )


def cache_contains_valid_feature(
    cache,
    cache_key,
) -> tuple[bool, object | None]:
    """
    Check whether a valid feature already exists.

    Returns:
        (True, feature) if valid
        (False, None) otherwise
    """

    try:

        cached = cache._disk_get(
            cache_key
        )

        if cached is None:
            return False, None

        if not isinstance(
            cached,
            torch.Tensor,
        ):
            return False, None

        if cached.ndim != 3:
            return False, None

        if cached.shape[0] != 1:
            return False, None

        if (
            cached.shape[-1]
            != EXPECTED_ENCODER_DIM
        ):
            return False, None

        if not torch.isfinite(
            cached
        ).all():
            return False, None

        return True, cached

    except Exception:

        # Corrupt/incompatible cache entries are treated
        # as cache misses and regenerated.
        return False, None


# ============================================================================
# STATE
# ============================================================================

def initial_state(
    train_count: int,
    val_count: int,
    fingerprint: str,
) -> dict:

    return {
        "status": "running",

        "started_at": (
            datetime.now().isoformat()
        ),

        "updated_at": (
            datetime.now().isoformat()
        ),

        "model_name": MODEL_NAME,

        "encoder_dtype": str(
            ENCODER_DTYPE
        ),

        "disk_dtype": DISK_DTYPE,

        "target_sample_rate": (
            TARGET_SAMPLE_RATE
        ),

        "expected_encoder_dim": (
            EXPECTED_ENCODER_DIM
        ),

        "fingerprint": fingerprint,

        "train_total": train_count,

        "val_total": val_count,

        "test_total": 601,

        "train_completed": 0,

        "val_completed": 0,

        "cache_hits": 0,

        "new_features": 0,

        "failures": 0,

        "total_encoder_seconds": 0.0,

        "total_wall_seconds": 0.0,

        "last_dataset": None,

        "last_index": None,

        "last_duration_sec": None,

        "last_error": None,
    }


# ============================================================================
# PROCESS SPLIT
# ============================================================================

def process_records(
    dataset_name: str,
    records,
    manager,
    model,
    feature_extractor,
    cache,
    fingerprint: str,
    state: dict,
) -> dict:
    """
    Process one dataset split.

    Only train and val are allowed.
    """

    if dataset_name not in {
        "train",
        "val",
    }:

        raise RuntimeError(
            f"Unsafe dataset name: "
            f"{dataset_name}"
        )

    total = len(records)

    completed = 0

    split_encoder_time = 0.0

    split_start = (
        time.perf_counter()
    )

    print(
        f"\n{'=' * 72}"
    )

    print(
        f"PRECOMPUTING {dataset_name.upper()}"
    )

    print(
        f"{'=' * 72}"
    )

    print(
        f"Samples: {total:,}"
    )

    for position, record in enumerate(
        records,
        start=1,
    ):

        sample_start = (
            time.perf_counter()
        )

        cache_key = get_cache_key(
            cache,
            record,
            fingerprint,
        )

        # ------------------------------------------------------------
        # Existing cache
        # ------------------------------------------------------------

        valid_cache, cached_feature = (
            cache_contains_valid_feature(
                cache,
                cache_key,
            )
        )

        if valid_cache:

            completed += 1

            state[
                "cache_hits"
            ] += 1

            state[
                f"{dataset_name}_completed"
            ] += 1

            state[
                "last_dataset"
            ] = dataset_name

            state[
                "last_index"
            ] = int(record.index)

            state[
                "last_duration_sec"
            ] = float(
                record.duration_sec
            )

            state[
                "last_error"
            ] = None

            wall_time = (
                time.perf_counter()
                - sample_start
            )

            append_telemetry(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),
                    "dataset": dataset_name,
                    "position": position,
                    "total": total,
                    "record_index": int(
                        record.index
                    ),
                    "duration_sec": float(
                        record.duration_sec
                    ),
                    "status": "cache_hit",
                    "encoder_time_sec": 0.0,
                    "wall_time_sec": wall_time,
                    "feature_shape": list(
                        cached_feature.shape
                    ),
                    "feature_dtype": str(
                        cached_feature.dtype
                    ),
                }
            )

            del cached_feature

            if (
                position == 1
                or position % PROGRESS_INTERVAL == 0
                or position == total
            ):

                elapsed = (
                    time.perf_counter()
                    - split_start
                )

                rate = (
                    completed / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"[{position:5d}/{total:5d}] "
                    f"{dataset_name.upper():5s} | "
                    f"CACHE HIT | "
                    f"{rate:.2f} samples/s"
                )

            continue

        # ------------------------------------------------------------
        # Cache miss -> encode
        # ------------------------------------------------------------

        try:

            waveform = load_audio(
                Path(record.audio_path)
            )

            actual_duration = (
                waveform.numel()
                / TARGET_SAMPLE_RATE
            )

            (
                features,
                encoder_time,
                model_inputs,
            ) = encode_sample(
                model,
                feature_extractor,
                waveform,
            )

            # --------------------------------------------------------
            # Convert encoder BF16 -> FP16 for disk
            # --------------------------------------------------------

            disk_features = (
                features
                .detach()
                .to(
                    dtype=torch.float16,
                    device="cpu",
                )
                .contiguous()
            )

            # --------------------------------------------------------
            # Validate converted feature
            # --------------------------------------------------------

            if disk_features.ndim != 3:

                raise RuntimeError(
                    "Converted feature rank "
                    "is invalid."
                )

            if (
                disk_features.shape[-1]
                != EXPECTED_ENCODER_DIM
            ):

                raise RuntimeError(
                    "Converted feature hidden "
                    "dimension is invalid."
                )

            if not torch.isfinite(
                disk_features
            ).all():

                raise RuntimeError(
                    "Converted feature contains "
                    "NaN or Inf."
                )

            # --------------------------------------------------------
            # Disk cache
            # --------------------------------------------------------

            disk_before = (
                cache.disk_size_bytes()
            )

            cache._disk_put(
                cache_key,
                disk_features,
                metadata={
                    "dataset": dataset_name,

                    "record_index": int(
                        record.index
                    ),

                    "duration_sec": float(
                        record.duration_sec
                    ),

                    "actual_loaded_duration_sec": (
                        float(actual_duration)
                    ),

                    "sample_rate": (
                        TARGET_SAMPLE_RATE
                    ),

                    "encoder_model": (
                        MODEL_NAME
                    ),

                    "encoder_dtype": (
                        str(ENCODER_DTYPE)
                    ),

                    "disk_dtype": (
                        DISK_DTYPE
                    ),

                    "feature_shape": list(
                        disk_features.shape
                    ),

                    "feature_hidden_size": (
                        EXPECTED_ENCODER_DIM
                    ),

                    "precompute_version": (
                        "1.0"
                    ),
                },
            )

            disk_after = (
                cache.disk_size_bytes()
            )

            disk_delta = (
                disk_after
                - disk_before
            )

            # --------------------------------------------------------
            # Read-back validation
            # --------------------------------------------------------

            cache.clear_ram()

            disk_check = (
                cache._disk_get(
                    cache_key
                )
            )

            if disk_check is None:

                raise RuntimeError(
                    "Cache write succeeded but "
                    "read-back returned None."
                )

            if (
                tuple(disk_check.shape)
                != tuple(
                    disk_features.shape
                )
            ):

                raise RuntimeError(
                    "Cache read-back shape mismatch."
                )

            if not torch.isfinite(
                disk_check
            ).all():

                raise RuntimeError(
                    "Cache read-back contains "
                    "NaN or Inf."
                )

            # --------------------------------------------------------
            # Accounting
            # --------------------------------------------------------

            completed += 1

            state[
                "new_features"
            ] += 1

            state[
                f"{dataset_name}_completed"
            ] += 1

            state[
                "total_encoder_seconds"
            ] += encoder_time

            split_encoder_time += (
                encoder_time
            )

            state[
                "last_dataset"
            ] = dataset_name

            state[
                "last_index"
            ] = int(record.index)

            state[
                "last_duration_sec"
            ] = float(
                record.duration_sec
            )

            state[
                "last_error"
            ] = None

            wall_time = (
                time.perf_counter()
                - sample_start
            )

            state[
                "total_wall_seconds"
            ] += wall_time

            append_telemetry(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),

                    "dataset": dataset_name,

                    "position": position,

                    "total": total,

                    "record_index": int(
                        record.index
                    ),

                    "duration_sec": float(
                        record.duration_sec
                    ),

                    "actual_loaded_duration_sec": (
                        float(actual_duration)
                    ),

                    "status": "computed",

                    "encoder_time_sec": (
                        float(encoder_time)
                    ),

                    "wall_time_sec": (
                        float(wall_time)
                    ),

                    "feature_shape": list(
                        disk_features.shape
                    ),

                    "feature_dtype": str(
                        disk_features.dtype
                    ),

                    "feature_size_mb": (
                        disk_features.numel()
                        * disk_features.element_size()
                        / (1024 ** 2)
                    ),

                    "disk_delta_mb": (
                        disk_delta
                        / (1024 ** 2)
                    ),

                    "vram_peak_mb": (
                        gpu_memory()[
                            "peak_allocated_mb"
                        ]
                    ),

                    "vram_free_mb": (
                        gpu_memory()[
                            "free_mb"
                        ]
                    ),
                }
            )

            # --------------------------------------------------------
            # Cleanup
            # --------------------------------------------------------

            del disk_check
            del disk_features
            del features
            del waveform
            del model_inputs

            cleanup_memory()

        except Exception as exc:

            state[
                "failures"
            ] += 1

            state[
                "last_dataset"
            ] = dataset_name

            state[
                "last_index"
            ] = int(record.index)

            state[
                "last_duration_sec"
            ] = float(
                record.duration_sec
            )

            state[
                "last_error"
            ] = str(exc)

            append_telemetry(
                {
                    "timestamp": (
                        datetime.now().isoformat()
                    ),

                    "dataset": dataset_name,

                    "position": position,

                    "total": total,

                    "record_index": int(
                        record.index
                    ),

                    "duration_sec": float(
                        record.duration_sec
                    ),

                    "status": "failure",

                    "error": str(exc),
                }
            )

            print(
                f"\n!!! FAILURE !!!"
            )

            print(
                f"Dataset: {dataset_name}"
            )

            print(
                f"Position: "
                f"{position}/{total}"
            )

            print(
                f"Index: "
                f"{record.index}"
            )

            print(
                f"Duration: "
                f"{record.duration_sec:.3f}s"
            )

            print(
                f"Error: {exc}"
            )

            # --------------------------------------------------------
            # Important:
            #
            # We continue rather than destroying the whole
            # precomputation run.
            # --------------------------------------------------------

            cleanup_memory()

        # ------------------------------------------------------------
        # Progress reporting
        # ------------------------------------------------------------

        if (
            position == 1
            or position % PROGRESS_INTERVAL == 0
            or position == total
        ):

            elapsed = (
                time.perf_counter()
                - split_start
            )

            rate = (
                completed / elapsed
                if elapsed > 0
                else 0.0
            )

            remaining = (
                total - position
            )

            eta_seconds = (
                remaining / rate
                if rate > 0
                else 0.0
            )

            memory = gpu_memory()

            print(
                f"[{position:5d}/{total:5d}] "
                f"{dataset_name.upper():5s} | "
                f"completed={completed:5d} | "
                f"new={state['new_features']:5d} | "
                f"hits={state['cache_hits']:5d} | "
                f"fail={state['failures']:3d} | "
                f"rate={rate:.2f}/s | "
                f"ETA={eta_seconds / 60:.1f}m | "
                f"VRAM free={memory['free_mb']:.0f}MB"
            )

            state[
                "updated_at"
            ] = datetime.now().isoformat()

            atomic_write_json(
                STATE_FILE,
                state,
            )

    split_wall_time = (
        time.perf_counter()
        - split_start
    )

    print(
        f"\n{dataset_name.upper()} COMPLETE"
    )

    print(
        f"  Total samples: {total:,}"
    )

    print(
        f"  Completed: {completed:,}"
    )

    print(
        f"  Encoder time: "
        f"{split_encoder_time:.2f}s"
    )

    print(
        f"  Wall time: "
        f"{split_wall_time:.2f}s"
    )

    print(
        f"  Failures: "
        f"{state['failures']:,}"
    )

    return state


# ============================================================================
# FINAL CACHE VALIDATION
# ============================================================================

def validate_split_cache(
    dataset_name: str,
    records,
    cache,
    fingerprint,
) -> dict:
    """
    Validate every expected cache entry for train or val.

    This does NOT process the encoder again.
    It only checks the disk cache.
    """

    total = len(records)

    valid = 0

    invalid = 0

    print(
        f"\nValidating {dataset_name.upper()} cache..."
    )

    for position, record in enumerate(
        records,
        start=1,
    ):

        cache_key = cache.build_key(
            record,
            fingerprint,
        )

        cache_path = cache.cache_path(
            cache_key
        )

        if not cache_path.exists():

            invalid += 1

        else:

            try:

                cached = cache._disk_get(
                    cache_key
                )

                if (
                    cached is not None
                    and cached.ndim == 3
                    and cached.shape[0] == 1
                    and cached.shape[-1]
                    == EXPECTED_ENCODER_DIM
                    and torch.isfinite(
                        cached
                    ).all()
                ):

                    valid += 1

                else:

                    invalid += 1

                del cached

            except Exception:

                invalid += 1

        if (
            position == 1
            or position % PROGRESS_INTERVAL == 0
            or position == total
        ):

            print(
                f"  [{position:5d}/{total:5d}] "
                f"valid={valid:5d} "
                f"invalid={invalid:3d}"
            )

    return {
        "dataset": dataset_name,
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "complete": (
            valid == total
            and invalid == 0
        ),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 72)
    print(
        "V2 OFFLINE Wav2Vec2-BERT "
        "FEATURE PRECOMPUTATION"
    )
    print("=" * 72)

    # ========================================================================
    # STEP 1: CUDA
    # ========================================================================

    print("\n[1/8] Hardware check...")

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is unavailable.\n"
            "This precomputation utility is "
            "designed to use the RTX 2050."
        )

    gpu_name = (
        torch.cuda.get_device_name(0)
    )

    memory = gpu_memory()

    print(
        f"GPU: {gpu_name}"
    )

    print(
        f"CUDA device: "
        f"{torch.cuda.current_device()}"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        f"Encoder dtype: "
        f"{ENCODER_DTYPE}"
    )

    print(
        f"Initial free VRAM: "
        f"{memory['free_mb']:.1f} MB"
    )

    # ========================================================================
    # STEP 2: DATA MANAGER
    # ========================================================================

    print(
        "\n[2/8] Loading Data Manager..."
    )

    manager = DataManager()

    train_records = manager.train
    val_records = manager.val
    test_records = manager.test

    print(
        f"TRAIN: {len(train_records):,}"
    )

    print(
        f"VAL:   {len(val_records):,}"
    )

    print(
        f"TEST:  {len(test_records):,}"
    )

    # Exact dataset protection.
    if len(train_records) != 10199:

        raise RuntimeError(
            "TRAIN SET SIZE CHANGED.\n"
            f"Expected: 10199\n"
            f"Found: {len(train_records)}"
        )

    if len(val_records) != 1207:

        raise RuntimeError(
            "VAL SET SIZE CHANGED.\n"
            f"Expected: 1207\n"
            f"Found: {len(val_records)}"
        )

    if len(test_records) != 601:

        raise RuntimeError(
            "TEST SET SIZE CHANGED.\n"
            f"Expected: 601\n"
            f"Found: {len(test_records)}"
        )

    print(
        "Dataset counts: PASS"
    )

    print(
        "Test-set protection: "
        "PASS (601 samples)"
    )

    # ========================================================================
    # STEP 3: MODEL
    # ========================================================================

    print(
        "\n[3/8] Loading feature extractor..."
    )

    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    print(
        "Feature extractor: PASS"
    )

    # ========================================================================
    # STEP 4: ENCODER
    # ========================================================================

    print(
        "\n[4/8] Loading frozen Wav2Vec2-BERT..."
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Dtype: {ENCODER_DTYPE}"
    )

    model = (
        Wav2Vec2BertModel.from_pretrained(
            MODEL_NAME,
            dtype=ENCODER_DTYPE,
        )
        .to(DEVICE)
    )

    model.eval()

    for parameter in model.parameters():

        parameter.requires_grad = False

    synchronize()

    after_model = gpu_memory()

    print(
        f"VRAM after encoder load: "
        f"{after_model['allocated_mb']:.1f} MB"
    )

    print(
        f"VRAM free: "
        f"{after_model['free_mb']:.1f} MB"
    )

    hidden_size = getattr(
        model.config,
        "output_hidden_size",
        None,
    )

    if hidden_size is None:

        hidden_size = getattr(
            model.config,
            "hidden_size",
            None,
        )

    print(
        f"Encoder output dimension: "
        f"{hidden_size}"
    )

    if hidden_size != EXPECTED_ENCODER_DIM:

        raise RuntimeError(
            "Encoder output dimension mismatch.\n"
            f"Expected: {EXPECTED_ENCODER_DIM}\n"
            f"Found: {hidden_size}"
        )

    print(
        "Encoder configuration: PASS"
    )

    # ========================================================================
    # STEP 5: CACHE
    # ========================================================================

    print(
        "\n[5/8] Initializing feature cache..."
    )

    cache = FrozenEncoderFeatureCache(
        FeatureCacheConfig(
            cache_dir=str(
                CACHE_DIR
            ),
            max_ram_items=2,
            disk_dtype=DISK_DTYPE,
            enable_ram_cache=True,
            enable_disk_cache=True,
        )
    )

    # IMPORTANT:
    #
    # This fingerprint must remain consistent with the benchmark
    # so that the five benchmark samples can be reused.
    fingerprint = cache.build_fingerprint(
        model_name=MODEL_NAME,
        sampling_rate=TARGET_SAMPLE_RATE,
        feature_extractor_config={
            "sampling_rate": (
                TARGET_SAMPLE_RATE
            ),
            "encoder_dtype": str(
                ENCODER_DTYPE
            ),
            "output_hidden_size": (
                EXPECTED_ENCODER_DIM
            ),
        },
    )

    print(
        f"Cache directory: "
        f"{CACHE_DIR}"
    )

    print(
        f"Fingerprint: "
        f"{fingerprint}"
    )

    # ========================================================================
    # STEP 6: STATE
    # ========================================================================

    print(
        "\n[6/8] Preparing resumable state..."
    )

    state = initial_state(
        train_count=len(
            train_records
        ),
        val_count=len(
            val_records
        ),
        fingerprint=fingerprint,
    )

    # Existing state is informational only.
    #
    # We deliberately do NOT trust previous counters to skip samples.
    # The actual cache contents are authoritative.
    if STATE_FILE.exists():

        try:

            with open(
                STATE_FILE,
                "r",
                encoding="utf-8",
            ) as f:

                previous_state = (
                    json.load(f)
                )

            if (
                previous_state.get(
                    "fingerprint"
                )
                == fingerprint
            ):

                print(
                    "Previous compatible "
                    "precompute state found."
                )

                print(
                    "Existing valid cache entries "
                    "will be reused."
                )

            else:

                print(
                    "Previous state uses a "
                    "different fingerprint."
                )

                print(
                    "Cache validation will determine "
                    "what must be recomputed."
                )

        except Exception as exc:

            print(
                "Previous state could not be read:"
                f" {exc}"
            )

    atomic_write_json(
        STATE_FILE,
        state,
    )

    # ========================================================================
    # STEP 7: PRECOMPUTE
    # ========================================================================

    print(
        "\n[7/8] Precomputing TRAIN + VAL..."
    )

    print(
        "IMPORTANT:"
    )

    print(
        "  TRAIN = 10,199 samples"
    )

    print(
        "  VAL   = 1,207 samples"
    )

    print(
        "  TEST  = 601 samples -> EXCLUDED"
    )

    print(
        "  Batch size = 1"
    )

    print(
        "  Encoder = frozen"
    )

    print(
        "  Disk features = FP16"
    )

    overall_start = (
        time.perf_counter()
    )

    # ------------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------------

    state = process_records(
        dataset_name="train",
        records=train_records,
        manager=manager,
        model=model,
        feature_extractor=feature_extractor,
        cache=cache,
        fingerprint=fingerprint,
        state=state,
    )

    # ------------------------------------------------------------------------
    # VAL
    # ------------------------------------------------------------------------

    state = process_records(
        dataset_name="val",
        records=val_records,
        manager=manager,
        model=model,
        feature_extractor=feature_extractor,
        cache=cache,
        fingerprint=fingerprint,
        state=state,
    )

    overall_wall_time = (
        time.perf_counter()
        - overall_start
    )

    state[
        "total_wall_seconds"
    ] = overall_wall_time

    # ========================================================================
    # STEP 8: FINAL VALIDATION
    # ========================================================================

    print(
        "\n[8/8] Performing final cache validation..."
    )

    train_validation = (
        validate_split_cache(
            "train",
            train_records,
            cache,
            fingerprint,
        )
    )

    val_validation = (
        validate_split_cache(
            "val",
            val_records,
            cache,
            fingerprint,
        )
    )

    # Explicit test-set exclusion check.
    #
    # We do NOT generate cache keys for test records.
    #
    # This ensures the precomputation workflow never intentionally
    # processes the protected test set.
    test_validation = {
        "dataset": "test",
        "total": len(test_records),
        "processed": 0,
        "excluded": True,
        "reason": (
            "Test set is reserved exclusively "
            "for final evaluation."
        ),
    }

    # ========================================================================
    # FINAL REPORT
    # ========================================================================

    final_memory = gpu_memory()

    total_expected = (
        len(train_records)
        + len(val_records)
    )

    total_valid = (
        train_validation["valid"]
        + val_validation["valid"]
    )

    total_invalid = (
        train_validation["invalid"]
        + val_validation["invalid"]
    )

    complete = (
        train_validation["complete"]
        and val_validation["complete"]
        and total_invalid == 0
        and len(test_records) == 601
    )

    state[
        "status"
    ] = (
        "complete"
        if complete
        else "incomplete"
    )

    state[
        "updated_at"
    ] = datetime.now().isoformat()

    state[
        "completed_at"
    ] = datetime.now().isoformat()

    state[
        "final_train_valid"
    ] = train_validation["valid"]

    state[
        "final_val_valid"
    ] = val_validation["valid"]

    state[
        "final_invalid"
    ] = total_invalid

    atomic_write_json(
        STATE_FILE,
        state,
    )

    report = {
        "status": (
            "COMPLETE"
            if complete
            else "INCOMPLETE"
        ),

        "timestamp": (
            datetime.now().isoformat()
        ),

        "model": {
            "name": MODEL_NAME,
            "encoder": (
                "Wav2Vec2BertModel"
            ),
            "feature_extractor": (
                "AutoFeatureExtractor"
            ),
            "encoder_dtype": str(
                ENCODER_DTYPE
            ),
            "disk_dtype": DISK_DTYPE,
            "output_hidden_size": (
                EXPECTED_ENCODER_DIM
            ),
        },

        "hardware": {
            "gpu": gpu_name,
            "device": str(DEVICE),
            "torch_version": (
                torch.__version__
            ),
            "initial_memory": (
                memory
            ),
            "after_encoder_load": (
                after_model
            ),
            "final_memory": (
                final_memory
            ),
        },

        "dataset": {
            "train": len(
                train_records
            ),
            "val": len(
                val_records
            ),
            "test": len(
                test_records
            ),
            "expected_train": 10199,
            "expected_val": 1207,
            "expected_test": 601,
        },

        "protection": {
            "csv_modified": False,
            "samples_removed": False,
            "test_processed": False,
            "test_excluded": True,
        },

        "cache": {
            "directory": str(
                CACHE_DIR
            ),
            "fingerprint": fingerprint,
            "disk_size_mb": (
                cache.disk_size_bytes()
                / (1024 ** 2)
            ),
            "statistics": (
                cache.statistics()
            ),
        },

        "precomputation": {
            "expected_features": (
                total_expected
            ),
            "valid_features": (
                total_valid
            ),
            "invalid_features": (
                total_invalid
            ),
            "new_features_written": (
                state["new_features"]
            ),
            "cache_hits": (
                state["cache_hits"]
            ),
            "failures": (
                state["failures"]
            ),
            "total_encoder_seconds": (
                state[
                    "total_encoder_seconds"
                ]
            ),
            "total_wall_seconds": (
                overall_wall_time
            ),
        },

        "validation": {
            "train": train_validation,
            "val": val_validation,
            "test": test_validation,
            "all_train_valid": (
                train_validation[
                    "complete"
                ]
            ),
            "all_val_valid": (
                val_validation[
                    "complete"
                ]
            ),
            "overall_complete": complete,
        },
    }

    atomic_write_json(
        FINAL_REPORT_FILE,
        report,
    )

    # ========================================================================
    # CLEANUP MODEL
    # ========================================================================

    print(
        "\nReleasing Wav2Vec2-BERT..."
    )

    del model
    del feature_extractor

    cleanup_memory()

    final_after_unload = gpu_memory()

    # ========================================================================
    # FINAL OUTPUT
    # ========================================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "V2 FEATURE PRECOMPUTATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTRAIN:"
        f" {train_validation['valid']:,}"
        f" / {train_validation['total']:,}"
        f" valid"
    )

    print(
        f"VAL:"
        f"   {val_validation['valid']:,}"
        f" / {val_validation['total']:,}"
        f" valid"
    )

    print(
        f"TEST:"
        f"  {len(test_records):,}"
        f" / 601 protected and excluded"
    )

    print(
        f"\nNew features written:"
        f" {state['new_features']:,}"
    )

    print(
        f"Existing cache hits:"
        f" {state['cache_hits']:,}"
    )

    print(
        f"Failures:"
        f" {state['failures']:,}"
    )

    print(
        f"Invalid cache entries:"
        f" {total_invalid:,}"
    )

    print(
        f"Total cache size:"
        f" {report['cache']['disk_size_mb']:.2f} MB"
    )

    print(
        f"Total wall time:"
        f" {overall_wall_time / 60:.2f} minutes"
    )

    print(
        f"\nVRAM after encoder unload:"
    )

    print(
        f"  Allocated:"
        f" {final_after_unload['allocated_mb']:.1f} MB"
    )

    print(
        f"  Reserved:"
        f" {final_after_unload['reserved_mb']:.1f} MB"
    )

    print(
        f"  Free:"
        f" {final_after_unload['free_mb']:.1f} MB"
    )

    print(
        f"\nReport:"
        f"\n{FINAL_REPORT_FILE}"
    )

    print(
        f"\nState:"
        f"\n{STATE_FILE}"
    )

    print(
        f"\nTelemetry:"
        f"\n{TELEMETRY_FILE}"
    )

    print(
        "\n" + "=" * 72
    )

    if complete:

        print(
            "STATUS: PASS"
        )

        print(
            "TRAIN + VAL feature cache is complete."
        )

        print(
            "TEST set remains protected."
        )

        print(
            "Ready for cached V2 QLoRA integration."
        )

    else:

        print(
            "STATUS: INCOMPLETE"
        )

        print(
            "Some features still require "
            "precomputation or repair."
        )

        print(
            "Review the final report and "
            "telemetry before training."
        )

    print(
        "=" * 72
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()