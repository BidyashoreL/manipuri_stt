"""
V2 REAL Wav2Vec2-BERT Feature Cache Benchmark
===============================================

Purpose
-------
Benchmark the frozen Wav2Vec2-BERT 2.0 encoder independently before
integrating feature caching into the V2 training pipeline.

Architecture:

    Audio
      |
      v
    Wav2Vec2-BERT 2.0
      |
      v
    FP16/BF16 hidden features
      |
      v
    NVMe feature cache


Hardware target
---------------
    NVIDIA RTX 2050
    4 GB VRAM

Representative samples
----------------------
    LOW       ~2.5 sec
    NORMAL    ~4.2 sec
    MEDIUM    ~7.0 sec
    HIGH      ~10.0 sec
    EXTREME   ~18.0 sec

Measurements
------------
    - encoder load VRAM
    - encoder inference time
    - total first-pass time
    - output feature shape
    - feature tensor size
    - peak VRAM
    - remaining VRAM
    - disk cache write size
    - disk cache read time

Safety
------
    This script does NOT:
        - load Qwen
        - perform QLoRA training
        - modify train.csv
        - modify val.csv
        - modify test.csv
        - remove dataset samples

The benchmark is isolated so that we can determine whether
Wav2Vec2-BERT should remain on GPU or be moved to a CPU/NVMe
feature-cache pipeline during actual training.
"""

from __future__ import annotations

import gc
import inspect
import json
import time
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
        "torchaudio is required by this benchmark for proper "
        "anti-aliased audio resampling."
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

# IMPORTANT:
#
# Use BF16 to match the proven V1 encoder configuration and keep
# the dynamic range of FP32 while using approximately half the
# storage of FP32.
#
# RTX 2050 is Ampere-class and supports BF16.
ENCODER_DTYPE = torch.bfloat16

# Resolve paths relative to this script.
V2_DIR = Path(__file__).resolve().parent

CACHE_DIR = V2_DIR / "cache" / "features"

OUTPUT_DIR = V2_DIR / "logs"
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "encoder_cache_benchmark.json"
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
# GPU MEMORY UTILITIES
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

    allocated = (
        torch.cuda.memory_allocated()
    )

    reserved = (
        torch.cuda.memory_reserved()
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
    )

    return {
        "allocated_mb": (
            allocated / (1024 ** 2)
        ),
        "reserved_mb": (
            reserved / (1024 ** 2)
        ),
        "peak_allocated_mb": (
            peak_allocated / (1024 ** 2)
        ),
        "peak_reserved_mb": (
            peak_reserved / (1024 ** 2)
        ),
        "free_mb": (
            free_bytes / (1024 ** 2)
        ),
        "total_mb": (
            total_bytes / (1024 ** 2)
        ),
    }


def reset_peak_memory() -> None:
    """
    Reset CUDA peak-memory statistics.

    Note:
        This does NOT free memory.
        It only resets the peak counters.
    """

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()


def synchronize() -> None:
    """
    Synchronize CUDA.
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cleanup_memory() -> None:
    """
    Release unused Python/CUDA memory between samples.
    """

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.synchronize()

        torch.cuda.empty_cache()

        torch.cuda.ipc_collect()


# ============================================================================
# AUDIO RESAMPLING
# ============================================================================

def resample_audio(
    waveform: torch.Tensor,
    source_sample_rate: int,
) -> torch.Tensor:
    """
    High-quality anti-aliased resampling using torchaudio.

    Input:
        waveform: 1-D float32 tensor

    Output:
        1-D float32 tensor at TARGET_SAMPLE_RATE
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


# ============================================================================
# AUDIO LOADING
# ============================================================================

def load_audio(path: Path) -> torch.Tensor:
    """
    Load audio as mono float32 at 16 kHz.

    Stereo/multi-channel audio is converted to mono.

    Non-16kHz audio is resampled using torchaudio's
    anti-aliased resampler.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Audio file does not exist:\n{path}"
        )

    audio, sample_rate = sf.read(
        str(path),
        dtype="float32",
    )

    waveform = torch.from_numpy(
        audio
    )

    # ------------------------------------------------------------
    # Stereo / multi-channel -> mono
    # ------------------------------------------------------------

    if waveform.ndim == 2:

        waveform = waveform.mean(
            dim=1
        )

    waveform = waveform.contiguous()

    # ------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        waveform = resample_audio(
            waveform,
            int(sample_rate),
        )

    return waveform.contiguous()


# ============================================================================
# REPRESENTATIVE SAMPLE SELECTION
# ============================================================================

def select_representative_samples(records):
    """
    Select samples closest to representative durations.

    No dataset rows are removed or modified.
    """

    targets = [
        ("LOW", 2.5),
        ("NORMAL", 4.2),
        ("MEDIUM", 7.0),
        ("HIGH", 10.0),
        ("EXTREME", 18.0),
    ]

    selected = []

    used_indices = set()

    for label, target in targets:

        candidates = [
            record
            for record in records
            if record.index
            not in used_indices
        ]

        if not candidates:
            break

        record = min(
            candidates,
            key=lambda r: abs(
                float(r.duration_sec)
                - target
            ),
        )

        selected.append(
            (label, record)
        )

        used_indices.add(
            record.index
        )

    return selected


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def model_input_from_feature_extractor(
    feature_extractor,
    waveform: torch.Tensor,
):
    """
    Convert raw waveform into Wav2Vec2-BERT inputs.

    Feature extraction is performed on CPU.
    """

    processed = feature_extractor(
        waveform.numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )

    return processed


# ============================================================================
# ENCODER FORWARD
# ============================================================================

def run_encoder(
    model,
    feature_extractor,
    waveform: torch.Tensor,
):
    """
    Run one frozen Wav2Vec2-BERT inference pass.

    Returns:
        features
        encoder_time
        model_inputs
    """

    processed = (
        model_input_from_feature_extractor(
            feature_extractor,
            waveform,
        )
    )

    model_inputs = {}

    model_signature = inspect.signature(
        model.forward
    )

    for key, value in processed.items():

        if (
            key
            not in model_signature.parameters
        ):
            continue

        if isinstance(
            value,
            torch.Tensor,
        ):

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
            "model-compatible tensor inputs.\n\n"
            f"Extractor keys: "
            f"{list(processed.keys())}\n\n"
            f"Model parameters: "
            f"{list(model_signature.parameters.keys())}"
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

    return (
        features.detach(),
        elapsed,
        model_inputs,
    )


# ============================================================================
# FEATURE VALIDATION
# ============================================================================

def validate_features(
    features: torch.Tensor,
    record,
) -> None:
    """
    Validate Wav2Vec2-BERT output.
    """

    if features.ndim != 3:

        raise RuntimeError(
            "Unexpected encoder output rank.\n"
            "Expected: [batch, time, hidden]\n"
            f"Received: {tuple(features.shape)}"
        )

    batch_size = (
        features.shape[0]
    )

    hidden_size = (
        features.shape[-1]
    )

    if batch_size != 1:

        raise RuntimeError(
            f"Expected batch size 1, "
            f"got {batch_size}."
        )

    if hidden_size != EXPECTED_ENCODER_DIM:

        raise RuntimeError(
            "Unexpected Wav2Vec2-BERT "
            "hidden dimension.\n"
            f"Expected: "
            f"{EXPECTED_ENCODER_DIM}\n"
            f"Received: {hidden_size}\n"
            f"Sample index: "
            f"{record.index}"
        )

    if not torch.isfinite(
        features
    ).all():

        raise RuntimeError(
            "Encoder produced NaN or Inf values."
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 72)
    print(
        "V2 REAL Wav2Vec2-BERT "
        "FEATURE CACHE BENCHMARK"
    )
    print("=" * 72)

    # ========================================================================
    # STEP 1
    # ========================================================================

    print("\n[1/7] Checking CUDA...")

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available.\n"
            "This benchmark requires "
            "the NVIDIA RTX 2050."
        )

    gpu_name = (
        torch.cuda.get_device_name(0)
    )

    print(
        f"GPU: {gpu_name}"
    )

    print(
        f"CUDA device index: "
        f"{torch.cuda.current_device()}"
    )

    print(
        f"PyTorch version: "
        f"{torch.__version__}"
    )

    print(
        f"Encoder dtype: "
        f"{ENCODER_DTYPE}"
    )

    initial_memory = gpu_memory()

    print(
        f"Initial free VRAM: "
        f"{initial_memory['free_mb']:.1f} MB"
    )

    print(
        f"Initial allocated VRAM: "
        f"{initial_memory['allocated_mb']:.1f} MB"
    )

    print(
        f"Initial reserved VRAM: "
        f"{initial_memory['reserved_mb']:.1f} MB"
    )

    # ========================================================================
    # STEP 2
    # ========================================================================

    print(
        "\n[2/7] Loading Data Manager..."
    )

    manager = DataManager()

    train_records = manager.train

    print(
        f"Training records: "
        f"{len(train_records):,}"
    )

    print(
        f"Validation records: "
        f"{len(manager.val):,}"
    )

    print(
        f"Test records: "
        f"{len(manager.test):,}"
    )

    # Protect test set.
    if len(manager.test) != 601:

        raise RuntimeError(
            "TEST SET PROTECTION FAILURE.\n"
            f"Expected 601 records, "
            f"found {len(manager.test)}."
        )

    print(
        "Test-set protection: "
        "PASS (601 samples)"
    )

    # ========================================================================
    # STEP 3
    # ========================================================================

    print(
        "\n[3/7] Selecting representative samples..."
    )

    selected = (
        select_representative_samples(
            train_records
        )
    )

    if len(selected) != 5:

        raise RuntimeError(
            "Could not select all five "
            "representative samples."
        )

    print("\nSelected samples:")

    for label, record in selected:

        print(
            f"  {label:8s} | "
            f"idx={record.index:5d} | "
            f"duration="
            f"{record.duration_sec:.3f}s | "
            f"projected="
            f"{record.estimated_projected_tokens}"
        )

    # ========================================================================
    # STEP 4
    # ========================================================================

    print(
        "\n[4/7] Loading feature extractor "
        "+ frozen encoder..."
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        "Processor: AutoFeatureExtractor"
    )

    print(
        "Encoder: Wav2Vec2BertModel"
    )

    print(
        f"Precision: {ENCODER_DTYPE}"
    )

    # IMPORTANT:
    #
    # AutoProcessor is deliberately NOT used.
    #
    # facebook/w2v-bert-2.0 is being used here as a bare
    # speech encoder. AutoProcessor may attempt to construct
    # a tokenizer and fail because there is no CTC vocabulary.
    feature_extractor = (
        AutoFeatureExtractor.from_pretrained(
            MODEL_NAME
        )
    )

    model = (
        Wav2Vec2BertModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=ENCODER_DTYPE,
        )
        .to(DEVICE)
    )

    model.eval()

    # Freeze encoder.
    for parameter in model.parameters():

        parameter.requires_grad = False

    synchronize()

    after_model = gpu_memory()

    print("\nVRAM after encoder load:")

    print(
        f"  Allocated: "
        f"{after_model['allocated_mb']:.1f} MB"
    )

    print(
        f"  Reserved : "
        f"{after_model['reserved_mb']:.1f} MB"
    )

    print(
        f"  Free     : "
        f"{after_model['free_mb']:.1f} MB"
    )

    print(
        f"  Total    : "
        f"{after_model['total_mb']:.1f} MB"
    )

    # Model dimensions.
    model_hidden_size = getattr(
        model.config,
        "output_hidden_size",
        None,
    )

    if model_hidden_size is None:

        model_hidden_size = getattr(
            model.config,
            "hidden_size",
            None,
        )

    print(
        f"  Model output hidden size: "
        f"{model_hidden_size}"
    )

    if (
        model_hidden_size
        != EXPECTED_ENCODER_DIM
    ):

        raise RuntimeError(
            "Unexpected model configuration.\n"
            f"Expected output dimension: "
            f"{EXPECTED_ENCODER_DIM}\n"
            f"Detected: "
            f"{model_hidden_size}"
        )

    # ========================================================================
    # STEP 5
    # ========================================================================

    print(
        "\n[5/7] Initializing feature cache..."
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Cache directory: "
        f"{CACHE_DIR}"
    )

    cache = FrozenEncoderFeatureCache(
        FeatureCacheConfig(
            cache_dir=str(CACHE_DIR),
            max_ram_items=2,
            disk_dtype="float16",
            enable_ram_cache=True,
            enable_disk_cache=True,
        )
    )

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
        f"Fingerprint: "
        f"{fingerprint}"
    )

    # ========================================================================
    # STEP 6
    # ========================================================================

    print(
        "\n[6/7] Running real encoder benchmark..."
    )

    print("=" * 72)

    results = []

    for number, (
        label,
        record,
    ) in enumerate(
        selected,
        start=1,
    ):

        print(
            f"\n[{number}/{len(selected)}] "
            f"{label}"
        )

        print(
            f"Index: {record.index}"
        )

        print(
            f"Duration: "
            f"{record.duration_sec:.3f}s"
        )

        print(
            f"Audio: "
            f"{record.audio_path}"
        )

        # --------------------------------------------------------------------
        # Load audio
        # --------------------------------------------------------------------

        waveform = load_audio(
            Path(record.audio_path)
        )

        print(
            f"Audio samples: "
            f"{waveform.numel():,}"
        )

        actual_duration = (
            waveform.numel()
            / TARGET_SAMPLE_RATE
        )

        print(
            f"Actual loaded duration: "
            f"{actual_duration:.3f}s"
        )

        # --------------------------------------------------------------------
        # Memory before encoder
        # --------------------------------------------------------------------

        before = gpu_memory()

        reset_peak_memory()

        synchronize()

        # --------------------------------------------------------------------
        # Encoder
        # --------------------------------------------------------------------

        start_total = (
            time.perf_counter()
        )

        (
            features,
            encoder_time,
            model_inputs,
        ) = run_encoder(
            model,
            feature_extractor,
            waveform,
        )

        synchronize()

        first_pass_time = (
            time.perf_counter()
            - start_total
        )

        after = gpu_memory()

        # --------------------------------------------------------------------
        # Validate output
        # --------------------------------------------------------------------

        validate_features(
            features,
            record,
        )

        # --------------------------------------------------------------------
        # Feature statistics
        # --------------------------------------------------------------------

        feature_shape = tuple(
            features.shape
        )

        feature_elements = (
            features.numel()
        )

        # Actual tensor size.
        actual_feature_bytes = (
            features.element_size()
            * feature_elements
        )

        # Equivalent FP16 / FP32 sizes.
        feature_fp16_bytes = (
            feature_elements * 2
        )

        feature_fp32_bytes = (
            feature_elements * 4
        )

        # --------------------------------------------------------------------
        # Encoder result
        # --------------------------------------------------------------------

        print("\nEncoder result:")

        print(
            f"  Shape: "
            f"{feature_shape}"
        )

        print(
            f"  Feature dtype: "
            f"{features.dtype}"
        )

        print(
            f"  Hidden dimension: "
            f"{feature_shape[-1]}"
        )

        print(
            f"  Encoder time: "
            f"{encoder_time:.4f}s"
        )

        print(
            f"  Total first pass: "
            f"{first_pass_time:.4f}s"
        )

        print(
            f"  Actual tensor size: "
            f"{actual_feature_bytes / (1024 ** 2):.2f} MB"
        )

        print(
            f"  Equivalent FP16 size: "
            f"{feature_fp16_bytes / (1024 ** 2):.2f} MB"
        )

        print(
            f"  Equivalent FP32 size: "
            f"{feature_fp32_bytes / (1024 ** 2):.2f} MB"
        )

        # --------------------------------------------------------------------
        # VRAM
        # --------------------------------------------------------------------

        print("\nVRAM:")

        print(
            f"  Before: "
            f"{before['allocated_mb']:.1f} MB "
            f"allocated"
        )

        print(
            f"  After:  "
            f"{after['allocated_mb']:.1f} MB "
            f"allocated"
        )

        print(
            f"  Peak:   "
            f"{after['peak_allocated_mb']:.1f} MB "
            f"allocated"
        )

        print(
            f"  Reserved peak: "
            f"{after['peak_reserved_mb']:.1f} MB"
        )

        print(
            f"  Free:   "
            f"{after['free_mb']:.1f} MB"
        )

        # --------------------------------------------------------------------
        # Cache key
        # --------------------------------------------------------------------

        cache_key = cache.build_key(
            record,
            fingerprint,
        )

        cache_path = cache.cache_path(
            cache_key
        )

        # --------------------------------------------------------------------
        # Disk cache write
        # --------------------------------------------------------------------

        disk_before = (
            cache.disk_size_bytes()
        )

        cache._disk_put(
            cache_key,
            features,
            metadata={
                "benchmark": True,
                "label": label,
                "record_index": int(
                    record.index
                ),
                "duration_sec": float(
                    record.duration_sec
                ),
                "sample_rate": (
                    TARGET_SAMPLE_RATE
                ),
                "feature_dtype": str(
                    features.dtype
                ),
                "feature_shape": list(
                    feature_shape
                ),
                "encoder_model": (
                    MODEL_NAME
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

        print("\nCache write:")

        print(
            f"  File: "
            f"{cache_path.name}"
        )

        print(
            f"  Disk increase: "
            f"{disk_delta / (1024 ** 2):.2f} MB"
        )

        # --------------------------------------------------------------------
        # Clear RAM cache
        # --------------------------------------------------------------------

        cache.clear_ram()

        synchronize()

        # --------------------------------------------------------------------
        # Disk cache read
        # --------------------------------------------------------------------

        disk_start = (
            time.perf_counter()
        )

        disk_features = (
            cache._disk_get(
                cache_key
            )
        )

        disk_load_time = (
            time.perf_counter()
            - disk_start
        )

        if disk_features is None:

            raise RuntimeError(
                "Disk cache returned None."
            )

        if (
            tuple(disk_features.shape)
            != feature_shape
        ):

            raise RuntimeError(
                "Disk cached feature "
                "shape mismatch.\n"
                f"Original: "
                f"{feature_shape}\n"
                f"Cached: "
                f"{tuple(disk_features.shape)}"
            )

        print("\nDisk cache read:")

        print(
            f"  Load time: "
            f"{disk_load_time:.4f}s"
        )

        print(
            f"  Shape: "
            f"{tuple(disk_features.shape)}"
        )

        print(
            f"  Dtype: "
            f"{disk_features.dtype}"
        )

        # --------------------------------------------------------------------
        # Store results
        # --------------------------------------------------------------------

        results.append(
            {
                "label": label,

                "record_index": int(
                    record.index
                ),

                "duration_sec": float(
                    record.duration_sec
                ),

                "actual_loaded_duration_sec": float(
                    actual_duration
                ),

                "estimated_projected_tokens": int(
                    record.estimated_projected_tokens
                ),

                "feature_shape": list(
                    feature_shape
                ),

                "feature_elements": int(
                    feature_elements
                ),

                "feature_dtype": str(
                    features.dtype
                ),

                "feature_element_size_bytes": int(
                    features.element_size()
                ),

                "feature_hidden_size": int(
                    feature_shape[-1]
                ),

                "feature_actual_mb": (
                    actual_feature_bytes
                    / (1024 ** 2)
                ),

                "feature_fp16_mb": (
                    feature_fp16_bytes
                    / (1024 ** 2)
                ),

                "feature_fp32_mb": (
                    feature_fp32_bytes
                    / (1024 ** 2)
                ),

                "encoder_time_sec": float(
                    encoder_time
                ),

                "first_pass_total_sec": float(
                    first_pass_time
                ),

                "vram_before_allocated_mb": (
                    before[
                        "allocated_mb"
                    ]
                ),

                "vram_after_allocated_mb": (
                    after[
                        "allocated_mb"
                    ]
                ),

                "vram_peak_allocated_mb": (
                    after[
                        "peak_allocated_mb"
                    ]
                ),

                "vram_peak_reserved_mb": (
                    after[
                        "peak_reserved_mb"
                    ]
                ),

                "vram_free_after_mb": (
                    after[
                        "free_mb"
                    ]
                ),

                "disk_cache_delta_mb": (
                    disk_delta
                    / (1024 ** 2)
                ),

                "disk_load_time_sec": float(
                    disk_load_time
                ),

                "cache_file": str(
                    cache_path
                ),
            }
        )

        # --------------------------------------------------------------------
        # Cleanup
        # --------------------------------------------------------------------

        del features
        del disk_features
        del waveform
        del model_inputs

        cleanup_memory()

    # ========================================================================
    # STEP 7
    # ========================================================================

    print(
        "\n[7/7] Saving benchmark results..."
    )

    final_memory = gpu_memory()

    summary = {
        "benchmark": {
            "name": (
                "V2 REAL "
                "Wav2Vec2-BERT "
                "Feature Cache Benchmark"
            ),
            "version": "1.1",
        },

        "model": {
            "model_name": MODEL_NAME,
            "encoder_class": (
                "Wav2Vec2BertModel"
            ),
            "feature_extractor_class": (
                "AutoFeatureExtractor"
            ),
            "dtype": str(
                ENCODER_DTYPE
            ),
            "expected_output_hidden_size": (
                EXPECTED_ENCODER_DIM
            ),
        },

        "device": {
            "device": str(
                DEVICE
            ),
            "gpu": gpu_name,
            "cuda_device_index": int(
                torch.cuda.current_device()
            ),
            "torch_version": (
                torch.__version__
            ),
        },

        "audio": {
            "target_sample_rate": (
                TARGET_SAMPLE_RATE
            ),
            "resampler": (
                "torchaudio.transforms.Resample"
            ),
        },

        "paths": {
            "v2_directory": str(
                V2_DIR
            ),
            "cache_directory": str(
                CACHE_DIR
            ),
            "output_file": str(
                OUTPUT_FILE
            ),
        },

        "dataset": {
            "train_records": len(
                manager.train
            ),
            "validation_records": len(
                manager.val
            ),
            "test_records": len(
                manager.test
            ),
            "test_set_protected": (
                len(manager.test)
                == 601
            ),
        },

        "memory": {
            "initial": initial_memory,
            "after_encoder_load": (
                after_model
            ),
            "final": final_memory,
        },

        "cache": {
            "fingerprint": fingerprint,
            "statistics": (
                cache.statistics()
            ),
            "disk_size_mb": (
                cache.disk_size_bytes()
                / (1024 ** 2)
            ),
        },

        "results": results,
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_FILE}"
    )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print("\n" + "=" * 72)
    print(
        "ENCODER CACHE BENCHMARK COMPLETE"
    )
    print("=" * 72)

    print(
        f"\nSamples tested: "
        f"{len(results)}"
    )

    print(
        f"Cache disk size: "
        f"{summary['cache']['disk_size_mb']:.2f} MB"
    )

    print("\nRepresentative results:")

    for result in results:

        print(
            f"\n{result['label']:8s} | "
            f"{result['duration_sec']:6.2f}s | "
            f"shape="
            f"{result['feature_shape']} | "
            f"dtype="
            f"{result['feature_dtype']} | "
            f"encoder="
            f"{result['encoder_time_sec']:.3f}s | "
            f"peak VRAM="
            f"{result['vram_peak_allocated_mb']:.1f} MB | "
            f"free="
            f"{result['vram_free_after_mb']:.1f} MB | "
            f"disk="
            f"{result['disk_cache_delta_mb']:.2f} MB | "
            f"disk load="
            f"{result['disk_load_time_sec']:.3f}s"
        )

    # ========================================================================
    # VALIDATION
    # ========================================================================

    print("\nValidation:")

    all_shapes_valid = all(
        result[
            "feature_hidden_size"
        ]
        == EXPECTED_ENCODER_DIM
        for result in results
    )

    all_disk_valid = all(
        result[
            "disk_load_time_sec"
        ] >= 0
        for result in results
    )

    test_set_valid = (
        len(manager.test)
        == 601
    )

    print(
        "  Output dimension 1024: "
        f"{'PASS' if all_shapes_valid else 'FAIL'}"
    )

    print(
        "  Disk cache read: "
        f"{'PASS' if all_disk_valid else 'FAIL'}"
    )

    print(
        "  Test set unchanged: "
        f"{'PASS' if test_set_valid else 'FAIL'}"
    )

    if not all_shapes_valid:

        raise RuntimeError(
            "Benchmark validation failed: "
            "unexpected feature dimension."
        )

    if not all_disk_valid:

        raise RuntimeError(
            "Benchmark validation failed: "
            "disk cache validation failed."
        )

    if not test_set_valid:

        raise RuntimeError(
            "Benchmark validation failed: "
            "test set is not 601 samples."
        )

    print(
        "\nBenchmark finished successfully."
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()