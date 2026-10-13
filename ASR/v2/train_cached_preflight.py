import gc
import inspect
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

# Reduce CUDA allocator fragmentation; must be set before the first CUDA allocation.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True",
)

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


# ============================================================================
# PATHS
# ============================================================================

THIS_FILE = Path(__file__).resolve()

V2_DIR = THIS_FILE.parent

ASR_DIR = V2_DIR.parent

PROJECT_ROOT = ASR_DIR.parent

LOG_DIR = V2_DIR / "logs"

CACHE_DIR = V2_DIR / "cache" / "features"

# This is the fingerprint used by the existing precomputed Wav2Vec2-BERT
# feature cache. Do not change it unless the cache is deliberately rebuilt.
EXPECTED_FEATURE_CACHE_FINGERPRINT = (
    "ae2243975120c0e7c1ece6d11557de23788f0a4143fab3da2c5feeb5fab5292d"
)
FEATURE_CACHE_DISK_DTYPE = "float16"

LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_JSON = (
    LOG_DIR
    / "cached_qlora_preflight.json"
)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

ENCODER_MODEL_NAME = (
    "facebook/w2v-bert-2.0"
)

TARGET_SAMPLE_RATE = 16000

ENCODER_DTYPE = torch.bfloat16

QWEN_MODEL_NAME = (
    "Qwen/Qwen2.5-3B-Instruct"
)

DEVICE = torch.device(
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

COMPUTE_DTYPE = torch.bfloat16


# ============================================================================
# QLORA CONFIGURATION
# ============================================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# IMPORTANT:
# These modules must remain trainable because the tokenizer
# may contain newly added Meitei Mayek tokens.
LORA_MODULES_TO_SAVE = [
    "embed_tokens",
    "lm_head",
]


# ============================================================================
# PROJECTOR CONFIGURATION
# ============================================================================

ENCODER_HIDDEN = 1024

LLM_HIDDEN = 2048

PROJECTOR_CONV_KERNEL = 3

PROJECTOR_CONV_STRIDE = 2

EXPECTED_PROJECTOR_PARAMS = 5_250_048


# ============================================================================
# DATASET PROTECTION
# ============================================================================

EXPECTED_TRAIN_ROWS = 10_199

EXPECTED_VAL_ROWS = 1_207

EXPECTED_TEST_ROWS = 601


# ============================================================================
# PREFLIGHT CONFIGURATION
# ============================================================================

OVERFIT_SAMPLES = 20

OVERFIT_EPOCHS = 3

MAX_TARGET_TOKENS = 256

OVERFIT_LR = 2e-4

OVERFIT_WEIGHT_DECAY = 0.01


# ============================================================================
# PRINTING
# ============================================================================

def print_header(
    title: str,
) -> None:

    print()

    print("=" * 72)

    print(title)

    print("=" * 72)


# ============================================================================
# CUDA MEMORY HELPERS
# ============================================================================

def cleanup_cuda() -> None:

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def reset_cuda_peak() -> None:

    if torch.cuda.is_available():

        torch.cuda.reset_peak_memory_stats(
            device=0
        )


def cuda_memory() -> dict[str, float]:

    if not torch.cuda.is_available():

        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "peak_mb": 0.0,
            "free_mb": 0.0,
            "total_mb": 0.0,
        }

    allocated = (
        torch.cuda.memory_allocated(0)
    )

    reserved = (
        torch.cuda.memory_reserved(0)
    )

    peak = (
        torch.cuda.max_memory_allocated(0)
    )

    free, total = (
        torch.cuda.mem_get_info(0)
    )

    return {
        "allocated_mb":
            allocated / 1024**2,

        "reserved_mb":
            reserved / 1024**2,

        "peak_mb":
            peak / 1024**2,

        "free_mb":
            free / 1024**2,

        "total_mb":
            total / 1024**2,
    }


def print_cuda_memory(
    label: str,
) -> dict[str, float]:

    mem = cuda_memory()

    print(
        f"{label}: "
        f"allocated="
        f"{mem['allocated_mb']:.1f} MB | "
        f"reserved="
        f"{mem['reserved_mb']:.1f} MB | "
        f"peak="
        f"{mem['peak_mb']:.1f} MB | "
        f"free="
        f"{mem['free_mb']:.1f} MB"
    )

    return mem


# ============================================================================
# HARDWARE INSPECTION
# ============================================================================

def inspect_hardware() -> dict[str, Any]:

    print_header(
        "HARDWARE VALIDATION"
    )

    result = {
        "cuda_available":
            torch.cuda.is_available(),

        "device":
            str(DEVICE),

        "torch_version":
            torch.__version__,
    }

    if not torch.cuda.is_available():

        print(
            "CUDA AVAILABLE: NO"
        )

        return result

    props = (
        torch.cuda.get_device_properties(0)
    )

    free, total = (
        torch.cuda.mem_get_info(0)
    )

    result.update(
        {
            "gpu_name":
                props.name,

            "cuda_device":
                0,

            "compute_capability":
                f"{props.major}.{props.minor}",

            "physical_vram_mb":
                total / 1024**2,

            "free_vram_mb":
                free / 1024**2,

            "physical_vram_gb":
                total / 1024**3,

            "free_vram_gb":
                free / 1024**3,
        }
    )

    print(
        f"GPU: {props.name}"
    )

    print(
        "CUDA device: 0"
    )

    print(
        f"Compute capability: "
        f"{props.major}.{props.minor}"
    )

    print(
        f"Physical VRAM: "
        f"{total / 1024**3:.2f} GB"
    )

    print(
        f"Free VRAM: "
        f"{free / 1024**3:.2f} GB"
    )

    return result


# ============================================================================
# DATA MANAGER
# ============================================================================

def load_data_manager():

    try:

        from data_manager import (
            DataManager
        )

    except ImportError:

        from .data_manager import (
            DataManager
        )

    attempts = [

        lambda:
            DataManager(
                PROJECT_ROOT
            ),

        lambda:
            DataManager(
                project_root=
                    PROJECT_ROOT
            ),

        lambda:
            DataManager(
                root_dir=
                    PROJECT_ROOT
            ),

        lambda:
            DataManager(),
    ]

    errors = []

    for attempt in attempts:

        try:

            return attempt()

        except Exception as exc:

            errors.append(
                str(exc)
            )

    raise RuntimeError(
        "Unable to initialize "
        "DataManager.\n"
        f"Errors: {errors[-4:]}"
    )


def call_method_if_exists(
    obj: Any,
    names: list[str],
    *args,
    **kwargs,
):

    for name in names:

        method = getattr(
            obj,
            name,
            None,
        )

        if not callable(method):

            continue

        try:

            return method(
                *args,
                **kwargs,
            )

        except TypeError:

            continue

    return None


def extract_split(
    manager: Any,
    split: str,
):

    attributes = [
        split,
        f"{split}_records",
        f"{split}_data",
    ]

    for attr in attributes:

        if hasattr(
            manager,
            attr,
        ):

            value = getattr(
                manager,
                attr,
            )

            if value is not None:

                return value

    methods = [
        f"get_{split}",
        f"load_{split}",
        f"get_{split}_records",
    ]

    value = call_method_if_exists(
        manager,
        methods,
    )

    if value is not None:

        return value

    raise RuntimeError(
        f"Could not obtain "
        f"'{split}' split."
    )


def load_datasets():

    print_header(
        "DATASET VALIDATION"
    )

    manager = (
        load_data_manager()
    )

    call_method_if_exists(
        manager,
        [
            "load",
            "load_all",
            "load_datasets",
            "initialize",
            "prepare",
        ],
    )

    train = extract_split(
        manager,
        "train",
    )

    val = extract_split(
        manager,
        "val",
    )

    test = extract_split(
        manager,
        "test",
    )

    print(
        f"TRAIN: {len(train):,}"
    )

    print(
        f"VAL:   {len(val):,}"
    )

    print(
        f"TEST:  {len(test):,}"
    )

    if (
        len(train)
        != EXPECTED_TRAIN_ROWS
    ):

        raise RuntimeError(
            "TRAIN row count changed.\n"
            f"Expected: "
            f"{EXPECTED_TRAIN_ROWS:,}\n"
            f"Found: "
            f"{len(train):,}"
        )

    if (
        len(val)
        != EXPECTED_VAL_ROWS
    ):

        raise RuntimeError(
            "VAL row count changed.\n"
            f"Expected: "
            f"{EXPECTED_VAL_ROWS:,}\n"
            f"Found: "
            f"{len(val):,}"
        )

    if (
        len(test)
        != EXPECTED_TEST_ROWS
    ):

        raise RuntimeError(
            "TEST row count changed.\n"
            f"Expected: "
            f"{EXPECTED_TEST_ROWS:,}\n"
            f"Found: "
            f"{len(test):,}"
        )

    print(
        "Dataset counts: PASS"
    )

    print(
        "Test-set protection: "
        "PASS (601 samples)"
    )

    return (
        manager,
        train,
        val,
        test,
    )


# ============================================================================
# RECORD HELPERS
# ============================================================================

def get_record_value(
    record: Any,
    names: list[str],
    default=None,
):

    for name in names:

        if (
            isinstance(record, dict)
            and name in record
        ):

            return record[name]

        if hasattr(
            record,
            name,
        ):

            return getattr(
                record,
                name,
            )

    return default


def record_index(
    record: Any,
) -> int:

    value = get_record_value(
        record,
        [
            "index",
            "idx",
            "record_index",
        ],
        0,
    )

    return int(value)


def record_text(
    record: Any,
) -> str:

    value = get_record_value(
        record,
        [
            "text",
            "transcript",
            "sentence",
        ],
        "",
    )

    return str(value)


def record_audio_path(
    record: Any,
) -> str:

    value = get_record_value(
        record,
        [
            "audio_path",
            "path",
            "resolved_audio_path",
        ],
        "",
    )

    return str(value)


# ============================================================================
# FEATURE CACHE
# ============================================================================

def load_feature_cache():

    print_header(
        "FEATURE CACHE"
    )

    try:

        from feature_cache import (
            FrozenEncoderFeatureCache,
            FeatureCacheConfig,
        )

    except ImportError:

        from .feature_cache import (
            FrozenEncoderFeatureCache,
            FeatureCacheConfig,
        )

    # Instantiate directly with cache_dir because FeatureCacheConfig is frozen
    config = FeatureCacheConfig(
        cache_dir=CACHE_DIR
    )

    cache = (
        FrozenEncoderFeatureCache(
            config=config,
            project_dir=PROJECT_ROOT,
        )
    )

    fingerprint = cache.build_fingerprint(
        model_name=ENCODER_MODEL_NAME,
        sampling_rate=TARGET_SAMPLE_RATE,
        feature_extractor_config={
            "sampling_rate": (
                TARGET_SAMPLE_RATE
            ),
            "encoder_dtype": str(
                ENCODER_DTYPE
            ),
            "output_hidden_size": (
                ENCODER_HIDDEN
            ),
            
        },
    )

    cache.model_fingerprint = fingerprint

    if fingerprint != EXPECTED_FEATURE_CACHE_FINGERPRINT:
        raise RuntimeError(
            "Feature-cache fingerprint mismatch.\n"
            f"Expected existing-cache fingerprint: {EXPECTED_FEATURE_CACHE_FINGERPRINT}\n"
            f"Computed fingerprint:                 {fingerprint}\n\n"
            "The preflight refuses to use a different cache identity. "
            "Check precompute_features.py and reproduce its exact "
            "build_fingerprint configuration before continuing."
        )

    print(
        f"Configured cache directory: "
        f"{config.cache_dir}"
    )

    print(
        f"Resolved cache directory:   "
        f"{cache.cache_dir}"
    )

    print(
        f"Cache directory exists:     "
        f"{Path(cache.cache_dir).exists()}"
    )

    print(
        f"Model fingerprint:          "
        f"{fingerprint}"
    )

    expected = (
        CACHE_DIR.resolve()
    )

    actual = (
        Path(
            cache.cache_dir
        ).resolve()
    )

    if actual != expected:

        raise RuntimeError(
            "Unexpected cache directory.\n"
            f"Expected:\n{expected}\n"
            f"Actual:\n{actual}"
        )

    print(
        "Feature cache path validation: "
        "PASS"
    )

    return cache


# ============================================================================
# CACHE API ADAPTER
# ============================================================================

def call_cache_method_with_signature(
    method,
    record: Any,
    key: Any = None,
    cache: Any = None,
    model_fingerprint: str = "",
):
    """
    Call a feature-cache method while adapting to its exact signature.

    Important:
    - model_fingerprint must be the actual cache fingerprint.
    - We do NOT substitute model_name for model_fingerprint.
    - If `method` is bound to a FrozenEncoderFeatureCache instance,
      method.__self__ is used automatically.
    """

    # Recover the bound cache instance automatically.
    if cache is None and hasattr(method, "__self__"):
        cache = method.__self__

    signature = inspect.signature(method)
    parameters = list(signature.parameters.values())

    idx = record_index(record)

    audio_path = record_audio_path(record)

    # If the caller did not provide a fingerprint, try to recover
    # an already-computed fingerprint from the cache object.
    if not model_fingerprint and cache is not None:

        if hasattr(cache, "model_fingerprint"):
            value = getattr(
                cache,
                "model_fingerprint",
            )

            model_fingerprint = (
                value()
                if callable(value)
                else value
            )

        elif hasattr(cache, "config") and hasattr(
            cache.config,
            "model_fingerprint",
        ):
            value = getattr(
                cache.config,
                "model_fingerprint",
            )

            model_fingerprint = (
                value()
                if callable(value)
                else value
            )

        if not model_fingerprint and hasattr(cache, "build_fingerprint"):
            model_fingerprint = cache.build_fingerprint(
                model_name=ENCODER_MODEL_NAME,
                sampling_rate=TARGET_SAMPLE_RATE,
                feature_extractor_config={
                    "sampling_rate": (
                        TARGET_SAMPLE_RATE
                    ),
                    "encoder_dtype": str(
                        ENCODER_DTYPE
                    ),
                    "output_hidden_size": (
                        ENCODER_HIDDEN
                    ),
                },
            )

    mapping = {
        "record": record,
        "item": record,
        "sample": record,
        "entry": record,

        "index": idx,
        "idx": idx,
        "record_index": idx,

        "audio_path": audio_path,
        "path": audio_path,
        "wav_path": audio_path,

        "key": key,
        "cache_key": key,
        "feature_key": key,

        "model_fingerprint": model_fingerprint,
        "fingerprint": model_fingerprint,

        "cache": cache,
    }

    args = []
    kwargs = {}

    for parameter in parameters:

        if parameter.name == "self":
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        if parameter.name in mapping:

            value = mapping[
                parameter.name
            ]

            if (
                parameter.kind
                == inspect.Parameter.POSITIONAL_ONLY
            ):
                args.append(value)

            else:
                kwargs[
                    parameter.name
                ] = value

            continue

        if (
            parameter.default
            is not inspect.Parameter.empty
        ):
            continue

        raise RuntimeError(
            "Cannot map required cache "
            f"parameter '{parameter.name}' "
            f"for method "
            f"{method.__name__}().\n"
            f"Signature: {signature}"
        )

    return method(
        *args,
        **kwargs,
    )

def cache_is_cached(
    cache: Any,
    record: Any,
) -> bool:

    method = getattr(
        cache,
        "is_cached",
        None,
    )

    if not callable(method):

        raise RuntimeError(
            "FrozenEncoderFeatureCache "
            "does not provide is_cached()."
        )

    try:

        result = (
            call_cache_method_with_signature(
                method,
                record,
                cache=cache,
                model_fingerprint=getattr(
                    cache,
                    "model_fingerprint",
                    "",
                ),
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed calling is_cached().\n"
            f"Record: "
            f"{record_index(record)}\n"
            f"Error: {exc}"
        )

    return bool(result)


def build_cache_key(
    cache: Any,
    record: Any,
):

    method = getattr(
        cache,
        "build_key",
        None,
    )

    if not callable(method):

        raise RuntimeError(
            "FrozenEncoderFeatureCache "
            "does not provide build_key()."
        )

    try:

        return (
            call_cache_method_with_signature(
                method,
                record,
                cache=cache,
                model_fingerprint=getattr(
                    cache,
                    "model_fingerprint",
                    "",
                ),
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed calling build_key().\n"
            f"Record: "
            f"{record_index(record)}\n"
            f"Error: {exc}"
        )


def resolve_cache_path(
    cache: Any,
    record: Any,
    key: Any,
):

    method = getattr(
        cache,
        "cache_path",
        None,
    )

    if not callable(method):

        raise RuntimeError(
            "FrozenEncoderFeatureCache "
            "does not provide cache_path()."
        )

    try:

        value = (
            call_cache_method_with_signature(
                method,
                record,
                key=key,
            )
        )

    except Exception as exc:

        raise RuntimeError(
            "Failed calling cache_path().\n"
            f"Record: "
            f"{record_index(record)}\n"
            f"Error: {exc}"
        )

    if value is None:

        raise RuntimeError(
            "cache_path() returned None."
        )

    return Path(value)


def extract_feature_tensor(
    payload: Any,
) -> Optional[torch.Tensor]:

    if torch.is_tensor(
        payload
    ):

        return payload

    if isinstance(
        payload,
        dict,
    ):

        possible_keys = [
            "features",
            "encoder_features",
            "hidden_states",
            "tensor",
            "value",
        ]

        for key in possible_keys:

            value = payload.get(
                key
            )

            if torch.is_tensor(
                value
            ):

                return value

    return None


def cache_get(
    cache: Any,
    record: Any,
) -> Optional[torch.Tensor]:

    """
    Read an EXISTING cache entry.

    This function deliberately does NOT call
    get_or_compute().

    A missing feature therefore causes the
    preflight to fail instead of silently
    loading Wav2Vec2-BERT.
    """

    if not cache_is_cached(
        cache,
        record,
    ):

        return None

    key = build_cache_key(
        cache,
        record,
    )

    path = resolve_cache_path(
        cache,
        record,
        key,
    )

    if not path.exists():

        return None

    try:

        payload = torch.load(
            path,
            map_location="cpu",
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to load cache file.\n"
            f"Path: {path}\n"
            f"Error: {exc}"
        )

    features = extract_feature_tensor(
        payload
    )

    if features is None:

        raise RuntimeError(
            "Cache file does not contain "
            "a recognizable feature tensor.\n"
            f"Path: {path}"
        )

    return features


def normalize_feature_shape(
    features: torch.Tensor,
) -> torch.Tensor:

    if not torch.is_tensor(
        features
    ):

        raise TypeError(
            "Cached feature is not a tensor.\n"
            f"Type: {type(features)}"
        )

    features = (
        features
        .detach()
        .cpu()
    )

    # [1, T, 1024] -> [T, 1024]
    if features.ndim == 3:

        if features.shape[0] != 1:

            raise ValueError(
                "Unexpected batch dimension.\n"
                f"Shape: {features.shape}"
            )

        features = features[0]

    if features.ndim != 2:

        raise ValueError(
            "Expected [T, 1024] features.\n"
            f"Found: {features.shape}"
        )

    if (
        features.shape[-1]
        != ENCODER_HIDDEN
    ):

        raise ValueError(
            "Unexpected encoder hidden size.\n"
            f"Expected: "
            f"{ENCODER_HIDDEN}\n"
            f"Found: "
            f"{features.shape[-1]}"
        )

    if not torch.isfinite(
        features
    ).all():

        raise ValueError(
            "Cached features contain "
            "NaN or Inf."
        )

    return features


def validate_real_cache(
    cache: Any,
    train,
    val,
):

    print_header(
        "REAL FEATURE CACHE VALIDATION"
    )

    candidates = []

    for split_name, collection in [
        ("train", train),
        ("val", val),
    ]:

        if len(collection) == 0:

            continue

        positions = [
            0,
            len(collection) // 4,
            len(collection) // 2,
            (3 * len(collection)) // 4,
            len(collection) - 1,
        ]

        for position in positions:

            if (
                0
                <= position
                < len(collection)
            ):

                candidates.append(
                    (
                        split_name,
                        collection[position],
                    )
                )

    validated = []

    seen = set()

    for split_name, record in candidates:

        identity = (
            split_name,
            record_index(record),
        )

        if identity in seen:

            continue

        seen.add(identity)

        cached = cache_is_cached(
            cache,
            record,
        )

        if not cached:

            print(
                f"MISS: {split_name} "
                f"idx="
                f"{record_index(record)}"
            )

            continue

        features = cache_get(
            cache,
            record,
        )

        if features is None:

            print(
                f"MISS: {split_name} "
                f"idx="
                f"{record_index(record)} "
                "(cache entry unavailable)"
            )

            continue

        features = (
            normalize_feature_shape(
                features
            )
        )

        projected_tokens = math.ceil(
            features.shape[0]
            / PROJECTOR_CONV_STRIDE
        )

        print(
            f"PASS: {split_name} "
            f"idx="
            f"{record_index(record)} | "
            f"encoder_frames="
            f"{features.shape[0]} | "
            f"projected_tokens~="
            f"{projected_tokens} | "
            f"dtype="
            f"{features.dtype}"
        )

        validated.append(
            {
                "split":
                    split_name,

                "index":
                    record_index(record),

                "encoder_frames":
                    int(
                        features.shape[0]
                    ),

                "projected_tokens":
                    int(
                        projected_tokens
                    ),

                "dtype":
                    str(
                        features.dtype
                    ),
            }
        )

    if not validated:

        raise RuntimeError(
            "No existing cached features "
            "could be loaded.\n\n"
            "Generate the Wav2Vec2-BERT "
            "feature cache before running "
            "this preflight."
        )

    print()

    print(
        "Validated existing cache samples: "
        f"{len(validated)}"
    )

    return validated


# ============================================================================
# AUDIO PROJECTOR
# ============================================================================

class AudioProjector(
    nn.Module
):

    def __init__(
        self,
        encoder_hidden:
            int = ENCODER_HIDDEN,

        llm_hidden:
            int = LLM_HIDDEN,
    ):

        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=
                encoder_hidden,

            out_channels=
                encoder_hidden,

            kernel_size=
                PROJECTOR_CONV_KERNEL,

            stride=
                PROJECTOR_CONV_STRIDE,

            padding=1,

            bias=True,
        )

        self.linear = nn.Linear(
            encoder_hidden,
            llm_hidden,
        )

        self.norm = nn.LayerNorm(
            llm_hidden
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim == 2:

            x = x.unsqueeze(0)

        # [B,T,C] -> [B,C,T]

        x = x.transpose(
            1,
            2,
        )

        x = self.conv(
            x
        )

        # [B,C,T] -> [B,T,C]

        x = x.transpose(
            1,
            2,
        )

        x = self.linear(
            x
        )

        x = self.norm(
            x
        )

        return x


def count_trainable_parameters(
    module: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def inspect_projector(
    projector: AudioProjector,
):

    print_header(
        "AUDIO PROJECTOR"
    )

    total = sum(
        parameter.numel()
        for parameter in projector.parameters()
    )

    trainable = (
        count_trainable_parameters(
            projector
        )
    )

    print(
        f"Projector total params:     "
        f"{total:,}"
    )

    print(
        f"Projector trainable params: "
        f"{trainable:,}"
    )

    print(
        f"Expected exact count:       "
        f"{EXPECTED_PROJECTOR_PARAMS:,}"
    )

    if not (
        5_000_000
        < trainable
        < 5_500_000
    ):

        raise RuntimeError(
            "Unexpected projector "
            "parameter count.\n"
            f"Found: {trainable:,}"
        )

    if (
        trainable
        != EXPECTED_PROJECTOR_PARAMS
    ):

        raise RuntimeError(
            "Projector parameter count "
            "does not match expected "
            "architecture.\n"
            f"Expected: "
            f"{EXPECTED_PROJECTOR_PARAMS:,}\n"
            f"Found: "
            f"{trainable:,}"
        )

    print(
        "Projector parameter count: PASS"
    )

    return {
        "total_params":
            total,

        "trainable_params":
            trainable,
    }


# ============================================================================
# TOKENIZER
# ============================================================================

def collect_meitei_characters(
    train,
    val,
) -> list[str]:

    chars = set()

    for record in (
        list(train)
        + list(val)
    ):

        text = record_text(
            record
        )

        for character in text:

            code = ord(
                character
            )

            # Meetei Mayek block
            if (
                0xABC0
                <= code
                <= 0xABFF
            ):

                chars.add(
                    character
                )

            # Supplementary block
            elif (
                0xAAE0
                <= code
                <= 0xAAFF
            ):

                chars.add(
                    character
                )

    return sorted(
        chars,
        key=ord,
    )


def tokenizer_single_token_check(
    tokenizer,
    characters,
):

    missing = []

    for character in characters:

        token_ids = (
            tokenizer.encode(
                character,
                add_special_tokens=False,
            )
        )

        if len(token_ids) != 1:

            missing.append(
                character
            )

    return missing


def load_tokenizer(
    train,
    val,
):

    print_header(
        "TOKENIZER"
    )

    print(
        f"Loading tokenizer: "
        f"{QWEN_MODEL_NAME}"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            QWEN_MODEL_NAME,

            trust_remote_code=True,

            use_fast=True,
        )
    )

    if tokenizer.pad_token is None:

        if (
            tokenizer.eos_token
            is not None
        ):

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        else:

            tokenizer.add_special_tokens(
                {
                    "pad_token":
                        "<pad>"
                }
            )

    characters = (
        collect_meitei_characters(
            train,
            val,
        )
    )

    print(
        "Meitei Mayek characters "
        "found in train/val: "
        f"{len(characters)}"
    )

    missing = (
        tokenizer_single_token_check(
            tokenizer,
            characters,
        )
    )

    print(
        "Characters not represented "
        "as single tokens: "
        f"{len(missing)}"
    )

    if missing:

        tokenizer.add_tokens(
            missing,
            special_tokens=False,
        )

        print(
            f"Added {len(missing)} "
            "Meitei Mayek tokens."
        )

    print(
        f"Final tokenizer vocabulary: "
        f"{len(tokenizer):,}"
    )

    if (
        tokenizer.eos_token_id
        is None
    ):

        raise RuntimeError(
            "Tokenizer has no EOS token."
        )

    return tokenizer


# ============================================================================
# QWEN + QLORA
# ============================================================================

def load_qwen_qlora(
    tokenizer,
):

    print_header(
        "QWEN 4-BIT QLORA"
    )

    cleanup_cuda()

    reset_cuda_peak()

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is required."
        )

    print(
        f"Loading: "
        f"{QWEN_MODEL_NAME}"
    )

    print(
        "Quantization: 4-bit NF4"
    )

    print(
        "Double quantization: ENABLED"
    )

    print(
        f"Compute dtype: "
        f"{COMPUTE_DTYPE}"
    )

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,

            bnb_4bit_quant_type=
                "nf4",

            bnb_4bit_use_double_quant=
                True,

            bnb_4bit_compute_dtype=
                COMPUTE_DTYPE,

        )
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            QWEN_MODEL_NAME,

            quantization_config=
                bnb_config,

            device_map={
                "": 0
            },

            trust_remote_code=True,

            torch_dtype=
                COMPUTE_DTYPE,

            low_cpu_mem_usage=True,
        )
    )

    # ------------------------------------------------------------------------
    # Vocabulary resize
    # ------------------------------------------------------------------------

    embedding_size = (
        model
        .get_input_embeddings()
        .weight
        .shape[0]
    )

    if (
        len(tokenizer)
        != embedding_size
    ):

        print(
            "Resizing token embeddings: "
            f"{embedding_size:,} -> "
            f"{len(tokenizer):,}"
        )

        model.resize_token_embeddings(
            len(tokenizer)
        )

    # ------------------------------------------------------------------------
    # K-bit preparation -- 4 GB VRAM SAFE
    #
    # PEFT's prepare_model_for_kbit_training() casts non-quantized
    # parameters to FP32. On a 4 GB RTX 2050 this creates a large
    # temporary VRAM spike after vocabulary resizing.
    #
    # We therefore perform only the preparation operations required
    # for QLoRA while preserving BF16 parameters.
    # ------------------------------------------------------------------------

    for param in model.parameters():

        param.requires_grad = False

    model.gradient_checkpointing_enable()

    try:

        model.enable_input_require_grads()

    except Exception:

        pass

    # ------------------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        target_modules=
            LORA_TARGET_MODULES,

        modules_to_save=[
            "embed_tokens",
            "lm_head",
        ],

        bias="none",

        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    # Cast modules_to_save (embed_tokens, lm_head) selectively:
    #
    # - embed_tokens: cast to COMPUTE_DTYPE (bfloat16) so its output
    #   feeds correctly into the bfloat16 attention layers.
    #
    # - lm_head: intentionally left in fp32.  prepare_model_for_kbit_training
    #   keeps all LayerNorm layers in fp32, so the hidden_states flowing
    #   into lm_head are fp32.  Casting lm_head to bfloat16 would create
    #   a float != bfloat16 mismatch.  Instead we register a forward
    #   pre-hook that auto-casts the input to the weight dtype so it is
    #   robust to any future changes.
    for name, module in model.named_modules():
        if "modules_to_save" in name and "embed_tokens" in name:
            module.to(dtype=COMPUTE_DTYPE)
            if hasattr(module, "modules_to_save"):
                for adapter_module in module.modules_to_save.values():
                    adapter_module.to(dtype=COMPUTE_DTYPE)

    # Register a forward pre-hook on each lm_head modules_to_save
    # wrapper so that its input is cast to the weight's dtype before
    # the linear operation, making it resilient to norm-layer upcasts.
    def _cast_input_to_weight_dtype(module, args):
        if not args:
            return args
        x = args[0]
        try:
            weight = next(
                p for p in module.parameters()
                if p.dtype.is_floating_point
            )
            if x.dtype != weight.dtype:
                x = x.to(dtype=weight.dtype)
        except StopIteration:
            pass
        return (x,) + args[1:]

    for name, module in model.named_modules():
        if "lm_head" in name and "modules_to_save" in name:
            module.register_forward_pre_hook(
                _cast_input_to_weight_dtype
            )

    print()

    print(
        "Gradient checkpointing: "
        "ENABLED"
    )

    model.print_trainable_parameters()

    # ------------------------------------------------------------------------
    # Verify embed_tokens and lm_head
    # ------------------------------------------------------------------------

    trainable_embedding_params = []

    embedding_param_count = 0

    for name, parameter in (
        model.named_parameters()
    ):

        if (
            "embed_tokens" in name
            or "lm_head" in name
        ):

            if parameter.requires_grad:

                trainable_embedding_params.append(
                    name
                )

                embedding_param_count += (
                    parameter.numel()
                )

    if not trainable_embedding_params:

        raise RuntimeError(
            "CRITICAL: embed_tokens/lm_head "
            "are not trainable after "
            "PEFT wrapping."
        )

    print(
        "Trainable embedding/output-head "
        "modules: PASS"
    )

    print(
        "Embedding/lm_head trainable "
        f"parameters: "
        f"{embedding_param_count:,}"
    )

    for name in (
        trainable_embedding_params[:10]
    ):

        print(
            f"  {name}"
        )

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()

    print(
        f"QLORA trainable params: "
        f"{trainable:,}"
    )

    print(
        f"Total visible params:   "
        f"{total:,}"
    )

    memory = print_cuda_memory(
        "Qwen + QLORA resident memory"
    )

    return (
        model,
        trainable,
        total,
        memory,
        embedding_param_count,
    )


# ============================================================================
# PROMPT / TOKEN ENCODING
# ============================================================================

PROMPT_TEXT = (
    "Transcribe the following Manipuri "
    "speech into Meitei Mayek script.\n"
    "Transcript:\n"
)


def encode_prompt(
    tokenizer,
) -> torch.Tensor:

    encoded = tokenizer(
        PROMPT_TEXT,

        add_special_tokens=True,

        return_tensors="pt",
    )

    return (
        encoded["input_ids"][0]
    )


def encode_target(
    tokenizer,
    text: str,
) -> torch.Tensor:

    text = text.strip()

    if not text:

        raise ValueError(
            "Transcript is empty."
        )

    ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    if not ids:

        raise ValueError(
            "Transcript produced "
            "zero tokens."
        )

    # Explicit EOS target.
    if (
        tokenizer.eos_token_id
        is not None
    ):

        ids.append(
            tokenizer.eos_token_id
        )

    ids = ids[
        :MAX_TARGET_TOKENS
    ]

    if len(ids) < 2:

        raise ValueError(
            "Transcript is too short "
            "after tokenization."
        )

    return torch.tensor(
        ids,
        dtype=torch.long,
    )


# ============================================================================
# BUILD MODEL INPUTS
# ============================================================================

def build_inputs_embeds_and_labels(
    model,
    tokenizer,
    projector: AudioProjector,
    cached_features: torch.Tensor,
    transcript: str,
):

    # ------------------------------------------------------------------------
    # Normalize cached features
    # ------------------------------------------------------------------------

    cached_features = (
        normalize_feature_shape(
            cached_features
        )
    )

    # ------------------------------------------------------------------------
    # Projector device / dtype
    # ------------------------------------------------------------------------

    projector_param = next(
        projector.parameters()
    )

    projector_device = (
        projector_param.device
    )

    projector_dtype = (
        projector_param.dtype
    )

    # Cached features are normally FP16
    # on disk. Convert directly to the
    # projector dtype/device.
    cached_features = (
        cached_features.to(
            device=projector_device,

            dtype=projector_dtype,
        )
    )

    if (
        cached_features.dtype
        != projector_dtype
    ):

        raise RuntimeError(
            "Projector input dtype mismatch.\n"
            f"Feature dtype: "
            f"{cached_features.dtype}\n"
            f"Projector dtype: "
            f"{projector_dtype}"
        )

    # ------------------------------------------------------------------------
    # AUDIO PROJECTOR
    # ------------------------------------------------------------------------

    audio_embeds = projector(
        cached_features
    )

    if audio_embeds.ndim != 3:

        raise RuntimeError(
            "Unexpected projector output.\n"
            f"Shape: "
            f"{audio_embeds.shape}"
        )

    # ------------------------------------------------------------------------
    # PROMPT
    # ------------------------------------------------------------------------

    prompt_ids = (
        encode_prompt(
            tokenizer
        )
        .to(DEVICE)
    )

    # ------------------------------------------------------------------------
    # TARGET
    # ------------------------------------------------------------------------

    target_ids = (
        encode_target(
            tokenizer,
            transcript,
        )
        .to(DEVICE)
    )

    # ------------------------------------------------------------------------
    # QWEN EMBEDDINGS
    # ------------------------------------------------------------------------

    embedding_layer = (
        model.get_input_embeddings()
    )

    prompt_embeds = (
        embedding_layer(
            prompt_ids.unsqueeze(0)
        )
    )

    target_embeds = (
        embedding_layer(
            target_ids.unsqueeze(0)
        )
    )

    if prompt_embeds.dtype != audio_embeds.dtype:
        prompt_embeds = prompt_embeds.to(dtype=audio_embeds.dtype)

    if target_embeds.dtype != audio_embeds.dtype:
        target_embeds = target_embeds.to(dtype=audio_embeds.dtype)

    # ------------------------------------------------------------------------
    # DEVICE CHECK
    # ------------------------------------------------------------------------

    if (
        audio_embeds.device
        != prompt_embeds.device
    ):

        raise RuntimeError(
            "Audio/text embedding "
            "device mismatch.\n"
            f"Audio: "
            f"{audio_embeds.device}\n"
            f"Text: "
            f"{prompt_embeds.device}"
        )

    # ------------------------------------------------------------------------
    # DTYPE CHECK
    # ------------------------------------------------------------------------

    if (
        audio_embeds.dtype
        != prompt_embeds.dtype
    ):

        raise RuntimeError(
            "Audio/text embedding "
            "dtype mismatch.\n"
            f"Audio: "
            f"{audio_embeds.dtype}\n"
            f"Text: "
            f"{prompt_embeds.dtype}"
        )

    # ------------------------------------------------------------------------
    # CONCATENATE
    # ------------------------------------------------------------------------

    inputs_embeds = torch.cat(
        [
            audio_embeds,
            prompt_embeds,
            target_embeds,
        ],
        dim=1,
    )

    # Final dtype guard: ensure the concatenated embedding is in the
    # compute dtype expected by the quantised model's layers.
    if inputs_embeds.dtype != COMPUTE_DTYPE:
        inputs_embeds = inputs_embeds.to(dtype=COMPUTE_DTYPE)

    audio_len = (
        audio_embeds.shape[1]
    )

    prompt_len = (
        prompt_embeds.shape[1]
    )

    target_len = (
        target_ids.shape[0]
    )

    # ------------------------------------------------------------------------
    # LABELS
    #
    # HF causal LM performs the internal
    # causal shift.
    #
    # Do NOT manually shift here.
    # ------------------------------------------------------------------------

    labels = torch.full(
        (
            1,
            inputs_embeds.shape[1],
        ),
        -100,
        dtype=torch.long,
        device=DEVICE,
    )

    labels[
        0,
        audio_len + prompt_len:
    ] = target_ids

    # ------------------------------------------------------------------------
    # ATTENTION MASK
    # ------------------------------------------------------------------------

    attention_mask = torch.ones(
        (
            1,
            inputs_embeds.shape[1],
        ),
        dtype=torch.long,
        device=DEVICE,
    )

    # ------------------------------------------------------------------------
    # TOKEN INFORMATION
    # ------------------------------------------------------------------------

    token_info = {

        "audio_tokens":
            int(audio_len),

        "prompt_tokens":
            int(prompt_len),

        "target_tokens":
            int(target_len),

        "total_tokens":
            int(
                inputs_embeds.shape[1]
            ),

        "projector_input_dtype":
            str(
                cached_features.dtype
            ),

        "projector_dtype":
            str(
                projector_dtype
            ),

        "audio_embedding_dtype":
            str(
                audio_embeds.dtype
            ),

        "qwen_embedding_dtype":
            str(
                prompt_embeds.dtype
            ),

        "input_embedding_dtype":
            str(
                inputs_embeds.dtype
            ),
    }

    # IMPORTANT:
    # Exactly FOUR return values.
    return (
        inputs_embeds,
        attention_mask,
        labels,
        token_info,
    )


# ============================================================================
# GRADIENT HELPERS
# ============================================================================

def gradient_norm(
    module: nn.Module,
) -> float:

    total_sq = 0.0

    for parameter in (
        module.parameters()
    ):

        if parameter.grad is None:

            continue

        grad = (
            parameter
            .grad
            .detach()
        )

        try:

            finite = (
                torch.isfinite(
                    grad
                ).all()
            )

        except Exception:

            finite = (
                torch.isfinite(
                    grad.float()
                ).all()
            )

        if not bool(finite):

            return float("nan")

        value = (
            grad.float()
            .norm(2)
            .item()
        )

        total_sq += (
            value * value
        )

    return math.sqrt(
        total_sq
    )


def gradient_norm_by_name(
    module: nn.Module,
    keywords: list[str],
) -> float:

    """
    Calculate gradient norm only for
    parameters whose names contain one
    of the requested keywords.

    Used to verify that the newly
    trainable Meitei Mayek embedding
    and output-head modules actually
    receive gradients.
    """

    total_sq = 0.0

    found = False

    for name, parameter in (
        module.named_parameters()
    ):

        if not any(
            keyword in name
            for keyword in keywords
        ):

            continue

        if not parameter.requires_grad:

            continue

        found = True

        if parameter.grad is None:

            continue

        try:

            grad = (
                parameter
                .grad
                .detach()
                .float()
            )

            if not torch.isfinite(
                grad
            ).all():

                return float("nan")

            value = (
                grad
                .norm(2)
                .item()
            )

        except Exception:

            return float("nan")

        total_sq += (
            value * value
        )

    if not found:

        return float("nan")

    return math.sqrt(
        total_sq
    )


def parameter_nan_inf_check(
    module: nn.Module,
):

    bad = []

    for name, parameter in (
        module.named_parameters()
    ):

        try:

            data = (
                parameter.detach()
            )

            try:

                finite = (
                    torch.isfinite(
                        data
                    ).all()
                )

            except Exception:

                finite = (
                    torch.isfinite(
                        data.float()
                    ).all()
                )

            if not bool(finite):

                bad.append(
                    name
                )

        except Exception:

            # Quantized parameters may
            # not always expose a normal
            # tensor interface.
            continue

    return (
        len(bad) == 0,
        bad,
    )


# ============================================================================
# STAGE 1
# ONE EXAMPLE FORWARD + BACKWARD
# ============================================================================

def stage_one(
    model,
    tokenizer,
    projector,
    cache,
    record,
):

    print_header(
        "STAGE 1: ONE-EXAMPLE "
        "FORWARD + BACKWARD"
    )

    cleanup_cuda()

    reset_cuda_peak()

    model.train()

    projector.train()

    model.zero_grad(
        set_to_none=True
    )

    projector.zero_grad(
        set_to_none=True
    )

    # ------------------------------------------------------------------------
    # CACHE ONLY
    # ------------------------------------------------------------------------

    features = cache_get(
        cache,
        record,
    )

    if features is None:

        raise RuntimeError(
            "Stage-1 sample has no "
            "cached feature.\n"
            f"Record index: "
            f"{record_index(record)}"
        )

    # ------------------------------------------------------------------------
    # BUILD INPUTS
    # ------------------------------------------------------------------------

    (
        inputs_embeds,
        attention_mask,
        labels,
        token_info,
    ) = build_inputs_embeds_and_labels(
        model=model,

        tokenizer=tokenizer,

        projector=projector,

        cached_features=features,

        transcript=record_text(
            record
        ),
    )

    print(
        f"Audio tokens:   "
        f"{token_info['audio_tokens']}"
    )

    print(
        f"Prompt tokens:  "
        f"{token_info['prompt_tokens']}"
    )

    print(
        f"Target tokens:  "
        f"{token_info['target_tokens']}"
    )

    print(
        f"Total sequence: "
        f"{token_info['total_tokens']}"
    )

    print(
        "Projector input dtype: "
        f"'{token_info['projector_input_dtype']}'"
    )

    print(
        "Projector dtype: "
        f"{token_info['projector_dtype']}"
    )

    print(
        "Audio embedding dtype: "
        f"{token_info['audio_embedding_dtype']}"
    )

    print(
        "Qwen embedding dtype: "
        f"{token_info['qwen_embedding_dtype']}"
    )

    print()

    # ------------------------------------------------------------------------
    # FORWARD
    # ------------------------------------------------------------------------

    print(
        "Running forward pass..."
    )

    start = time.perf_counter()

    outputs = model(
        inputs_embeds=
            inputs_embeds,

        attention_mask=
            attention_mask,

        labels=
            labels,

        use_cache=False,
    )

    loss = outputs.loss

    if loss is None:

        raise RuntimeError(
            "Model returned no loss."
        )

    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            "Stage-1 loss is NaN/Inf."
        )

    loss_value = (
        loss
        .detach()
        .float()
        .item()
    )

    print(
        f"Forward loss: "
        f"{loss_value:.6f}"
    )

    print()

    # ------------------------------------------------------------------------
    # BACKWARD
    # ------------------------------------------------------------------------

    print(
        "Running backward pass..."
    )

    loss.backward()

    # ------------------------------------------------------------------------
    # EMBEDDING / LM HEAD GRADIENT
    # ------------------------------------------------------------------------

    embedding_grad = (
        gradient_norm_by_name(
            model,
            [
                "embed_tokens",
                "lm_head",
            ],
        )
    )

    print(
        "Embedding/lm_head grad norm: "
        f"{embedding_grad:.6f}"
    )

    if (
        not math.isfinite(
            embedding_grad
        )
        or embedding_grad <= 0
    ):

        raise RuntimeError(
            "Meitei Mayek embedding/lm_head "
            "gradients are missing or zero."
        )

    print(
        "Meitei Mayek embedding gradient: "
        "PASS"
    )

    # ------------------------------------------------------------------------
    # GENERAL GRADIENTS
    # ------------------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - start
    )

    projector_grad = (
        gradient_norm(
            projector
        )
    )

    qlora_grad = (
        gradient_norm(
            model
        )
    )

    print(
        f"Backward time: "
        f"{elapsed:.2f} sec"
    )

    print(
        f"Projector grad norm: "
        f"{projector_grad:.6f}"
    )

    print(
        f"QLORA grad norm: "
        f"{qlora_grad:.6f}"
    )

    if (
        not math.isfinite(
            projector_grad
        )
        or projector_grad <= 0
    ):

        raise RuntimeError(
            "Projector gradients are "
            "missing, zero, NaN, or Inf."
        )

    if (
        not math.isfinite(
            qlora_grad
        )
        or qlora_grad <= 0
    ):

        raise RuntimeError(
            "QLORA gradients are "
            "missing, zero, NaN, or Inf."
        )

    # ------------------------------------------------------------------------
    # PARAMETER FINITE CHECK
    # ------------------------------------------------------------------------

    model_ok, model_bad = (
        parameter_nan_inf_check(
            model
        )
    )

    projector_ok, projector_bad = (
        parameter_nan_inf_check(
            projector
        )
    )

    if not model_ok:

        raise RuntimeError(
            "NaN/Inf in model parameters:\n"
            f"{model_bad[:10]}"
        )

    if not projector_ok:

        raise RuntimeError(
            "NaN/Inf in projector parameters:\n"
            f"{projector_bad[:10]}"
        )

    peak = print_cuda_memory(
        "Stage-1 memory"
    )

    print()

    print(
        "STAGE 1 BACKWARD: PASS"
    )

    print(
        "Non-zero projector gradients: PASS"
    )

    print(
        "Non-zero QLORA gradients: PASS"
    )

    print(
        "Non-zero embedding gradients: PASS"
    )

    print(
        "Finite parameters: PASS"
    )

    return {

        "loss":
            float(
                loss_value
            ),

        "projector_grad_norm":
            float(
                projector_grad
            ),

        "qlora_grad_norm":
            float(
                qlora_grad
            ),

        "embedding_lm_head_grad_norm":
            float(
                embedding_grad
            ),

        "elapsed_sec":
            float(
                elapsed
            ),

        "token_info":
            token_info,

        "cuda_memory":
            peak,
    }


# ============================================================================
# STAGE 2
# 20-SAMPLE OVERFIT TEST
# ============================================================================

def select_overfit_records(
    train,
    limit:
        int = OVERFIT_SAMPLES,
):

    if len(train) < limit:

        raise RuntimeError(
            f"Need at least "
            f"{limit} training samples."
        )

    return list(
        train[:limit]
    )


def stage_two_overfit(
    model,
    tokenizer,
    projector,
    cache,
    records,
):

    print_header(
        "STAGE 2: 20-SAMPLE "
        "OVERFIT TEST"
    )

    model.train()

    projector.train()

    trainable_parameters = [

        parameter

        for parameter in (

            list(
                projector.parameters()
            )

            +

            list(
                model.parameters()
            )
        )

        if parameter.requires_grad
    ]

    if not trainable_parameters:

        raise RuntimeError(
            "No trainable parameters "
            "found."
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters,

        lr=OVERFIT_LR,

        weight_decay=
            OVERFIT_WEIGHT_DECAY,
    )

    history = []

    epoch_times = []

    for epoch in range(
        1,
        OVERFIT_EPOCHS + 1,
    ):

        epoch_start = (
            time.perf_counter()
        )

        losses = []

        print()

        print(
            f"Overfit epoch "
            f"{epoch}/"
            f"{OVERFIT_EPOCHS}"
        )

        for step, record in enumerate(
            records,
            start=1,
        ):

            optimizer.zero_grad(
                set_to_none=True
            )

            # ---------------------------------------------------------------
            # CACHE ONLY
            # ---------------------------------------------------------------

            features = cache_get(
                cache,
                record,
            )

            if features is None:

                raise RuntimeError(
                    "Missing cached feature "
                    f"for record "
                    f"{record_index(record)}"
                )

            # ---------------------------------------------------------------
            # INPUT BUILD
            # ---------------------------------------------------------------

            (
                inputs_embeds,
                attention_mask,
                labels,
                token_info,
            ) = build_inputs_embeds_and_labels(
                model=model,

                tokenizer=tokenizer,

                projector=projector,

                cached_features=features,

                transcript=record_text(
                    record
                ),
            )

            # ---------------------------------------------------------------
            # FORWARD
            # ---------------------------------------------------------------

            outputs = model(
                inputs_embeds=
                    inputs_embeds,

                attention_mask=
                    attention_mask,

                labels=
                    labels,

                use_cache=False,
            )

            loss = outputs.loss

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "NaN/Inf loss during "
                    "overfit.\n"
                    f"Epoch: {epoch}\n"
                    f"Step: {step}\n"
                    f"Record: "
                    f"{record_index(record)}"
                )

            # ---------------------------------------------------------------
            # BACKWARD
            # ---------------------------------------------------------------

            loss.backward()

            # ---------------------------------------------------------------
            # PROJECTOR GRADIENT
            # ---------------------------------------------------------------

            projector_grad = (
                gradient_norm(
                    projector
                )
            )

            if (
                not math.isfinite(
                    projector_grad
                )
                or projector_grad <= 0
            ):

                raise RuntimeError(
                    "Invalid projector gradient "
                    "during overfit.\n"
                    f"Epoch: {epoch}\n"
                    f"Step: {step}\n"
                    f"Grad: "
                    f"{projector_grad}"
                )

            # ---------------------------------------------------------------
            # EMBEDDING GRADIENT
            # ---------------------------------------------------------------

            embedding_grad = (
                gradient_norm_by_name(
                    model,
                    [
                        "embed_tokens",
                        "lm_head",
                    ],
                )
            )

            if (
                not math.isfinite(
                    embedding_grad
                )
                or embedding_grad <= 0
            ):

                raise RuntimeError(
                    "Invalid embedding/lm_head "
                    "gradient during overfit.\n"
                    f"Epoch: {epoch}\n"
                    f"Step: {step}\n"
                    f"Grad: "
                    f"{embedding_grad}"
                )

            # ---------------------------------------------------------------
            # OPTIMIZER
            # ---------------------------------------------------------------

            optimizer.step()

            loss_value = (
                loss
                .detach()
                .float()
                .item()
            )

            losses.append(
                loss_value
            )

            if (
                step == 1
                or step == len(records)
            ):

                print(
                    f"  step "
                    f"{step:02d}/"
                    f"{len(records)} "
                    f"loss="
                    f"{loss_value:.6f} "
                    f"projector_grad="
                    f"{projector_grad:.6f} "
                    f"embedding_grad="
                    f"{embedding_grad:.6f}"
                )

            # ---------------------------------------------------------------
            # RELEASE TEMPORARY COMPUTATION GRAPH
            # ---------------------------------------------------------------

            del outputs

            del loss

            del inputs_embeds

            del attention_mask

            del labels

            # Aggressive cleanup is intentional
            # for the 4-GB PREFLIGHT only.
            cleanup_cuda()

        average_loss = (
            sum(losses)
            / len(losses)
        )

        epoch_time = (
            time.perf_counter()
            - epoch_start
        )

        epoch_times.append(
            float(
                epoch_time
            )
        )

        history.append(
            float(
                average_loss
            )
        )

        print()

        print(
            f"Overfit epoch {epoch}: "
            f"avg loss = "
            f"{average_loss:.6f}"
        )

        print(
            f"Epoch time: "
            f"{epoch_time:.2f} sec"
        )

        print_cuda_memory(
            f"After overfit epoch {epoch}"
        )

    print()

    print(
        "Overfit loss history:"
    )

    print(
        history
    )

    # ------------------------------------------------------------------------
    # LOSS VALIDATION
    # ------------------------------------------------------------------------

    if not all(
        math.isfinite(
            value
        )
        for value in history
    ):

        raise RuntimeError(
            "Overfit history contains "
            "NaN/Inf."
        )

    overall_decrease = (
        history[-1]
        < history[0]
    )

    monotonic = all(
        history[index + 1]
        <= history[index]

        for index in range(
            len(history) - 1
        )
    )

    if not overall_decrease:

        raise RuntimeError(
            "20-sample overfit did not "
            "show an overall loss decrease.\n"
            f"History: {history}"
        )

    print()

    print(
        "Overall overfit loss decrease: "
        "PASS"
    )

    if monotonic:

        print(
            "Monotonic epoch-to-epoch "
            "decrease: PASS"
        )

    else:

        print(
            "Monotonic epoch-to-epoch "
            "decrease: NOT STRICT"
        )

        print(
            "Overall decrease is accepted "
            "as the learning-signal check."
        )

    return {

        "loss_history":
            history,

        "overall_decrease":
            bool(
                overall_decrease
            ),

        "monotonic":
            bool(
                monotonic
            ),

        "epoch_times_sec":
            epoch_times,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():

    total_start = (
        time.perf_counter()
    )

    result: dict[str, Any] = {

        "script":
            str(
                THIS_FILE
            ),

        "project_root":
            str(
                PROJECT_ROOT
            ),

        "v2_dir":
            str(
                V2_DIR
            ),

        "cache_dir":
            str(
                CACHE_DIR
            ),

        "model_name":
            QWEN_MODEL_NAME,

        "device":
            str(
                DEVICE
            ),

        "compute_dtype":
            str(
                COMPUTE_DTYPE
            ),

        "encoder_hidden":
            ENCODER_HIDDEN,

        "llm_hidden":
            LLM_HIDDEN,

        "projector_expected_params":
            EXPECTED_PROJECTOR_PARAMS,

        "lora_r":
            LORA_R,

        "lora_alpha":
            LORA_ALPHA,

        "lora_dropout":
            LORA_DROPOUT,

        "lora_target_modules":
            LORA_TARGET_MODULES,

        "lora_modules_to_save":
            LORA_MODULES_TO_SAVE,

        "status":
            "STARTING",
    }

    try:

        # ====================================================================
        # 1. HARDWARE
        # ====================================================================

        hardware = (
            inspect_hardware()
        )

        result[
            "hardware"
        ] = hardware

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA is unavailable."
            )

        # ====================================================================
        # 2. DATASET
        # ====================================================================

        (
            manager,
            train,
            val,
            test,
        ) = load_datasets()

        result[
            "dataset"
        ] = {

            "train":
                len(train),

            "val":
                len(val),

            "test":
                len(test),
        }

        # ====================================================================
        # 3. FEATURE CACHE
        # ====================================================================

        cache = (
            load_feature_cache()
        )

        cache_validation = (
            validate_real_cache(
                cache,
                train,
                val,
            )
        )

        result[
            "cache_validation"
        ] = cache_validation

        # ====================================================================
        # 4. PROJECTOR
        # ====================================================================

        projector = (
            AudioProjector(
                encoder_hidden=
                    ENCODER_HIDDEN,

                llm_hidden=
                    LLM_HIDDEN,
            )
            .to(
                DEVICE,
                dtype=COMPUTE_DTYPE,
            )
        )

        projector_info = (
            inspect_projector(
                projector
            )
        )

        result[
            "projector"
        ] = projector_info

        # ====================================================================
        # 5. TOKENIZER
        # ====================================================================

        tokenizer = (
            load_tokenizer(
                train,
                val,
            )
        )

        result[
            "tokenizer"
        ] = {

            "vocab_size":
                len(tokenizer),

            "pad_token_id":
                tokenizer.pad_token_id,

            "eos_token_id":
                tokenizer.eos_token_id,
        }

        # ====================================================================
        # 6. QWEN + QLORA
        # ====================================================================

        (
            model,
            qlora_trainable,
            model_total,
            resident_memory,
            embedding_param_count,
        ) = load_qwen_qlora(
            tokenizer
        )

        result[
            "qlora"
        ] = {

            "trainable_params":
                qlora_trainable,

            "total_params":
                model_total,

            "embedding_lm_head_trainable_params":
                embedding_param_count,

            "gradient_checkpointing":
                True,

            "quantization":
                "4-bit NF4",

            "double_quant":
                True,

            "compute_dtype":
                str(
                    COMPUTE_DTYPE
                ),

            "modules_to_save":
                LORA_MODULES_TO_SAVE,
        }

        result[
            "resident_memory"
        ] = resident_memory

        # ====================================================================
        # 7. STAGE 1
        # ====================================================================

        stage1_record = (
            train[0]
        )

        print()

        print(
            "Stage-1 record:"
        )

        print(
            f"index="
            f"{record_index(stage1_record)}"
        )

        print(
            f"audio="
            f"{record_audio_path(stage1_record)}"
        )

        print(
            f"text="
            f"{record_text(stage1_record)}"
        )

        stage1 = stage_one(
            model=model,

            tokenizer=tokenizer,

            projector=projector,

            cache=cache,

            record=stage1_record,
        )

        result[
            "stage1"
        ] = stage1

        # ====================================================================
        # 8. STAGE 2
        # ====================================================================

        overfit_records = (
            select_overfit_records(
                train,
                OVERFIT_SAMPLES,
            )
        )

        stage2 = (
            stage_two_overfit(
                model=model,

                tokenizer=tokenizer,

                projector=projector,

                cache=cache,

                records=
                    overfit_records,
            )
        )

        result[
            "stage2"
        ] = stage2

        # ====================================================================
        # 9. FINAL MEMORY
        # ====================================================================

        cleanup_cuda()

        result[
            "final_memory"
        ] = print_cuda_memory(
            "Final preflight memory"
        )

        result[
            "elapsed_total_sec"
        ] = (
            time.perf_counter()
            - total_start
        )

        result[
            "status"
        ] = "PASS"

        print_header(
            "CACHED QLORA PREFLIGHT: PASS"
        )

        print(
            f"Projector params: "
            f"{projector_info['trainable_params']:,}"
        )

        print(
            f"QLORA trainable params: "
            f"{qlora_trainable:,}"
        )

        print(
            f"Embedding/lm_head params: "
            f"{embedding_param_count:,}"
        )

        print(
            f"Stage-1 loss: "
            f"{stage1['loss']:.6f}"
        )

        print(
            f"Projector grad norm: "
            f"{stage1['projector_grad_norm']:.6f}"
        )

        print(
            f"QLORA grad norm: "
            f"{stage1['qlora_grad_norm']:.6f}"
        )

        print(
            "Embedding/lm_head grad norm: "
            f"{stage1['embedding_lm_head_grad_norm']:.6f}"
        )

        print(
            "Stage-1 peak VRAM: "
            f"{stage1['cuda_memory']['peak_mb']:.1f} MB"
        )

        print(
            "Overfit loss history: "
            f"{stage2['loss_history']}"
        )

        print(
            "Projector gradients: PASS"
        )

        print(
            "QLORA gradients: PASS"
        )

        print(
            "Embedding gradients: PASS"
        )

        print(
            "20-sample learning signal: PASS"
        )

        print(
            "601-row test protection: PASS"
        )

    except Exception as exc:

        result[
            "status"
        ] = "FAIL"

        result[
            "error"
        ] = {

            "type":
                type(exc).__name__,

            "message":
                str(exc),

            "traceback":
                traceback.format_exc(),
        }

        print_header(
            "CACHED QLORA PREFLIGHT: FAIL"
        )

        print(
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        traceback.print_exc()

    finally:

        try:

            OUTPUT_JSON.write_text(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

            print()

            print(
                "Diagnostic JSON:"
            )

            print(
                OUTPUT_JSON
            )

        except Exception as write_error:

            print(
                "WARNING: Failed to write "
                "diagnostic JSON."
            )

            print(
                write_error
            )

        cleanup_cuda()

    if (
        result["status"]
        != "PASS"
    ):

        raise SystemExit(1)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()