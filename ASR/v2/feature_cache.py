"""
V2 Frozen Encoder Feature Cache
================================

Caches outputs of the frozen Wav2Vec2-BERT encoder.

Architecture:

    Audio
      |
      v
    Frozen Wav2Vec2-BERT
      |
      v
    [FEATURE CACHE]  <-- this module
      |
      v
    Trainable Audio Projector
      |
      v
    Qwen2.5-3B QLoRA

Design goals
------------
1. Never modify train.csv / val.csv / test.csv.
2. Cache ONLY frozen encoder features.
3. Use disk as the persistent source of truth.
4. Keep RAM cache bounded.
5. Use atomic writes to avoid corrupted cache files.
6. Validate cached tensors before using them.
7. Detect configuration/model changes through a cache fingerprint.
8. Support lazy caching: features are generated only when needed.
9. Support batch precomputation later.
10. Never silently use a stale cache.

This module does NOT decide how much VRAM is available.
That responsibility belongs to ResourceController.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


import torch

try:
    from .data_manager import DataManager, AudioRecord
except ImportError:
    from data_manager import DataManager, AudioRecord


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureCacheConfig:
    cache_dir: str = "ASR/v2/cache/features"

    # Maximum number of feature tensors held in RAM.
    max_ram_items: int = 32

    # Store tensors in CPU memory on disk.
    disk_dtype: str = "float16"

    # Version the cache format.
    cache_version: str = "v1"

    # Whether to use the RAM LRU cache.
    enable_ram_cache: bool = True

    # Whether disk cache is enabled.
    enable_disk_cache: bool = True


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------

class FrozenEncoderFeatureCache:

    def __init__(
        self,
        config: FeatureCacheConfig | None = None,
        project_dir: str | Path | None = None,
    ):
        self.config = config or FeatureCacheConfig()

        if project_dir is None:
            # scheduler.py / feature_cache.py:
            # ASR/v2/<file>
            # project root = parent of ASR
            self.project_dir = (
                Path(__file__).resolve().parents[2]
            )
        else:
            self.project_dir = Path(project_dir).resolve()

        configured_cache = Path(
            self.config.cache_dir
        )

        if configured_cache.is_absolute():
            self.cache_dir = configured_cache
        else:
            self.cache_dir = (
                self.project_dir / configured_cache
            )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # OrderedDict provides a simple LRU cache.
        self.ram_cache: OrderedDict[
            str,
            torch.Tensor,
        ] = OrderedDict()

        self.stats = {
            "ram_hits": 0,
            "ram_misses": 0,
            "disk_hits": 0,
            "disk_misses": 0,
            "encoder_computations": 0,
            "writes": 0,
            "invalid_entries": 0,
            "evictions": 0,
        }

    # -----------------------------------------------------------------
    # Basic utilities
    # -----------------------------------------------------------------

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    # -----------------------------------------------------------------
    # Cache fingerprint
    # -----------------------------------------------------------------

    def build_fingerprint(
        self,
        model_name: str,
        sampling_rate: int,
        feature_extractor_config: Optional[dict] = None,
    ) -> str:

        payload = {
            "cache_version": self.config.cache_version,
            "model_name": model_name,
            "sampling_rate": int(sampling_rate),
            "feature_extractor_config":
                feature_extractor_config or {},
            "disk_dtype": self.config.disk_dtype,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        )

        return self._sha256(serialized)

    # -----------------------------------------------------------------
    # Record key
    # -----------------------------------------------------------------

    def build_key(
        self,
        record,
        model_fingerprint: str,
    ) -> str:

        """
        Generate a stable cache key.

        We intentionally include:
        - resolved audio path
        - file size
        - modification timestamp
        - transcription-independent record index
        - encoder fingerprint

        If the audio file changes, the cache becomes invalid.
        If the encoder configuration changes, the cache becomes invalid.
        """

        audio_path = Path(
            record.audio_path
        ).resolve()

        try:
            stat = audio_path.stat()

            file_size = stat.st_size
            mtime_ns = stat.st_mtime_ns

        except FileNotFoundError:
            file_size = -1
            mtime_ns = -1

        payload = {
            "record_index": int(record.index),
            "audio_path": str(audio_path),
            "file_size": file_size,
            "mtime_ns": mtime_ns,
            "model_fingerprint":
                model_fingerprint,
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
        )

        return self._sha256(serialized)

    # -----------------------------------------------------------------
    # Cache file
    # -----------------------------------------------------------------

    def cache_path(
        self,
        key: str,
    ) -> Path:

        return self.cache_dir / f"{key}.pt"

    # -----------------------------------------------------------------
    # Metadata file
    # -----------------------------------------------------------------

    def metadata_path(
        self,
        key: str,
    ) -> Path:

        return self.cache_dir / f"{key}.json"

    # -----------------------------------------------------------------
    # RAM cache
    # -----------------------------------------------------------------

    def _ram_get(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:

        if not self.config.enable_ram_cache:
            return None

        if key not in self.ram_cache:
            self.stats["ram_misses"] += 1
            return None

        tensor = self.ram_cache.pop(key)

        # Reinsert at end = most recently used.
        self.ram_cache[key] = tensor

        self.stats["ram_hits"] += 1

        return tensor

    def _ram_put(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> None:

        if not self.config.enable_ram_cache:
            return

        # Store CPU tensor only.
        tensor = tensor.detach().cpu().contiguous()

        if key in self.ram_cache:
            self.ram_cache.pop(key)

        self.ram_cache[key] = tensor

        while (
            len(self.ram_cache)
            > self.config.max_ram_items
        ):
            self.ram_cache.popitem(
                last=False
            )

            self.stats["evictions"] += 1

    # -----------------------------------------------------------------
    # Disk save
    # -----------------------------------------------------------------

    def _disk_dtype(
        self,
    ) -> torch.dtype:

        if self.config.disk_dtype == "float16":
            return torch.float16

        if self.config.disk_dtype == "float32":
            return torch.float32

        if self.config.disk_dtype == "bfloat16":
            return torch.bfloat16

        raise ValueError(
            f"Unsupported disk dtype: "
            f"{self.config.disk_dtype}"
        )

    def _atomic_torch_save(
        self,
        payload: dict,
        destination: Path,
    ) -> None:

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = destination.with_suffix(
            ".tmp"
        )

        try:

            torch.save(
                payload,
                temporary,
            )

            # Atomic replacement.
            os.replace(
                temporary,
                destination,
            )

        finally:

            if temporary.exists():
                try:
                    temporary.unlink()
                except Exception:
                    pass

    # -----------------------------------------------------------------
    # Disk read
    # -----------------------------------------------------------------

    def _disk_get(
        self,
        key: str,
    ) -> Optional[torch.Tensor]:

        if not self.config.enable_disk_cache:
            return None

        path = self.cache_path(key)

        if not path.exists():
            self.stats["disk_misses"] += 1
            return None

        try:

            payload = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
            )

            if not isinstance(payload, dict):
                raise ValueError(
                    "Cache payload is not a dictionary."
                )

            if payload.get(
                "cache_version"
            ) != self.config.cache_version:

                raise ValueError(
                    "Cache version mismatch."
                )

            tensor = payload.get(
                "features"
            )

            if not isinstance(
                tensor,
                torch.Tensor,
            ):
                raise ValueError(
                    "Cached features are not a tensor."
                )

            if tensor.ndim != 3:
                raise ValueError(
                    "Expected encoder features "
                    f"with 3 dimensions, got "
                    f"{tensor.ndim}."
                )

            if tensor.numel() == 0:
                raise ValueError(
                    "Cached tensor is empty."
                )

            tensor = tensor.detach().cpu()

            self.stats["disk_hits"] += 1

            return tensor

        except Exception as exc:

            self.stats["invalid_entries"] += 1

            print(
                f"[CACHE] Invalid entry removed: "
                f"{path.name}"
            )
            print(
                f"        Reason: {exc}"
            )

            try:
                path.unlink()
            except Exception:
                pass

            metadata = self.metadata_path(key)

            if metadata.exists():
                try:
                    metadata.unlink()
                except Exception:
                    pass

            return None

    # -----------------------------------------------------------------
    # Disk write
    # -----------------------------------------------------------------

    def _disk_put(
        self,
        key: str,
        tensor: torch.Tensor,
        metadata: Optional[dict] = None,
    ) -> None:

        if not self.config.enable_disk_cache:
            return

        tensor_cpu = (
            tensor
            .detach()
            .cpu()
            .contiguous()
        )

        target_dtype = self._disk_dtype()

        if tensor_cpu.dtype != target_dtype:
            tensor_cpu = tensor_cpu.to(
                target_dtype
            )

        payload = {
            "cache_version":
                self.config.cache_version,
            "created_at":
                time.time(),
            "shape":
                list(tensor_cpu.shape),
            "dtype":
                str(tensor_cpu.dtype),
            "features":
                tensor_cpu,
        }

        self._atomic_torch_save(
            payload,
            self.cache_path(key),
        )

        metadata_payload = {
            "cache_version":
                self.config.cache_version,
            "key":
                key,
            "created_at":
                time.time(),
            "shape":
                list(tensor_cpu.shape),
            "dtype":
                str(tensor_cpu.dtype),
        }

        if metadata:
            metadata_payload.update(
                metadata
            )

        metadata_path = self.metadata_path(
            key
        )

        temporary = metadata_path.with_suffix(
            ".tmp"
        )

        try:

            with open(
                temporary,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    metadata_payload,
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

                f.flush()
                os.fsync(f.fileno())

            os.replace(
                temporary,
                metadata_path,
            )

        finally:

            if temporary.exists():
                try:
                    temporary.unlink()
                except Exception:
                    pass

        self.stats["writes"] += 1

    # -----------------------------------------------------------------
    # Encoder computation
    # -----------------------------------------------------------------

    @torch.no_grad()
    def compute_features(
        self,
        encoder,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:

        """
        Compute frozen encoder features.

        The encoder remains in evaluation mode and gradients are disabled.

        Expected input:
            [batch, audio_samples]

        Expected output:
            [batch, time, hidden]
        """

        encoder.eval()

        if device is None:
            device = next(
                encoder.parameters()
            ).device

        input_values = input_values.to(
            device,
            non_blocking=True,
        )

        if attention_mask is not None:
            attention_mask = attention_mask.to(
                device,
                non_blocking=True,
            )

        if attention_mask is None:

            outputs = encoder(
                input_values=input_values
            )

        else:

            outputs = encoder(
                input_values=input_values,
                attention_mask=attention_mask,
            )

        # Wav2Vec2-BERT style models normally expose
        # last_hidden_state.
        if hasattr(
            outputs,
            "last_hidden_state",
        ):

            features = (
                outputs.last_hidden_state
            )

        elif isinstance(outputs, dict):

            features = outputs[
                "last_hidden_state"
            ]

        else:

            raise RuntimeError(
                "Unable to find "
                "last_hidden_state in encoder output."
            )

        if not isinstance(
            features,
            torch.Tensor,
        ):
            raise RuntimeError(
                "Encoder did not return a tensor."
            )

        if features.ndim != 3:
            raise RuntimeError(
                "Expected encoder output "
                "[batch,time,hidden], got "
                f"{tuple(features.shape)}"
            )

        self.stats[
            "encoder_computations"
        ] += int(
            input_values.shape[0]
        )

        return features.detach()

    # -----------------------------------------------------------------
    # Get or compute
    # -----------------------------------------------------------------

    def get_or_compute(
        self,
        record,
        model_fingerprint: str,
        encoder,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        device: Optional[torch.device | str] = None,
        metadata: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, bool]:

        """
        Returns:
            features, cache_hit

        Search order:

            RAM
             ↓
            Disk
             ↓
            Encoder computation
             ↓
            Disk write
             ↓
            RAM insertion
        """

        key = self.build_key(
            record,
            model_fingerprint,
        )

        # -------------------------------------------------------------
        # 1. RAM
        # -------------------------------------------------------------

        tensor = self._ram_get(key)

        if tensor is not None:
            return tensor, True

        # -------------------------------------------------------------
        # 2. Disk
        # -------------------------------------------------------------

        tensor = self._disk_get(key)

        if tensor is not None:

            self._ram_put(
                key,
                tensor,
            )

            return tensor, True

        # -------------------------------------------------------------
        # 3. Encoder
        # -------------------------------------------------------------

        tensor = self.compute_features(
            encoder=encoder,
            input_values=input_values,
            attention_mask=attention_mask,
            device=device,
        )

        # -------------------------------------------------------------
        # 4. Disk
        # -------------------------------------------------------------

        self._disk_put(
            key,
            tensor,
            metadata=metadata,
        )

        # -------------------------------------------------------------
        # 5. RAM
        # -------------------------------------------------------------

        self._ram_put(
            key,
            tensor,
        )

        return (
            tensor.detach().cpu(),
            False,
        )

    # -----------------------------------------------------------------
    # Cache statistics
    # -----------------------------------------------------------------

    def statistics(self) -> dict:

        stats = dict(
            self.stats
        )

        stats["ram_items"] = len(
            self.ram_cache
        )

        stats["ram_capacity"] = (
            self.config.max_ram_items
        )

        disk_files = list(
            self.cache_dir.glob("*.pt")
        )

        stats["disk_items"] = len(
            disk_files
        )

        return stats

    # -----------------------------------------------------------------
    # Human-readable report
    # -----------------------------------------------------------------

    def print_statistics(self) -> None:

        stats = self.statistics()

        print("\n" + "=" * 72)
        print("FEATURE CACHE STATISTICS")
        print("=" * 72)

        for key, value in stats.items():
            print(
                f"{key:28s}: {value}"
            )

        print("=" * 72)

    # -----------------------------------------------------------------
    # Cache inspection
    # -----------------------------------------------------------------

    def disk_size_bytes(self) -> int:

        total = 0

        for path in self.cache_dir.glob("*"):
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass

        return total

    def disk_size_gb(self) -> float:

        return (
            self.disk_size_bytes()
            / (1024 ** 3)
        )

    # -----------------------------------------------------------------
    # Clear operations
    # -----------------------------------------------------------------

    def clear_ram(self) -> None:

        self.ram_cache.clear()

    def clear_disk(self) -> None:

        """
        Explicitly clear all feature cache files.

        This is intentionally NOT called automatically.
        """

        if not self.cache_dir.exists():
            return

        for path in self.cache_dir.iterdir():

            if path.is_file():
                path.unlink()

    def clear_all(self) -> None:

        self.clear_ram()
        self.clear_disk()

    # -----------------------------------------------------------------
    # Cache warmup helper
    # -----------------------------------------------------------------

    def is_cached(
        self,
        record,
        model_fingerprint: str,
    ) -> bool:

        key = self.build_key(
            record,
            model_fingerprint,
        )

        if key in self.ram_cache:
            return True

        if self.cache_path(key).exists():
            return True

        return False


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("V2 FEATURE CACHE")
    print("=" * 72)

    print("\n[1/5] Initializing Data Manager...")

    try:
        manager = DataManager()

    except Exception as exc:

        print(
            f"Data Manager initialization failed:\n{exc}"
        )

        raise

    train_records = manager.train

    print(
        f"TRAIN records available: "
        f"{len(train_records):,}"
    )

    print("\n[2/5] Initializing cache...")

    config = FeatureCacheConfig(
        cache_dir="ASR/v2/cache/features",
        max_ram_items=4,
        disk_dtype="float16",
        enable_ram_cache=True,
        enable_disk_cache=True,
    )

    cache = FrozenEncoderFeatureCache(
        config=config
    )

    print(
        f"Cache directory:\n"
        f"  {cache.cache_dir}"
    )

    print(
        f"RAM capacity: "
        f"{config.max_ram_items} items"
    )

    print(
        f"Disk dtype: "
        f"{config.disk_dtype}"
    )

    print("\n[3/5] Testing fingerprint...")

    fingerprint = cache.build_fingerprint(
        model_name="facebook/w2v-bert-2.0",
        sampling_rate=16000,
        feature_extractor_config={
            "sampling_rate": 16000,
        },
    )

    print(
        f"Model fingerprint:\n"
        f"  {fingerprint}"
    )

    print("\n[4/5] Testing cache key generation...")

    sample = train_records[0]

    key = cache.build_key(
        sample,
        fingerprint,
    )

    print(
        f"Record index: "
        f"{sample.index}"
    )

    print(
        f"Audio duration: "
        f"{sample.duration_sec:.3f}s"
    )

    print(
        f"Estimated projected tokens: "
        f"{sample.estimated_projected_tokens}"
    )

    print(
        f"Cache key:\n"
        f"  {key}"
    )

    print(
        f"Cache file:\n"
        f"  {cache.cache_path(key)}"
    )

    print("\n[5/5] Testing synthetic feature storage...")

    # Synthetic encoder output.
    #
    # Shape:
    #   batch = 1
    #   time  = 100
    #   hidden = 2048
    #
    # This tests the cache without loading the actual
    # Wav2Vec2-BERT model.
    synthetic_features = torch.randn(
        1,
        100,
        2048,
        dtype=torch.float32,
    )

    cache._disk_put(
        key,
        synthetic_features,
        metadata={
            "test": True,
            "record_index":
                int(sample.index),
        },
    )

    cache._ram_put(
        key,
        synthetic_features,
    )

    loaded_ram = cache._ram_get(
        key
    )

    if loaded_ram is None:
        raise RuntimeError(
            "RAM cache test failed."
        )

    print(
        "RAM cache: PASS"
    )

    cache.clear_ram()

    loaded_disk = cache._disk_get(
        key
    )

    if loaded_disk is None:
        raise RuntimeError(
            "Disk cache test failed."
        )

    print(
        "Disk cache: PASS"
    )

    if tuple(
        loaded_disk.shape
    ) != (
        1,
        100,
        2048,
    ):
        raise RuntimeError(
            "Cached tensor shape changed."
        )

    print(
        "Tensor shape preservation: PASS"
    )

    cache.print_statistics()

    print(
        f"\nDisk cache size: "
        f"{cache.disk_size_gb():.6f} GB"
    )

    print(
        "\nCleaning synthetic test entry..."
    )

    cache.clear_all()

    print(
        "Synthetic cache cleanup: PASS"
    )

    print("\n" + "=" * 72)
    print("FEATURE CACHE TEST: PASS")
    print("=" * 72)

    print(
        "\nImportant:"
    )

    print(
        "No dataset CSV was modified."
    )

    print(
        "No real encoder was loaded during this test."
    )

    print(
        "No training was performed."
    )

    print(
        "Only synthetic cache data was written."
    )


if __name__ == "__main__":
    main()