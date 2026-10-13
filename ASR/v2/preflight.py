from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path


# ----------------------------------------------------------------------
# Make project-root imports reliable when launched directly.
# ----------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch


try:
    from ASR.v2.hardware_manager import HardwareManager
    from ASR.v2.memory_manager import MemoryManager
except ImportError:
    from manipuri_stt.ASR.v2.hardware_manager import HardwareManager
    from manipuri_stt.ASR.v2.memory_manager import MemoryManager


class PreflightProbe:
    """
    Measured hardware-fit benchmark.

    This is intentionally Soup-inspired rather than Soup-dependent.

    The goal is to measure the actual memory footprint of our intended
    QLoRA configuration before selecting the training strategy.

    Possible outcomes:

        RESIDENT_FIT
        RESIDENT_TIGHT
        STREAMING_OR_OFFLOAD_REQUIRED
        FAILED_NO_CUDA
        FAILED_MODEL_LOAD
        FAILED_RUNTIME
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        min_free_reserve_mb: float = 512.0,
        safety_fraction: float = 0.15,
    ):

        self.model_name = model_name

        self.hw = HardwareManager()

        self.memory = MemoryManager(
            safety_fraction=safety_fraction,
            min_free_mb=min_free_reserve_mb,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _cleanup():
        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    @staticmethod
    def _mb(value: int | float) -> float:
        return float(value) / (1024**2)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def _snapshot(self):
        return self.memory.snapshot()

    def _print_memory(self, label: str):
        snapshot = self._snapshot()

        print(
            f"{label:<32} "
            f"allocated={snapshot.allocated_mb:8.1f} MB | "
            f"reserved={snapshot.reserved_mb:8.1f} MB | "
            f"peak={snapshot.peak_allocated_mb:8.1f} MB | "
            f"free={snapshot.free_mb:8.1f} MB | "
            f"budget={snapshot.training_budget_mb:8.1f} MB"
        )

        return snapshot

    # ------------------------------------------------------------------
    # Idle probe
    # ------------------------------------------------------------------

    def probe_idle(self):

        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()

        snapshot = self._print_memory(
            "Idle CUDA state"
        )

        return {
            "free_mb": snapshot.free_mb,
            "budget_mb": snapshot.training_budget_mb,
            "allocated_mb": snapshot.allocated_mb,
        }

    # ------------------------------------------------------------------
    # Tiny allocation probe
    # ------------------------------------------------------------------

    def probe_allocator(self):

        torch.cuda.reset_peak_memory_stats()

        tensor = torch.empty(
            1024,
            1024,
            device="cuda",
            dtype=torch.float16,
        )

        torch.cuda.synchronize()

        snapshot = self._print_memory(
            "Allocator probe"
        )

        del tensor

        self._cleanup()

        return {
            "peak_allocated_mb":
                snapshot.peak_allocated_mb,
            "peak_reserved_mb":
                snapshot.peak_reserved_mb,
        }

    # ------------------------------------------------------------------
    # Qwen loading
    # ------------------------------------------------------------------

    def probe_qwen(self):

        print()
        print("[3/5] Loading Qwen 2.5-3B NF4...")
        print(
            "      This is the REAL model-memory probe."
        )

        try:

            from transformers import (
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
                device_map={
                    "": "cuda:0"
                },
                low_cpu_mem_usage=True,
                trust_remote_code=False,
            )

            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            )

            snapshot = self._print_memory(
                "Qwen NF4 loaded"
            )

            return {
                "model": model,
                "load_time_sec": elapsed,
                "allocated_mb":
                    snapshot.allocated_mb,
                "reserved_mb":
                    snapshot.reserved_mb,
                "peak_allocated_mb":
                    snapshot.peak_allocated_mb,
                "peak_reserved_mb":
                    snapshot.peak_reserved_mb,
                "free_mb":
                    snapshot.free_mb,
                "budget_mb":
                    snapshot.training_budget_mb,
            }

        except Exception as exc:

            print()
            print(
                "[ERROR] Qwen model loading failed:"
            )
            print(exc)

            return None

    # ------------------------------------------------------------------
    # QLoRA probe
    # ------------------------------------------------------------------

    def probe_lora(self, model):

        print()
        print("[4/5] Attaching QLoRA adapters...")

        try:

            from peft import (
                LoraConfig,
                get_peft_model,
            )

            config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,

                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],

                bias="none",
                task_type="CAUSAL_LM",
            )

            torch.cuda.reset_peak_memory_stats()

            model = get_peft_model(
                model,
                config,
            )

            torch.cuda.synchronize()

            snapshot = self._print_memory(
                "QLoRA attached"
            )

            trainable = 0
            total = 0

            for parameter in model.parameters():

                count = parameter.numel()

                total += count

                if parameter.requires_grad:
                    trainable += count

            trainable_percent = (
                100.0 * trainable / total
                if total
                else 0.0
            )

            print(
                f"Trainable parameters: "
                f"{trainable:,}"
            )

            print(
                f"Total parameters:     "
                f"{total:,}"
            )

            print(
                f"Trainable percentage:  "
                f"{trainable_percent:.4f}%"
            )

            return {
                "model": model,
                "allocated_mb":
                    snapshot.allocated_mb,
                "reserved_mb":
                    snapshot.reserved_mb,
                "peak_allocated_mb":
                    snapshot.peak_allocated_mb,
                "peak_reserved_mb":
                    snapshot.peak_reserved_mb,
                "free_mb":
                    snapshot.free_mb,
                "budget_mb":
                    snapshot.training_budget_mb,
                "trainable_parameters":
                    trainable,
                "total_parameters":
                    total,
                "trainable_percent":
                    trainable_percent,
            }

        except Exception as exc:

            print()
            print(
                "[ERROR] QLoRA attachment failed:"
            )
            print(exc)

            return None

    # ------------------------------------------------------------------
    # Forward/backward probe
    # ------------------------------------------------------------------

    def probe_training_step(
        self,
        model,
        sequence_length: int = 128,
        batch_size: int = 1,
    ):

        print()
        print(
            "[5/5] Running synthetic forward/backward probe..."
        )

        try:

            model.train()

            if hasattr(
                model,
                "gradient_checkpointing_enable",
            ):

                try:
                    model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={
                            "use_reentrant": False
                        }
                    )

                    print(
                        "Gradient checkpointing: ENABLED"
                    )

                except Exception as exc:

                    print(
                        "[WARN] Gradient checkpointing "
                        f"could not be enabled: {exc}"
                    )

            device = torch.device("cuda:0")

            input_ids = torch.randint(
                low=0,
                high=model.config.vocab_size,
                size=(
                    batch_size,
                    sequence_length,
                ),
                device=device,
                dtype=torch.long,
            )

            attention_mask = torch.ones_like(
                input_ids
            )

            labels = input_ids.clone()

            torch.cuda.empty_cache()

            torch.cuda.reset_peak_memory_stats()

            start = time.perf_counter()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            loss.backward()

            torch.cuda.synchronize()

            elapsed = (
                time.perf_counter() - start
            )

            snapshot = self._print_memory(
                "Forward + backward"
            )

            print(
                f"Probe loss:            "
                f"{float(loss.detach().cpu()):.6f}"
            )

            print(
                f"Probe time:            "
                f"{elapsed:.2f} sec"
            )

            result = {
                "sequence_length":
                    sequence_length,

                "batch_size":
                    batch_size,

                "loss":
                    float(loss.detach().cpu()),

                "elapsed_sec":
                    elapsed,

                "peak_allocated_mb":
                    snapshot.peak_allocated_mb,

                "peak_reserved_mb":
                    snapshot.peak_reserved_mb,

                "free_mb":
                    snapshot.free_mb,

                "budget_mb":
                    snapshot.training_budget_mb,

                "headroom_mb":
                    snapshot.headroom_mb,

                "over_budget":
                    snapshot.over_budget,
            }

            del outputs
            del loss
            del input_ids
            del attention_mask
            del labels

            try:
                model.zero_grad(
                    set_to_none=True
                )
            except Exception:
                pass

            self._cleanup()

            return result

        except torch.cuda.OutOfMemoryError as exc:

            self.memory.record_oom()
            self.memory.recover_from_oom()

            print()
            print(
                "[OOM] Synthetic training probe "
                "exceeded available VRAM."
            )
            print(exc)

            return {
                "oom": True,
                "over_budget": True,
                "peak_allocated_mb":
                    self._snapshot().peak_allocated_mb,
            }

        except Exception as exc:

            print()
            print(
                "[ERROR] Training probe failed:"
            )
            print(exc)

            return None

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    @staticmethod
    def decide(
        probe_result: dict | None,
        safety_headroom_mb: float = 256.0,
    ) -> str:

        if not probe_result:
            return "FAILED_RUNTIME"

        if probe_result.get("oom", False):
            return "STREAMING_OR_OFFLOAD_REQUIRED"

        if probe_result.get(
            "over_budget",
            False,
        ):
            return "STREAMING_OR_OFFLOAD_REQUIRED"

        headroom = probe_result.get(
            "headroom_mb",
            0.0,
        )

        if headroom < safety_headroom_mb:
            return "RESIDENT_TIGHT"

        return "RESIDENT_FIT"

    # ------------------------------------------------------------------
    # Complete benchmark
    # ------------------------------------------------------------------

    def run(
        self,
        sequence_length: int = 128,
        batch_size: int = 1,
        output_path: str | Path | None = None,
    ) -> str:

        print()
        print("=" * 78)
        print(
            "V2 MEASURED SOUP-INSPIRED PREFLIGHT"
        )
        print("=" * 78)

        profile = self.hw.inspect()

        if not profile.cuda_available:

            print(
                "[CRITICAL] No CUDA training GPU detected."
            )

            return "FAILED_NO_CUDA"

        print()
        print(
            f"GPU:                  "
            f"{profile.primary_gpu_name}"
        )

        print(
            f"Physical VRAM:        "
            f"{profile.primary_gpu_vram_gb:.2f} GB"
        )

        print(
            f"Initial free VRAM:    "
            f"{profile.cuda_free_vram_gb:.2f} GB"
        )

        results = {
            "timestamp": time.time(),
            "model_name": self.model_name,
            "gpu": profile.primary_gpu_name,
            "physical_vram_gb":
                profile.primary_gpu_vram_gb,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
        }

        print()
        print("[1/5] Measuring idle state...")

        results["idle"] = self.probe_idle()

        print()
        print("[2/5] Testing CUDA allocator...")

        results["allocator"] = (
            self.probe_allocator()
        )

        qwen_result = self.probe_qwen()

        if qwen_result is None:

            self._cleanup()

            return "FAILED_MODEL_LOAD"

        model = qwen_result.pop(
            "model"
        )

        results["qwen"] = qwen_result

        lora_result = self.probe_lora(
            model
        )

        if lora_result is None:

            del model
            self._cleanup()

            return "FAILED_RUNTIME"

        model = lora_result.pop(
            "model"
        )

        results["lora"] = lora_result

        training_result = (
            self.probe_training_step(
                model,
                sequence_length=sequence_length,
                batch_size=batch_size,
            )
        )

        results["training"] = training_result

        verdict = self.decide(
            training_result
        )

        results["verdict"] = verdict

        # Cleanup model
        del model

        self._cleanup()

        print()
        print("=" * 78)
        print("PREFLIGHT DECISION")
        print("=" * 78)

        if verdict == "RESIDENT_FIT":

            print(
                "✅ RESIDENT_FIT"
            )

            print(
                "Resident QLoRA is viable under "
                "the tested configuration."
            )

        elif verdict == "RESIDENT_TIGHT":

            print(
                "⚠️ RESIDENT_TIGHT"
            )

            print(
                "Resident QLoRA fits, but memory "
                "headroom is small."
            )

            print(
                "Use conservative sequence lengths "
                "and strong memory controls."
            )

        elif verdict == (
            "STREAMING_OR_OFFLOAD_REQUIRED"
        ):

            print(
                "⚠️ STREAMING_OR_OFFLOAD_REQUIRED"
            )

            print(
                "The measured resident configuration "
                "does not have enough safe VRAM."
            )

            print(
                "V2 must activate a lower-memory "
                "strategy before full training."
            )

        else:

            print(
                f"❌ {verdict}"
            )

        print("=" * 78)

        # --------------------------------------------------------------
        # Save benchmark JSON
        # --------------------------------------------------------------

        if output_path is None:

            output_path = (
                PROJECT_ROOT
                / "ASR"
                / "v2"
                / "logs"
                / "preflight.json"
            )

        output_path = Path(
            output_path
        ).resolve()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as f:

                json.dump(
                    results,
                    f,
                    indent=2,
                )

            print()
            print(
                f"Benchmark saved to:\n"
                f"{output_path}"
            )

        except Exception as exc:

            print(
                f"[WARN] Could not save "
                f"preflight JSON: {exc}"
            )

        return verdict


if __name__ == "__main__":

    probe = PreflightProbe()

    probe.run(
        sequence_length=128,
        batch_size=1,
    )
