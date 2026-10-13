from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class MemorySnapshot:
    timestamp: float

    total_mb: float
    free_mb: float

    allocated_mb: float
    reserved_mb: float

    peak_allocated_mb: float
    peak_reserved_mb: float

    training_budget_mb: float
    utilization_percent: float

    oom_events: int

    @property
    def headroom_mb(self) -> float:
        return max(
            0.0,
            self.training_budget_mb - self.peak_allocated_mb,
        )

    @property
    def over_budget(self) -> bool:
        return self.peak_allocated_mb > self.training_budget_mb


class MemoryManager:
    """
    Dynamic CUDA memory controller.

    Design principles:
    - Never assume the whole physical VRAM is available.
    - Preserve a configurable free-memory reserve.
    - Account for allocator overhead.
    - Measure actual peak memory.
    - Recover cleanly after CUDA OOM.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        safety_fraction: float = 0.15,
        min_free_mb: float = 512.0,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for MemoryManager."
            )

        self.device = torch.device(
            device or "cuda:0"
        )

        if self.device.type != "cuda":
            raise ValueError(
                "MemoryManager requires a CUDA device."
            )

        self.safety_fraction = float(
            max(0.0, min(0.50, safety_fraction))
        )

        self.min_free_mb = float(
            max(0.0, min_free_mb)
        )

        self.oom_events = 0

        torch.cuda.set_device(self.device)

        self._step_start_time = None

    # ------------------------------------------------------------------
    # Raw CUDA memory
    # ------------------------------------------------------------------

    def _raw_memory(self):
        free_bytes, total_bytes = torch.cuda.mem_get_info(
            self.device
        )

        allocated_bytes = torch.cuda.memory_allocated(
            self.device
        )

        reserved_bytes = torch.cuda.memory_reserved(
            self.device
        )

        peak_allocated_bytes = torch.cuda.max_memory_allocated(
            self.device
        )

        peak_reserved_bytes = torch.cuda.max_memory_reserved(
            self.device
        )

        return (
            free_bytes,
            total_bytes,
            allocated_bytes,
            reserved_bytes,
            peak_allocated_bytes,
            peak_reserved_bytes,
        )

    # ------------------------------------------------------------------
    # Dynamic budget
    # ------------------------------------------------------------------

    def training_budget_mb(self) -> float:
        """
        Calculate safe training budget from CURRENT CUDA-free memory.

        This is deliberately dynamic because Windows, the desktop,
        drivers, and other CUDA allocations can consume VRAM.
        """

        free_bytes, _, *_ = self._raw_memory()

        free_mb = free_bytes / (1024**2)

        safety_reserve = free_mb * self.safety_fraction

        budget = free_mb - safety_reserve - self.min_free_mb

        return max(0.0, budget)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> MemorySnapshot:
        (
            free_bytes,
            total_bytes,
            allocated_bytes,
            reserved_bytes,
            peak_allocated_bytes,
            peak_reserved_bytes,
        ) = self._raw_memory()

        total_mb = total_bytes / (1024**2)
        free_mb = free_bytes / (1024**2)

        allocated_mb = allocated_bytes / (1024**2)
        reserved_mb = reserved_bytes / (1024**2)

        peak_allocated_mb = peak_allocated_bytes / (1024**2)
        peak_reserved_mb = peak_reserved_bytes / (1024**2)

        utilization = (
            ((total_mb - free_mb) / total_mb) * 100
            if total_mb > 0
            else 0.0
        )

        return MemorySnapshot(
            timestamp=time.time(),

            total_mb=total_mb,
            free_mb=free_mb,

            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,

            peak_allocated_mb=peak_allocated_mb,
            peak_reserved_mb=peak_reserved_mb,

            training_budget_mb=self.training_budget_mb(),

            utilization_percent=utilization,

            oom_events=self.oom_events,
        )

    # ------------------------------------------------------------------
    # Step tracking
    # ------------------------------------------------------------------

    def begin_step(self) -> None:
        self._step_start_time = time.perf_counter()

        torch.cuda.reset_peak_memory_stats(
            self.device
        )

    def end_step(self) -> MemorySnapshot:
        return self.snapshot()

    # ------------------------------------------------------------------
    # OOM handling
    # ------------------------------------------------------------------

    def record_oom(self) -> None:
        self.oom_events += 1

    def recover_from_oom(self) -> None:
        """
        Best-effort CUDA OOM recovery.
        """

        self.record_oom()

        try:
            gc.collect()
        except Exception:
            pass

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @staticmethod
    def cleanup_cuda() -> None:
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

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------

    def can_allocate(self, estimated_mb: float) -> bool:
        return float(estimated_mb) <= self.training_budget_mb()

    def assert_within_budget(self, snapshot: MemorySnapshot) -> None:
        if snapshot.over_budget:
            raise MemoryError(
                f"CUDA peak allocation "
                f"{snapshot.peak_allocated_mb:.1f} MB exceeded "
                f"training budget "
                f"{snapshot.training_budget_mb:.1f} MB."
            )

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_snapshot(
        self,
        snapshot: Optional[MemorySnapshot] = None,
        prefix: str = "",
    ) -> None:

        snapshot = snapshot or self.snapshot()

        print(
            f"{prefix}"
            f"VRAM "
            f"allocated={snapshot.allocated_mb:.1f}MB | "
            f"reserved={snapshot.reserved_mb:.1f}MB | "
            f"peak={snapshot.peak_allocated_mb:.1f}MB | "
            f"free={snapshot.free_mb:.1f}MB | "
            f"budget={snapshot.training_budget_mb:.1f}MB | "
            f"headroom={snapshot.headroom_mb:.1f}MB | "
            f"util={snapshot.utilization_percent:.1f}% | "
            f"OOM={snapshot.oom_events}"
        )


if __name__ == "__main__":
    manager = MemoryManager()

    print("=" * 78)
    print("V2 MEMORY MANAGER TEST")
    print("=" * 78)

    snapshot = manager.snapshot()

    manager.print_snapshot(snapshot)

    print()
    print("Testing peak-memory tracking...")

    manager.begin_step()

    x = torch.randn(
        2048,
        2048,
        device="cuda",
        dtype=torch.float16,
    )

    y = x @ x

    del y
    del x

    snapshot = manager.end_step()

    manager.print_snapshot(snapshot)

    manager.cleanup_cuda()

    print()
    print("After cleanup:")

    manager.print_snapshot()

    print("=" * 78)
