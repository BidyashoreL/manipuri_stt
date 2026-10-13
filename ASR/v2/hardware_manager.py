from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import psutil
import torch


@dataclass
class HardwareProfile:
    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    os_name: str
    os_version: str
    python_version: str

    # ------------------------------------------------------------------
    # CPU
    # ------------------------------------------------------------------

    cpu_model: str
    cpu_count_physical: int
    cpu_count_logical: int

    # ------------------------------------------------------------------
    # RAM
    # ------------------------------------------------------------------

    total_ram_gb: float
    available_ram_gb: float
    used_ram_gb: float
    ram_percent: float

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    workspace_path: str
    storage_total_gb: float
    storage_free_gb: float
    storage_used_gb: float
    storage_free_percent: float

    # ------------------------------------------------------------------
    # CUDA
    # ------------------------------------------------------------------

    cuda_available: bool
    cuda_device_count: int
    cuda_version: str

    # ------------------------------------------------------------------
    # Primary CUDA GPU
    # ------------------------------------------------------------------

    primary_gpu_name: str
    primary_gpu_index: int
    primary_gpu_vram_gb: float
    primary_gpu_compute_capability: str
    primary_gpu_architecture: str

    # ------------------------------------------------------------------
    # Runtime GPU memory
    # ------------------------------------------------------------------

    cuda_free_vram_gb: float
    cuda_used_vram_gb: float

    # ------------------------------------------------------------------
    # Windows display adapters
    # ------------------------------------------------------------------

    display_adapters: list[str]

    # ------------------------------------------------------------------
    # Training decision
    # ------------------------------------------------------------------

    has_cuda_training_gpu: bool
    recommended_device: str


class HardwareManager:
    """
    V2 hardware and resource discovery.

    Responsibilities:
        - Detect CPU
        - Detect system RAM
        - Detect workspace storage
        - Detect CUDA availability
        - Identify the primary CUDA GPU
        - Report physical GPU VRAM
        - Report runtime free/used VRAM
        - Detect Windows display adapters
        - Recommend the training device

    Important:
        PyTorch CUDA device numbering is independent of Windows
        display-adapter numbering.

        Example:

            Windows:
                GPU 0 = Intel UHD
                GPU 1 = NVIDIA RTX 2050

            PyTorch:
                cuda:0 = NVIDIA RTX 2050

        Therefore we must NOT assume that Windows GPU 1 means
        torch.cuda device 1.

    Important VRAM note:
        torch.cuda.mem_get_info() returns:

            free_bytes, total_bytes

        It does NOT return:

            free_bytes, used_bytes

        Therefore:

            used_bytes = total_bytes - free_bytes
    """

    def __init__(
        self,
        workspace_path: Optional[str | Path] = None,
        cuda_index: int = 0,
    ):
        # Project root:
        #
        # ASR/v2/hardware_manager.py
        #       ↑
        # parents[0] = v2
        # parents[1] = ASR
        # parents[2] = project root
        #
        self.workspace_path = Path(
            workspace_path or Path(__file__).resolve().parents[2]
        ).resolve()

        self.cuda_index = cuda_index

    # ==================================================================
    # CPU
    # ==================================================================

    @staticmethod
    def _cpu_model() -> str:
        """
        Detect CPU model.

        First tries platform.processor().
        On Windows, falls back to WMI/CIM if necessary.
        """

        model = platform.processor()

        if model:
            return model

        if os.name == "nt":
            try:
                result = subprocess.run(
                    [
                        "powershell",
                        "-NoProfile",
                        "-Command",
                        (
                            "(Get-CimInstance Win32_Processor | "
                            "Select-Object -First 1 -ExpandProperty Name)"
                        ),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                value = result.stdout.strip()

                if value:
                    return value

            except Exception:
                pass

        return "Unknown CPU"

    # ==================================================================
    # Windows display adapters
    # ==================================================================

    @staticmethod
    def _display_adapters() -> list[str]:
        """
        Detect Windows display adapters.

        This is informational only.

        Do NOT use the Windows adapter index as the CUDA index.
        """

        adapters: list[str] = []

        if os.name != "nt":
            return adapters

        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "Get-CimInstance Win32_VideoController | "
                        "Select-Object -ExpandProperty Name"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            for line in result.stdout.splitlines():
                line = line.strip()

                if line and line not in adapters:
                    adapters.append(line)

        except Exception:
            pass

        return adapters

    # ==================================================================
    # CUDA
    # ==================================================================

    def _cuda_info(self) -> dict[str, Any]:
        """
        Inspect the primary CUDA device.

        Returns a normalized dictionary so the rest of V2 does not
        need to directly interact with torch.cuda.
        """

        info: dict[str, Any] = {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_version": str(torch.version.cuda or "Unknown"),

            "primary_gpu_name": "None (CPU Only)",
            "primary_gpu_index": -1,
            "primary_gpu_vram_gb": 0.0,
            "primary_gpu_compute_capability": "N/A",
            "primary_gpu_architecture": "N/A",

            "cuda_free_vram_gb": 0.0,
            "cuda_used_vram_gb": 0.0,
        }

        # --------------------------------------------------------------
        # CUDA availability
        # --------------------------------------------------------------

        if not torch.cuda.is_available():
            return info

        count = torch.cuda.device_count()

        if count <= 0:
            return info

        # Keep CUDA index within valid range.
        index = min(max(self.cuda_index, 0), count - 1)

        try:
            # ----------------------------------------------------------
            # Select CUDA device
            # ----------------------------------------------------------

            torch.cuda.set_device(index)

            # ----------------------------------------------------------
            # GPU properties
            # ----------------------------------------------------------

            props = torch.cuda.get_device_properties(index)

            physical_total_bytes = props.total_memory

            # ----------------------------------------------------------
            # Runtime memory
            #
            # IMPORTANT:
            #
            # torch.cuda.mem_get_info(index)
            #
            # returns:
            #
            #     free_bytes
            #     total_bytes
            #
            # NOT:
            #
            #     free_bytes
            #     used_bytes
            # ----------------------------------------------------------

            free_bytes, runtime_total_bytes = torch.cuda.mem_get_info(index)

            # Runtime total should normally agree closely with
            # physical_total_bytes. We use runtime_total_bytes for
            # calculating runtime usage because it belongs to the same
            # memory query as free_bytes.
            used_bytes = runtime_total_bytes - free_bytes

            # ----------------------------------------------------------
            # Compute capability
            # ----------------------------------------------------------

            major, minor = torch.cuda.get_device_capability(index)

            # ----------------------------------------------------------
            # Broad architecture classification
            # ----------------------------------------------------------

            if major >= 9:
                architecture = "Hopper/Blackwell-class"

            elif major >= 8:
                architecture = "Ampere-or-newer"

            elif major >= 7:
                architecture = "Volta/Turing-class"

            else:
                architecture = "Older CUDA architecture"

            # ----------------------------------------------------------
            # Populate result
            # ----------------------------------------------------------

            info.update(
                {
                    "cuda_available": True,
                    "cuda_device_count": count,

                    "primary_gpu_name": props.name,
                    "primary_gpu_index": index,

                    "primary_gpu_vram_gb": round(
                        physical_total_bytes / (1024**3),
                        3,
                    ),

                    "primary_gpu_compute_capability": (
                        f"{major}.{minor}"
                    ),

                    "primary_gpu_architecture": architecture,

                    "cuda_free_vram_gb": round(
                        free_bytes / (1024**3),
                        3,
                    ),

                    "cuda_used_vram_gb": round(
                        used_bytes / (1024**3),
                        3,
                    ),
                }
            )

        except Exception as exc:
            print(f"[WARN] CUDA inspection failed: {exc}")

        return info

    # ==================================================================
    # Full hardware inspection
    # ==================================================================

    def inspect(self) -> HardwareProfile:
        """
        Perform a complete hardware inspection.
        """

        # --------------------------------------------------------------
        # CPU
        # --------------------------------------------------------------

        cpu_logical = os.cpu_count() or 1

        cpu_physical = (
            psutil.cpu_count(logical=False)
            or cpu_logical
        )

        cpu_model = self._cpu_model()

        # --------------------------------------------------------------
        # RAM
        # --------------------------------------------------------------

        ram = psutil.virtual_memory()

        total_ram_gb = ram.total / (1024**3)

        available_ram_gb = (
            ram.available / (1024**3)
        )

        used_ram_gb = (
            ram.used / (1024**3)
        )

        # --------------------------------------------------------------
        # Storage
        # --------------------------------------------------------------

        try:
            disk = shutil.disk_usage(
                self.workspace_path
            )

        except Exception:
            disk = shutil.disk_usage(
                Path.cwd()
            )

        storage_total_gb = (
            disk.total / (1024**3)
        )

        storage_free_gb = (
            disk.free / (1024**3)
        )

        storage_used_gb = (
            disk.used / (1024**3)
        )

        storage_free_percent = (
            (disk.free / disk.total) * 100
            if disk.total
            else 0.0
        )

        # --------------------------------------------------------------
        # CUDA
        # --------------------------------------------------------------

        cuda = self._cuda_info()

        has_cuda = bool(
            cuda["cuda_available"]
        )

        # --------------------------------------------------------------
        # Construct profile
        # --------------------------------------------------------------

        return HardwareProfile(
            # System
            os_name=platform.system(),
            os_version=platform.release(),
            python_version=platform.python_version(),

            # CPU
            cpu_model=cpu_model,
            cpu_count_physical=cpu_physical,
            cpu_count_logical=cpu_logical,

            # RAM
            total_ram_gb=round(
                total_ram_gb,
                2,
            ),

            available_ram_gb=round(
                available_ram_gb,
                2,
            ),

            used_ram_gb=round(
                used_ram_gb,
                2,
            ),

            ram_percent=round(
                ram.percent,
                1,
            ),

            # Storage
            workspace_path=str(
                self.workspace_path
            ),

            storage_total_gb=round(
                storage_total_gb,
                2,
            ),

            storage_free_gb=round(
                storage_free_gb,
                2,
            ),

            storage_used_gb=round(
                storage_used_gb,
                2,
            ),

            storage_free_percent=round(
                storage_free_percent,
                1,
            ),

            # CUDA
            cuda_available=has_cuda,

            cuda_device_count=cuda[
                "cuda_device_count"
            ],

            cuda_version=cuda[
                "cuda_version"
            ],

            # Primary GPU
            primary_gpu_name=cuda[
                "primary_gpu_name"
            ],

            primary_gpu_index=cuda[
                "primary_gpu_index"
            ],

            primary_gpu_vram_gb=cuda[
                "primary_gpu_vram_gb"
            ],

            primary_gpu_compute_capability=cuda[
                "primary_gpu_compute_capability"
            ],

            primary_gpu_architecture=cuda[
                "primary_gpu_architecture"
            ],

            # Runtime memory
            cuda_free_vram_gb=cuda[
                "cuda_free_vram_gb"
            ],

            cuda_used_vram_gb=cuda[
                "cuda_used_vram_gb"
            ],

            # Windows
            display_adapters=self._display_adapters(),

            # Training
            has_cuda_training_gpu=has_cuda,

            recommended_device=(
                "cuda"
                if has_cuda
                else "cpu"
            ),
        )

    # ==================================================================
    # Runtime snapshot
    # ==================================================================

    def runtime_snapshot(self) -> dict[str, float]:
        """
        Lightweight runtime resource snapshot.

        Used later by the V2 resource controller and telemetry system.

        Returns:
            ram_available_gb
            ram_used_percent
            cuda_free_gb
            cuda_used_gb
        """

        ram = psutil.virtual_memory()

        result = {
            "ram_available_gb": (
                ram.available / (1024**3)
            ),

            "ram_used_percent": (
                ram.percent
            ),
        }

        # --------------------------------------------------------------
        # CUDA runtime memory
        # --------------------------------------------------------------

        if torch.cuda.is_available():
            try:
                free_bytes, runtime_total_bytes = (
                    torch.cuda.mem_get_info(
                        self.cuda_index
                    )
                )

                # IMPORTANT:
                #
                # mem_get_info() returns free + total.
                #
                # Calculate used explicitly.
                used_bytes = (
                    runtime_total_bytes
                    - free_bytes
                )

                result.update(
                    {
                        "cuda_free_gb": (
                            free_bytes / (1024**3)
                        ),

                        "cuda_used_gb": (
                            used_bytes / (1024**3)
                        ),
                    }
                )

            except Exception:
                result.update(
                    {
                        "cuda_free_gb": 0.0,
                        "cuda_used_gb": 0.0,
                    }
                )

        else:
            result.update(
                {
                    "cuda_free_gb": 0.0,
                    "cuda_used_gb": 0.0,
                }
            )

        return result

    # ==================================================================
    # Printing
    # ==================================================================

    def print_summary(self) -> None:
        """
        Print a human-readable hardware summary.
        """

        profile = self.inspect()

        print()
        print("=" * 78)
        print("V2 HARDWARE / RESOURCE PROFILE")
        print("=" * 78)

        # --------------------------------------------------------------
        # System
        # --------------------------------------------------------------

        print(
            f"OS:                    "
            f"{profile.os_name} "
            f"{profile.os_version}"
        )

        print(
            f"Python:                "
            f"{profile.python_version}"
        )

        # --------------------------------------------------------------
        # CPU
        # --------------------------------------------------------------

        print()
        print("CPU")
        print("-" * 78)

        print(
            f"Model:                 "
            f"{profile.cpu_model}"
        )

        print(
            f"Cores:                 "
            f"{profile.cpu_count_physical} physical / "
            f"{profile.cpu_count_logical} logical"
        )

        # --------------------------------------------------------------
        # RAM
        # --------------------------------------------------------------

        print()
        print("SYSTEM RAM")
        print("-" * 78)

        print(
            f"RAM:                   "
            f"{profile.available_ram_gb:.2f} GB free / "
            f"{profile.total_ram_gb:.2f} GB total"
        )

        print(
            f"RAM Usage:             "
            f"{profile.ram_percent:.1f}%"
        )

        # --------------------------------------------------------------
        # Storage
        # --------------------------------------------------------------

        print()
        print("STORAGE")
        print("-" * 78)

        print(
            f"Workspace:             "
            f"{profile.workspace_path}"
        )

        print(
            f"Storage:               "
            f"{profile.storage_free_gb:.2f} GB free / "
            f"{profile.storage_total_gb:.2f} GB total"
        )

        print(
            f"Free Space:            "
            f"{profile.storage_free_percent:.1f}%"
        )

        # --------------------------------------------------------------
        # CUDA
        # --------------------------------------------------------------

        print()
        print("CUDA")
        print("-" * 78)

        print(
            f"CUDA Available:        "
            f"{profile.cuda_available}"
        )

        print(
            f"CUDA Devices:          "
            f"{profile.cuda_device_count}"
        )

        print(
            f"CUDA Runtime:          "
            f"{profile.cuda_version}"
        )

        if profile.cuda_available:

            print(
                f"CUDA Device:           "
                f"{profile.primary_gpu_index}"
            )

            print(
                f"GPU:                   "
                f"{profile.primary_gpu_name}"
            )

            print(
                f"VRAM:                  "
                f"{profile.primary_gpu_vram_gb:.2f} GB"
            )

            print(
                f"Free VRAM:             "
                f"{profile.cuda_free_vram_gb:.2f} GB"
            )

            print(
                f"Used VRAM:             "
                f"{profile.cuda_used_vram_gb:.2f} GB"
            )

            print(
                f"Compute Capability:    "
                f"{profile.primary_gpu_compute_capability}"
            )

            print(
                f"Architecture:          "
                f"{profile.primary_gpu_architecture}"
            )

        # --------------------------------------------------------------
        # Windows display adapters
        # --------------------------------------------------------------

        print()
        print("WINDOWS DISPLAY ADAPTERS")
        print("-" * 78)

        if profile.display_adapters:

            for i, adapter in enumerate(
                profile.display_adapters
            ):
                print(
                    f"[{i}] {adapter}"
                )

        else:

            print(
                "No Windows display adapter "
                "information available."
            )

        # --------------------------------------------------------------
        # Training decision
        # --------------------------------------------------------------

        print()
        print("TRAINING DECISION")
        print("-" * 78)

        print(
            f"CUDA Training GPU:     "
            f"{profile.has_cuda_training_gpu}"
        )

        print(
            f"Recommended Device:    "
            f"{profile.recommended_device}"
        )

        print("=" * 78)
        print()


# ======================================================================
# Standalone execution
# ======================================================================

if __name__ == "__main__":

    manager = HardwareManager()

    manager.print_summary()