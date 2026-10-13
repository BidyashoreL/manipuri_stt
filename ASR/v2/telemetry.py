from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StepTelemetry:
    epoch: int
    step: int
    sample_id: str

    audio_seconds: float
    estimated_frames: int
    text_tokens: int

    loss: float
    step_time_sec: float

    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    free_mb: float

    training_budget_mb: float
    headroom_mb: float

    oom_events: int

    status: str

    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class TelemetryLogger:
    """
    Crash-resilient CSV telemetry logger.

    One CSV row = one training step.

    The log is deliberately plain CSV so it can be inspected with:
        pandas
        Excel
        PowerShell
        Python
        plotting tools
    """

    HEADERS = [
        "epoch",
        "step",
        "sample_id",

        "audio_seconds",
        "estimated_frames",
        "text_tokens",

        "loss",
        "step_time_sec",

        "allocated_mb",
        "reserved_mb",
        "peak_allocated_mb",
        "peak_reserved_mb",
        "free_mb",

        "training_budget_mb",
        "headroom_mb",

        "oom_events",

        "status",

        "timestamp",
    ]

    def __init__(
        self,
        log_dir: str | Path = "manipuri_stt/ASR/v2/logs",
        filename: str = "telemetry.csv",
    ):
        self.log_dir = Path(log_dir).resolve()
        self.log_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.telemetry_csv_path = (
            self.log_dir / filename
        )

        self._initialize_csv()

    # ------------------------------------------------------------------
    # CSV initialization
    # ------------------------------------------------------------------

    def _initialize_csv(self) -> None:
        if self.telemetry_csv_path.exists():
            return

        with open(
            self.telemetry_csv_path,
            mode="w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.writer(f)
            writer.writerow(self.HEADERS)

            f.flush()
            os.fsync(f.fileno())

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_step(
        self,
        telemetry: StepTelemetry,
    ) -> None:

        data = asdict(telemetry)

        row = [
            data.get(header, "")
            for header in self.HEADERS
        ]

        try:
            with open(
                self.telemetry_csv_path,
                mode="a",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.writer(f)
                writer.writerow(row)

                f.flush()
                os.fsync(f.fileno())

        except Exception as exc:
            print(
                f"[TELEMETRY ERROR] "
                f"Could not write telemetry: {exc}"
            )

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------

    @staticmethod
    def print_step_summary(
        telemetry: StepTelemetry,
    ) -> None:

        print(
            f"[Epoch {telemetry.epoch:02d} | "
            f"Step {telemetry.step:05d}] "
            f"Sample={telemetry.sample_id} | "
            f"Dur={telemetry.audio_seconds:.2f}s | "
            f"Tokens={telemetry.text_tokens} | "
            f"Loss={telemetry.loss:.4f} | "
            f"Time={telemetry.step_time_sec:.2f}s | "
            f"Peak={telemetry.peak_allocated_mb:.1f}MB | "
            f"Free={telemetry.free_mb:.1f}MB | "
            f"Headroom={telemetry.headroom_mb:.1f}MB | "
            f"OOM={telemetry.oom_events} | "
            f"{telemetry.status}"
        )

    # ------------------------------------------------------------------
    # Epoch summary
    # ------------------------------------------------------------------

    def log_epoch_summary(
        self,
        epoch: int,
        loss: float,
        cer: Optional[float],
        wer: Optional[float],
        elapsed_sec: float,
    ) -> None:

        path = self.log_dir / "epoch_summary.csv"

        headers = [
            "epoch",
            "loss",
            "cer",
            "wer",
            "elapsed_sec",
            "timestamp",
        ]

        exists = path.exists()

        try:
            with open(
                path,
                mode="a",
                newline="",
                encoding="utf-8",
            ) as f:

                writer = csv.writer(f)

                if not exists:
                    writer.writerow(headers)

                writer.writerow(
                    [
                        epoch,
                        loss,
                        "" if cer is None else cer,
                        "" if wer is None else wer,
                        elapsed_sec,
                        time.time(),
                    ]
                )

                f.flush()
                os.fsync(f.fileno())

        except Exception as exc:
            print(
                f"[TELEMETRY ERROR] "
                f"Could not write epoch summary: {exc}"
            )


if __name__ == "__main__":

    logger = TelemetryLogger()

    sample = StepTelemetry(
        epoch=1,
        step=1,
        sample_id="test_sample_001",

        audio_seconds=4.5,
        estimated_frames=225,
        text_tokens=18,

        loss=2.3456,
        step_time_sec=0.45,

        allocated_mb=1200.0,
        reserved_mb=1500.0,
        peak_allocated_mb=1800.0,
        peak_reserved_mb=2000.0,
        free_mb=2095.5,

        training_budget_mb=1700.0,
        headroom_mb=-100.0,

        oom_events=0,

        status="SUCCESS",
    )

    logger.log_step(sample)
    logger.print_step_summary(sample)

    logger.log_epoch_summary(
        epoch=1,
        loss=2.3456,
        cer=0.42,
        wer=0.88,
        elapsed_sec=120.0,
    )

    print()
    print(
        f"Telemetry CSV: "
        f"{logger.telemetry_csv_path}"
    )
