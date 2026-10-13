"""
V2 Duration-Aware Training Scheduler
====================================

Purpose
-------
Memory/resource-aware ordering for the Manipuri ASR training pipeline.

Important:
- Does NOT modify train.csv / val.csv / test.csv.
- Does NOT remove long-audio samples.
- Does NOT claim actual VRAM usage.
- Uses measured audio duration + estimated projected audio tokens.
- Keeps every sample exactly once per epoch.
- Deterministic given the same seed and epoch.
- Designed for batch_size=1 / gradient accumulation workloads.

Scheduling philosophy
---------------------
Shorter samples are generally safer for early computation.
Longer samples are distributed through the epoch instead of being allowed
to cluster unnecessarily.

The scheduler is an ORDERING layer, not a filtering layer.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    from .data_manager import DataManager, AudioRecord
except ImportError:
    from data_manager import DataManager, AudioRecord


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SchedulerConfig:
    seed: int = 42

    # Samples above this duration are treated as higher-risk.
    high_duration_sec: float = 8.0

    # Samples above this duration are treated as extreme.
    extreme_duration_sec: float = 15.0

    # Prevent unnecessary consecutive high-risk samples.
    avoid_consecutive_high_risk: bool = True

    # Curriculum is only used when explicitly requested.
    # First fraction of an epoch emphasizes safer samples.
    curriculum_fraction: float = 0.20

    # Number of records included in diagnostic preview.
    preview_size: int = 50


# ---------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------

class DurationAwareScheduler:
    """
    Duration-aware deterministic scheduler.

    Supported modes:

    1. "bucketed"
       - Main production mode.
       - Randomizes within duration buckets.
       - Uses weighted interleaving.
       - Attempts to avoid consecutive high-risk samples.

    2. "shuffle"
       - Pure deterministic shuffle.
       - Useful as a baseline.

    3. "curriculum"
       - Places safer samples earlier in the epoch.
       - Still includes ALL samples.
       - Useful during early experiments.

    Test data is never scheduled for training.
    """

    BUCKET_ORDER = [
        "0-3s",
        "3-5s",
        "5-8s",
        "8-12s",
        "12-15s",
        ">15s",
    ]

    HIGH_RISK_BUCKETS = {
        "8-12s",
        "12-15s",
        ">15s",
    }

    EXTREME_BUCKETS = {
        ">15s",
    }

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig()

    # -----------------------------------------------------------------
    # Bucket classification
    # -----------------------------------------------------------------

    def duration_bucket(self, duration_sec: float) -> str:
        d = float(duration_sec)

        if d < 3.0:
            return "0-3s"
        elif d < 5.0:
            return "3-5s"
        elif d < 8.0:
            return "5-8s"
        elif d < 12.0:
            return "8-12s"
        elif d <= 15.0:
            return "12-15s"
        else:
            return ">15s"

    def risk_level(self, record: AudioRecord) -> str:
        d = float(record.duration_sec)

        if d < 3.0:
            return "LOW"
        elif d < 5.0:
            return "NORMAL"
        elif d < 8.0:
            return "MEDIUM"
        elif d < 12.0:
            return "HIGH"
        elif d <= 15.0:
            return "VERY_HIGH"
        else:
            return "EXTREME"

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_records(records: Sequence[AudioRecord]) -> None:
        if not records:
            raise ValueError("Scheduler received zero records.")

        indices = [r.index for r in records]

        if len(indices) != len(set(indices)):
            raise ValueError(
                "Duplicate record indices detected before scheduling."
            )

        for r in records:
            if r.duration_sec < 0:
                raise ValueError(
                    f"Negative duration for record {r.index}: "
                    f"{r.duration_sec}"
                )

    # -----------------------------------------------------------------
    # Bucket grouping
    # -----------------------------------------------------------------

    def _group_by_bucket(
        self,
        records: Sequence[AudioRecord],
    ) -> Dict[str, List[AudioRecord]]:

        groups = defaultdict(list)

        for record in records:
            bucket = self.duration_bucket(record.duration_sec)
            groups[bucket].append(record)

        return {
            bucket: groups.get(bucket, [])
            for bucket in self.BUCKET_ORDER
        }

    # -----------------------------------------------------------------
    # Pure shuffle
    # -----------------------------------------------------------------

    def _shuffle(
        self,
        records: Sequence[AudioRecord],
        rng: random.Random,
    ) -> List[AudioRecord]:

        result = list(records)
        rng.shuffle(result)
        return result

    # -----------------------------------------------------------------
    # Weighted bucket selection
    # -----------------------------------------------------------------

    def _weighted_bucket_choice(
        self,
        available: Dict[str, List[AudioRecord]],
        rng: random.Random,
        previous_bucket: str | None,
    ) -> str:

        buckets = [
            b for b in self.BUCKET_ORDER
            if available.get(b)
        ]

        if not buckets:
            raise RuntimeError("No buckets available for selection.")

        # Prefer avoiding consecutive high-risk samples.
        safe_buckets = buckets

        if (
            self.config.avoid_consecutive_high_risk
            and previous_bucket in self.HIGH_RISK_BUCKETS
        ):
            non_high = [
                b for b in buckets
                if b not in self.HIGH_RISK_BUCKETS
            ]

            if non_high:
                safe_buckets = non_high

        # Weight by remaining population.
        total = sum(
            len(available[b])
            for b in safe_buckets
        )

        if total <= 0:
            return buckets[0]

        target = rng.uniform(0, total)
        cumulative = 0

        for bucket in safe_buckets:
            cumulative += len(available[bucket])

            if target <= cumulative:
                return bucket

        return safe_buckets[-1]

    # -----------------------------------------------------------------
    # Main bucketed scheduler
    # -----------------------------------------------------------------

    def _bucketed_schedule(
        self,
        records: Sequence[AudioRecord],
        rng: random.Random,
    ) -> List[AudioRecord]:

        groups = self._group_by_bucket(records)

        # Shuffle inside each duration group.
        for bucket in self.BUCKET_ORDER:
            rng.shuffle(groups[bucket])

        result: List[AudioRecord] = []
        previous_bucket = None

        while any(groups[b] for b in self.BUCKET_ORDER):

            bucket = self._weighted_bucket_choice(
                groups,
                rng,
                previous_bucket,
            )

            record = groups[bucket].pop()

            result.append(record)
            previous_bucket = bucket

        return result

    # -----------------------------------------------------------------
    # Curriculum scheduler
    # -----------------------------------------------------------------

    def _curriculum_schedule(
        self,
        records: Sequence[AudioRecord],
        rng: random.Random,
    ) -> List[AudioRecord]:

        groups = self._group_by_bucket(records)

        for bucket in self.BUCKET_ORDER:
            rng.shuffle(groups[bucket])

        safer_buckets = [
            "0-3s",
            "3-5s",
            "5-8s",
        ]

        high_buckets = [
            "8-12s",
            "12-15s",
            ">15s",
        ]

        safe_records = []

        for bucket in safer_buckets:
            safe_records.extend(groups[bucket])

        high_records = []

        for bucket in high_buckets:
            high_records.extend(groups[bucket])

        rng.shuffle(safe_records)
        rng.shuffle(high_records)

        # Put a fraction of safer examples first.
        target_safe_prefix = int(
            len(records) * self.config.curriculum_fraction
        )

        prefix = safe_records[:target_safe_prefix]

        remaining = (
            safe_records[target_safe_prefix:]
            + high_records
        )

        # The remaining samples use normal bucket-aware scheduling.
        remaining = self._bucketed_schedule(
            remaining,
            rng,
        )

        return prefix + remaining

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def schedule(
        self,
        records: Sequence[AudioRecord],
        epoch: int = 0,
        mode: str = "bucketed",
    ) -> List[AudioRecord]:

        self._validate_records(records)

        if mode not in {
            "bucketed",
            "shuffle",
            "curriculum",
        }:
            raise ValueError(
                f"Unknown scheduler mode: {mode}. "
                f"Use bucketed, shuffle, or curriculum."
            )

        # Epoch-dependent deterministic randomness.
        rng = random.Random(
            self.config.seed + int(epoch)
        )

        if mode == "shuffle":
            result = self._shuffle(records, rng)

        elif mode == "curriculum":
            result = self._curriculum_schedule(
                records,
                rng,
            )

        else:
            result = self._bucketed_schedule(
                records,
                rng,
            )

        self._validate_schedule_integrity(
            records,
            result,
        )

        return result

    # -----------------------------------------------------------------
    # Test protection
    # -----------------------------------------------------------------

    def evaluation_order(
        self,
        records: Sequence[AudioRecord],
    ) -> List[AudioRecord]:

        """
        Return evaluation records without changing their order.

        This exists intentionally so test/validation evaluation cannot
        accidentally inherit training scheduling logic.
        """

        self._validate_records(records)
        return list(records)

    # -----------------------------------------------------------------
    # Integrity
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_schedule_integrity(
        original: Sequence[AudioRecord],
        scheduled: Sequence[AudioRecord],
    ) -> None:

        original_ids = [r.index for r in original]
        scheduled_ids = [r.index for r in scheduled]

        if len(original_ids) != len(scheduled_ids):
            raise RuntimeError(
                "Scheduler changed the number of samples."
            )

        if len(scheduled_ids) != len(set(scheduled_ids)):
            raise RuntimeError(
                "Scheduler introduced duplicate samples."
            )

        if set(original_ids) != set(scheduled_ids):
            missing = set(original_ids) - set(scheduled_ids)
            extra = set(scheduled_ids) - set(original_ids)

            raise RuntimeError(
                "Scheduler changed dataset membership. "
                f"Missing={sorted(missing)[:10]}, "
                f"Extra={sorted(extra)[:10]}"
            )

    # -----------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------

    def summarize(
        self,
        records: Sequence[AudioRecord],
        scheduled: Sequence[AudioRecord],
    ) -> dict:

        original_counts = Counter(
            self.duration_bucket(r.duration_sec)
            for r in records
        )

        scheduled_counts = Counter(
            self.duration_bucket(r.duration_sec)
            for r in scheduled
        )

        risk_counts = Counter(
            self.risk_level(r)
            for r in scheduled
        )

        durations = [
            float(r.duration_sec)
            for r in scheduled
        ]

        projected = [
            int(r.estimated_projected_tokens)
            for r in scheduled
        ]

        return {
            "total_records": len(records),

            "original_bucket_counts": dict(
                original_counts
            ),

            "scheduled_bucket_counts": dict(
                scheduled_counts
            ),

            "risk_counts": dict(
                risk_counts
            ),

            "duration_sec": {
                "min": min(durations),
                "max": max(durations),
                "mean": sum(durations) / len(durations),
            },

            "estimated_projected_audio_tokens": {
                "min": min(projected),
                "max": max(projected),
                "mean": sum(projected) / len(projected),
            },

            "first_records": [
                {
                    "index": r.index,
                    "duration_sec": round(
                        float(r.duration_sec), 3
                    ),
                    "bucket": self.duration_bucket(
                        r.duration_sec
                    ),
                    "risk": self.risk_level(r),
                    "estimated_projected_tokens":
                        int(r.estimated_projected_tokens),
                    "source": r.source,
                }
                for r in scheduled[
                    : self.config.preview_size
                ]
            ],
        }

    # -----------------------------------------------------------------
    # Save diagnostics
    # -----------------------------------------------------------------

    def save_diagnostic(
        self,
        records: Sequence[AudioRecord],
        scheduled: Sequence[AudioRecord],
        path: str | Path,
        epoch: int = 0,
        mode: str = "bucketed",
    ) -> None:

        diagnostic = {
            "scheduler": asdict(self.config),
            "epoch": epoch,
            "mode": mode,
            "summary": self.summarize(
                records,
                scheduled,
            ),
        }

        path = Path(path)
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                diagnostic,
                f,
                indent=2,
                ensure_ascii=False,
            )


# ---------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("V2 DURATION-AWARE SCHEDULER")
    print("=" * 72)

    print("\n[1/4] Loading Data Manager...")

    manager = DataManager()

    train_records = manager.train
    val_records = manager.val
    test_records = manager.test

    print(
        f"TRAIN: {len(train_records):,}"
    )
    print(
        f"VAL  : {len(val_records):,}"
    )
    print(
        f"TEST : {len(test_records):,}"
    )

    if len(test_records) != 601:
        raise RuntimeError(
            f"TEST SET CORRUPTION: expected 601, "
            f"got {len(test_records)}"
        )

    print("\n[2/4] Initializing scheduler...")

    config = SchedulerConfig(
        seed=42,
        avoid_consecutive_high_risk=True,
    )

    scheduler = DurationAwareScheduler(config)

    print(
        "Mode: bucketed"
    )

    print("\n[3/4] Building epoch schedule...")

    scheduled = scheduler.schedule(
        train_records,
        epoch=0,
        mode="bucketed",
    )

    scheduler._validate_schedule_integrity(
        train_records,
        scheduled,
    )

    print(
        f"Scheduled samples: {len(scheduled):,}"
    )

    print("\nFirst 20 scheduled records:")
    print("-" * 72)

    for position, record in enumerate(
        scheduled[:20],
        start=1,
    ):
        print(
            f"{position:03d} | "
            f"idx={record.index:5d} | "
            f"{record.duration_sec:6.2f}s | "
            f"{scheduler.duration_bucket(record.duration_sec):7s} | "
            f"{scheduler.risk_level(record):9s} | "
            f"proj={record.estimated_projected_tokens:4d}"
        )

    print("\n[4/4] Running integrity + diagnostic checks...")

    summary = scheduler.summarize(
        train_records,
        scheduled,
    )

    # Test set must remain untouched.
    evaluation_test = scheduler.evaluation_order(
        test_records
    )

    if [
        r.index for r in evaluation_test
    ] != [
        r.index for r in test_records
    ]:
        raise RuntimeError(
            "Test evaluation order was modified."
        )

    diagnostic_path = (
        Path(__file__).resolve().parent
        / "logs"
        / "scheduler_diagnostic.json"
    )

    scheduler.save_diagnostic(
        train_records,
        scheduled,
        diagnostic_path,
        epoch=0,
        mode="bucketed",
    )

    print("\nBucket distribution:")
    for bucket in scheduler.BUCKET_ORDER:
        count = summary[
            "scheduled_bucket_counts"
        ].get(bucket, 0)

        print(
            f"  {bucket:8s}: {count:6,d}"
        )

    print("\nRisk distribution:")
    for risk, count in sorted(
        summary["risk_counts"].items()
    ):
        print(
            f"  {risk:10s}: {count:6,d}"
        )

    print(
        "\nDiagnostic saved:"
    )
    print(
        diagnostic_path
    )

    print("\n" + "=" * 72)
    print("SCHEDULER TEST: PASS")
    print("=" * 72)

    print(
        "\nImportant:"
    )
    print(
        "The scheduler reordered samples only."
    )
    print(
        "No CSV files were modified."
    )
    print(
        "No samples were discarded."
    )
    print(
        "Test set remains protected."
    )


if __name__ == "__main__":
    main()