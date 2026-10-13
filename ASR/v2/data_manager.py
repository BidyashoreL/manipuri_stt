from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

V2_DIR = Path(__file__).resolve().parent
ASR_DIR = V2_DIR.parent
PROJECT_DIR = ASR_DIR.parent

TRAIN_CSV = ASR_DIR / "train.csv"
VAL_CSV = ASR_DIR / "val.csv"
TEST_CSV = ASR_DIR / "test.csv"

DURATION_ANALYSIS_JSON = (
    V2_DIR / "logs" / "duration_analysis.json"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURE_RATE = 50.0

# AudioProjector:
#
# Conv1D(stride=2)
#
# Therefore:
#
# 50 encoder frames/sec / 2
# = 25 projected tokens/sec
#
PROJECTED_RATE = FEATURE_RATE / 2.0


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class AudioRecord:
    """
    Immutable representation of one dataset sample.

    The original CSV remains untouched.

    audio_path:
        Resolved path on the current machine.

    original_audio_path:
        Exact path stored in the CSV.

    duration_sec:
        Audio duration.

    estimated_projected_tokens:
        Approximate number of tokens entering Qwen after the
        frozen Wav2Vec2-BERT + AudioProjector pipeline.

    duration_bucket:
        Scheduler-friendly duration category.
    """

    index: int

    audio_path: str
    original_audio_path: str

    text: str
    source: str
    orig_script: str

    duration_sec: float
    estimated_projected_tokens: int

    duration_bucket: str


# ============================================================================
# DATA MANAGER
# ============================================================================

class DataManager:
    """
    V2 dataset manager.

    Responsibilities:

        1. Load train/validation/test CSVs.
        2. Resolve stale Windows paths.
        3. Load duration analysis metadata.
        4. Validate audio files.
        5. Attach duration and projected-sequence metadata.
        6. Preserve dataset split boundaries.
        7. Keep the 601-sample test set immutable.

    Important:

        This class NEVER modifies the CSV files.
    """

    # ------------------------------------------------------------------------
    # Expected split sizes
    # ------------------------------------------------------------------------

    EXPECTED_SPLIT_SIZES = {
        "train": 10199,
        "val": 1207,
        "test": 601,
    }

    # ------------------------------------------------------------------------
    # Duration buckets
    # ------------------------------------------------------------------------

    DURATION_BUCKETS = (
        ("0-3 sec", 0.0, 3.0),
        ("3-5 sec", 3.0, 5.0),
        ("5-8 sec", 5.0, 8.0),
        ("8-12 sec", 8.0, 12.0),
        ("12-15 sec", 12.0, 15.0),
        (">15 sec", 15.0, float("inf")),
    )

    # ------------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------------

    def __init__(
        self,
        duration_analysis_path: Optional[
            str | Path
        ] = None,
    ):
        self.duration_analysis_path = Path(
            duration_analysis_path
            or DURATION_ANALYSIS_JSON
        ).resolve()

        self._duration_metadata = (
            self._load_duration_metadata()
        )

        self._datasets: dict[
            str,
            list[AudioRecord],
        ] = {}

        self._loaded = False

    # ========================================================================
    # Duration metadata
    # ========================================================================

    def _load_duration_metadata(self) -> dict:
        """
        Load the duration analyzer output.

        This avoids repeatedly opening WAV headers.
        """

        if not self.duration_analysis_path.exists():
            raise FileNotFoundError(
                "Duration analysis file not found:\n"
                f"{self.duration_analysis_path}\n\n"
                "Run duration_analyzer.py first."
            )

        with open(
            self.duration_analysis_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return data

    # ========================================================================
    # Audio path resolver
    # ========================================================================

    @staticmethod
    def _is_valid_file(
        path: Path,
    ) -> bool:
        try:
            return (
                path.exists()
                and path.is_file()
            )
        except Exception:
            return False

    def resolve_audio_path(
        self,
        raw_path: str,
    ) -> Optional[Path]:
        """
        Resolve both current and legacy paths.

        Resolution order:

            1. Exact path
            2. ASR-relative path
            3. Project-relative path
            4. Legacy \\manipuri_stt\\ marker
            5. Known dataset directories
            6. Unique filename fallback

        CSV is never modified.
        """

        if raw_path is None:
            return None

        raw = (
            str(raw_path)
            .strip()
            .strip('"')
        )

        if not raw:
            return None

        normalized = raw.replace(
            "/",
            "\\",
        )

        # --------------------------------------------------------------------
        # 1. Exact path
        # --------------------------------------------------------------------

        direct = Path(raw)

        if self._is_valid_file(direct):
            return direct.resolve()

        # --------------------------------------------------------------------
        # 2. Relative to ASR
        # --------------------------------------------------------------------

        if not Path(raw).is_absolute():

            candidate = (
                ASR_DIR
                / Path(raw)
            )

            if self._is_valid_file(candidate):
                return candidate.resolve()

            # ----------------------------------------------------------------
            # 3. Relative to project
            # ----------------------------------------------------------------

            candidate = (
                PROJECT_DIR
                / Path(raw)
            )

            if self._is_valid_file(candidate):
                return candidate.resolve()

        # --------------------------------------------------------------------
        # 4. Legacy project marker
        #
        # Example:
        #
        # C:\Users\OldUser\...\manipuri_stt\benchmark\sample.wav
        #
        # becomes:
        #
        # <current project>\benchmark\sample.wav
        # --------------------------------------------------------------------

        lower_normalized = (
            normalized.lower()
        )

        marker = "\\manipuri_stt\\"

        marker_index = (
            lower_normalized.find(
                marker.lower()
            )
        )

        if marker_index >= 0:

            relative_part = normalized[
                marker_index
                + len(marker):
            ]

            relative_part = (
                relative_part
                .lstrip("\\/")
            )

            candidate = (
                PROJECT_DIR
                / Path(relative_part)
            )

            if self._is_valid_file(
                candidate
            ):
                return candidate.resolve()

        # --------------------------------------------------------------------
        # 5. Known directories
        # --------------------------------------------------------------------

        known_dirs = [
            "benchmark",
            "manipuri_benchmark_extracted",
            "manipuri_1k_project",
        ]

        for known_dir in known_dirs:

            marker = (
                "\\"
                + known_dir
                + "\\"
            )

            index = (
                lower_normalized.find(
                    marker.lower()
                )
            )

            if index < 0:
                continue

            relative_part = normalized[
                index + 1:
            ]

            candidate = (
                PROJECT_DIR
                / Path(relative_part)
            )

            if self._is_valid_file(
                candidate
            ):
                return candidate.resolve()

        # --------------------------------------------------------------------
        # 6. Filename fallback
        #
        # Only accept a unique match.
        # --------------------------------------------------------------------

        filename = Path(
            normalized
        ).name

        if filename:

            try:

                matches = list(
                    PROJECT_DIR.rglob(
                        filename
                    )
                )

                valid_matches = [
                    path
                    for path in matches
                    if self._is_valid_file(
                        path
                    )
                ]

                if len(valid_matches) == 1:
                    return (
                        valid_matches[0]
                        .resolve()
                    )

            except Exception:
                pass

        return None

    # ========================================================================
    # Duration bucket
    # ========================================================================

    @classmethod
    def duration_bucket(
        cls,
        duration_sec: float,
    ) -> str:
        """
        Assign a duration bucket.
        """

        for name, low, high in (
            cls.DURATION_BUCKETS
        ):

            if (
                duration_sec >= low
                and duration_sec < high
            ):
                return name

        # Defensive fallback.
        return ">15 sec"

    # ========================================================================
    # Dataset CSV loading
    # ========================================================================

    def _load_csv(
        self,
        split: str,
        csv_path: Path,
    ) -> list[AudioRecord]:

        if not csv_path.exists():
            raise FileNotFoundError(
                f"{split} CSV not found:\n"
                f"{csv_path}"
            )

        df = pd.read_csv(
            csv_path
        )

        expected_size = (
            self.EXPECTED_SPLIT_SIZES[
                split
            ]
        )

        # --------------------------------------------------------------------
        # Dataset integrity
        # --------------------------------------------------------------------

        if len(df) != expected_size:

            raise RuntimeError(
                f"{split} dataset size mismatch.\n"
                f"Expected: {expected_size:,}\n"
                f"Actual:   {len(df):,}\n"
            )

        # --------------------------------------------------------------------
        # Required columns
        # --------------------------------------------------------------------

        required_columns = {
            "audio_path",
            "text",
            "source",
            "orig_script",
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:

            raise RuntimeError(
                f"{split} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        # --------------------------------------------------------------------
        # Duration metadata
        # --------------------------------------------------------------------

        dataset_duration_metadata = (
            self._duration_metadata
            .get("datasets", {})
            .get(split, {})
        )

        if not dataset_duration_metadata:

            raise RuntimeError(
                f"No duration metadata found "
                f"for split '{split}'."
            )

        # --------------------------------------------------------------------
        # We intentionally recalculate durations here from audio files.
        #
        # This gives the DataManager a trustworthy record-level duration
        # rather than depending on only aggregate statistics.
        #
        # The duration analyzer has already established that all files
        # resolve, so this should be safe.
        # --------------------------------------------------------------------

        records: list[AudioRecord] = []

        for index, row in df.iterrows():

            original_audio_path = str(
                row["audio_path"]
            )

            resolved_path = (
                self.resolve_audio_path(
                    original_audio_path
                )
            )

            if resolved_path is None:

                raise RuntimeError(
                    f"Could not resolve audio "
                    f"for {split} row {index}:\n"
                    f"{original_audio_path}"
                )

            # ----------------------------------------------------------------
            # Read actual duration.
            # ----------------------------------------------------------------

            duration_sec = (
                self._read_duration(
                    resolved_path
                )
            )

            if duration_sec is None:

                raise RuntimeError(
                    f"Could not read audio "
                    f"duration for {split} "
                    f"row {index}:\n"
                    f"{resolved_path}"
                )

            projected_tokens = max(
                1,
                int(
                    round(
                        duration_sec
                        * PROJECTED_RATE
                    )
                ),
            )

            bucket = (
                self.duration_bucket(
                    duration_sec
                )
            )

            # ----------------------------------------------------------------
            # Text/source fields
            # ----------------------------------------------------------------

            text = str(
                row["text"]
            )

            source = str(
                row["source"]
            )

            orig_script = str(
                row["orig_script"]
            )

            records.append(
                AudioRecord(
                    index=int(index),

                    audio_path=str(
                        resolved_path
                    ),

                    original_audio_path=(
                        original_audio_path
                    ),

                    text=text,

                    source=source,

                    orig_script=orig_script,

                    duration_sec=float(
                        duration_sec
                    ),

                    estimated_projected_tokens=(
                        projected_tokens
                    ),

                    duration_bucket=bucket,
                )
            )

        return records

    # ========================================================================
    # Audio duration reader
    # ========================================================================

    @staticmethod
    def _read_duration(
        audio_path: Path,
    ) -> Optional[float]:

        try:

            import soundfile as sf

            info = sf.info(
                str(audio_path)
            )

            if info.samplerate <= 0:
                return None

            duration = (
                float(info.frames)
                / float(info.samplerate)
            )

            if duration <= 0:
                return None

            return duration

        except Exception:

            return None

    # ========================================================================
    # Load all datasets
    # ========================================================================

    def load(
        self,
    ) -> None:
        """
        Load train, validation and test.

        The operation is deterministic.
        """

        if self._loaded:
            return

        print()
        print("=" * 78)
        print(
            "V2 DATA MANAGER"
        )
        print("=" * 78)

        print()
        print(
            f"Project: "
            f"{PROJECT_DIR}"
        )

        print(
            f"Duration metadata: "
            f"{self.duration_analysis_path}"
        )

        # --------------------------------------------------------------------
        # Train
        # --------------------------------------------------------------------

        print()
        print(
            "[1/3] Loading TRAIN..."
        )

        self._datasets[
            "train"
        ] = self._load_csv(
            "train",
            TRAIN_CSV,
        )

        print(
            f"      Loaded: "
            f"{len(self._datasets['train']):,}"
        )

        # --------------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------------

        print(
            "[2/3] Loading VAL..."
        )

        self._datasets[
            "val"
        ] = self._load_csv(
            "val",
            VAL_CSV,
        )

        print(
            f"      Loaded: "
            f"{len(self._datasets['val']):,}"
        )

        # --------------------------------------------------------------------
        # Test
        # --------------------------------------------------------------------

        print(
            "[3/3] Loading TEST..."
        )

        self._datasets[
            "test"
        ] = self._load_csv(
            "test",
            TEST_CSV,
        )

        print(
            f"      Loaded: "
            f"{len(self._datasets['test']):,}"
        )

        self._loaded = True

        # --------------------------------------------------------------------
        # Final validation
        # --------------------------------------------------------------------

        self.validate()

    # ========================================================================
    # Validation
    # ========================================================================

    def validate(
        self,
    ) -> None:
        """
        Validate split sizes and basic record integrity.
        """

        for split, expected in (
            self.EXPECTED_SPLIT_SIZES.items()
        ):

            actual = len(
                self._datasets.get(
                    split,
                    []
                )
            )

            if actual != expected:

                raise RuntimeError(
                    f"{split} integrity failure: "
                    f"{actual:,} != "
                    f"{expected:,}"
                )

        # --------------------------------------------------------------------
        # Test split must remain exactly 601 records.
        # --------------------------------------------------------------------

        if len(
            self._datasets["test"]
        ) != 601:

            raise RuntimeError(
                "CRITICAL: Test set is not "
                "exactly 601 samples."
            )

        # --------------------------------------------------------------------
        # Validate every record.
        # --------------------------------------------------------------------

        for split, records in (
            self._datasets.items()
        ):

            for record in records:

                if not record.text:
                    raise RuntimeError(
                        f"Empty text in "
                        f"{split} row "
                        f"{record.index}"
                    )

                if not self._is_valid_file(
                    Path(record.audio_path)
                ):

                    raise RuntimeError(
                        f"Resolved audio "
                        f"does not exist:\n"
                        f"{record.audio_path}"
                    )

                if record.duration_sec <= 0:

                    raise RuntimeError(
                        f"Invalid duration in "
                        f"{split} row "
                        f"{record.index}"
                    )

                if (
                    record.estimated_projected_tokens
                    <= 0
                ):

                    raise RuntimeError(
                        f"Invalid projected "
                        f"sequence length in "
                        f"{split} row "
                        f"{record.index}"
                    )

        print()
        print(
            "Dataset validation: PASS"
        )

        print(
            "Test-set protection: PASS "
            "(601 samples)"
        )

    # ========================================================================
    # Accessors
    # ========================================================================

    def get(
        self,
        split: str,
    ) -> list[AudioRecord]:

        if not self._loaded:
            self.load()

        if split not in self._datasets:

            raise KeyError(
                f"Unknown split: {split}"
            )

        # Return a copy of the list so callers cannot mutate the
        # DataManager's internal split ordering accidentally.
        return list(
            self._datasets[split]
        )

    @property
    def train(
        self,
    ) -> list[AudioRecord]:

        return self.get("train")

    @property
    def val(
        self,
    ) -> list[AudioRecord]:

        return self.get("val")

    @property
    def test(
        self,
    ) -> list[AudioRecord]:

        return self.get("test")

    # ========================================================================
    # Counts
    # ========================================================================

    def counts(
        self,
    ) -> dict[str, int]:

        if not self._loaded:
            self.load()

        return {
            split: len(records)
            for split, records
            in self._datasets.items()
        }

    # ========================================================================
    # Duration distribution
    # ========================================================================

    def duration_distribution(
        self,
        split: str = "train",
    ) -> dict[str, int]:

        records = self.get(
            split
        )

        result = {
            name: 0
            for name, _, _
            in self.DURATION_BUCKETS
        }

        for record in records:
            result[
                record.duration_bucket
            ] += 1

        return result

    # ========================================================================
    # Summary
    # ========================================================================

    def summary(
        self,
    ) -> dict:

        if not self._loaded:
            self.load()

        result = {
            "project_dir": str(
                PROJECT_DIR
            ),

            "splits": self.counts(),

            "duration_distributions": {
                split: self.duration_distribution(
                    split
                )
                for split in (
                    "train",
                    "val",
                    "test",
                )
            },
        }

        return result

    # ========================================================================
    # Iteration
    # ========================================================================

    def iter_split(
        self,
        split: str,
    ) -> Iterator[AudioRecord]:

        records = self.get(
            split
        )

        for record in records:
            yield record

    # ========================================================================
    # Debug sample
    # ========================================================================

    def print_samples(
        self,
        split: str = "train",
        count: int = 5,
    ) -> None:

        records = self.get(
            split
        )

        print()
        print(
            f"{split.upper()} SAMPLE RECORDS"
        )

        print(
            "-" * 78
        )

        for record in records[
            :count
        ]:

            print(
                f"Index:       "
                f"{record.index}"
            )

            print(
                f"Audio:       "
                f"{record.audio_path}"
            )

            print(
                f"Duration:    "
                f"{record.duration_sec:.2f} sec"
            )

            print(
                f"Projected:   "
                f"{record.estimated_projected_tokens}"
                f" tokens"
            )

            print(
                f"Bucket:      "
                f"{record.duration_bucket}"
            )

            print(
                f"Source:      "
                f"{record.source}"
            )

            print(
                f"Orig script: "
                f"{record.orig_script}"
            )

            print(
                f"Text:        "
                f"{record.text[:120]}"
            )

            print(
                "-" * 78
            )

    # ========================================================================
    # Export diagnostic JSON
    # ========================================================================

    def export_diagnostic(
        self,
        output_path: str | Path,
    ) -> None:

        if not self._loaded:
            self.load()

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {

            "project_dir": str(
                PROJECT_DIR
            ),

            "counts": self.counts(),

            "duration_distributions": {
                split: self.duration_distribution(
                    split
                )
                for split in (
                    "train",
                    "val",
                    "test",
                )
            },

            "samples": {
                split: [
                    asdict(record)
                    for record in records[
                        :10
                    ]
                ]
                for split, records
                in self._datasets.items()
            },
        }

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print()
        print(
            f"Diagnostic saved: "
            f"{output_path}"
        )


# ============================================================================
# STANDALONE TEST
# ============================================================================

def main():

    manager = DataManager()

    manager.load()

    print()
    print("=" * 78)
    print(
        "DATA MANAGER SUMMARY"
    )
    print("=" * 78)

    counts = manager.counts()

    print()
    print(
        f"TRAIN: {counts['train']:,}"
    )

    print(
        f"VAL:   {counts['val']:,}"
    )

    print(
        f"TEST:  {counts['test']:,}"
    )

    print(
        f"TOTAL: "
        f"{sum(counts.values()):,}"
    )

    print()
    print(
        "Duration distributions"
    )

    print(
        "-" * 50
    )

    for split in (
        "train",
        "val",
        "test",
    ):

        print()
        print(
            split.upper()
        )

        distribution = (
            manager.duration_distribution(
                split
            )
        )

        for bucket, count in (
            distribution.items()
        ):

            print(
                f"  {bucket:10s} "
                f"{count:6,d}"
            )

    # ------------------------------------------------------------------------
    # Show sample records.
    # ------------------------------------------------------------------------

    manager.print_samples(
        "train",
        count=3,
    )

    manager.print_samples(
        "test",
        count=3,
    )

    # ------------------------------------------------------------------------
    # Export diagnostic.
    # ------------------------------------------------------------------------

    diagnostic_path = (
        V2_DIR
        / "logs"
        / "data_manager_diagnostic.json"
    )

    manager.export_diagnostic(
        diagnostic_path
    )

    print()
    print("=" * 78)
    print(
        "DATA MANAGER TEST COMPLETE"
    )
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()