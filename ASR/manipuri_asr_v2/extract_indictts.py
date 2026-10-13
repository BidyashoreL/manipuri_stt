
import csv
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

SOURCE = Path(
    r"E:\hf_cache\hub\datasets--SPRINGLab--IndicTTS_Manipuri"
    r"\datasets--SPRINGLab--IndicTTS_Manipuri\snapshots"
    r"\a7472c5ec603e22a9ab1f1d52aa44e2b23b24d14\data"
)

OUTPUT = Path(r"E:\manipuri_indictts_extracted")
AUDIO_DIR = OUTPUT / "audio"
MANIFEST = OUTPUT / "metadata.csv"
PROGRESS = OUTPUT / "progress.json"

OUTPUT.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

shards = sorted(SOURCE.glob("train-*.parquet"))
if not shards:
    sys.exit(f"No Parquet shards found: {SOURCE}")

total = sum(pq.ParquetFile(p).metadata.num_rows for p in shards)

# Recover previously extracted records only if their audio exists.
existing = {}
if MANIFEST.exists():
    with MANIFEST.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (AUDIO_DIR / row["file_name"]).is_file():
                existing[row["file_name"]] = row

print(f"Source records: {total:,}")
print(f"Already extracted: {len(existing):,}")
print("Starting extraction...")

count = len(existing)
skipped = 0

try:
    with MANIFEST.open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file_name", "text", "gender", "source_path"],
        )
        writer.writeheader()
        writer.writerows(existing.values())

        for shard_num, shard in enumerate(shards, start=1):
            parquet = pq.ParquetFile(shard)
            print(f"\n[{shard_num}/{len(shards)}] {shard.name}")

            for batch in parquet.iter_batches(
                batch_size=32,
                columns=["audio", "text", "gender"],
            ):
                for row in batch.to_pylist():
                    audio = row.get("audio") or {}
                    audio_bytes = audio.get("bytes")
                    source_path = audio.get("path") or "audio.wav"
                    text = row.get("text")
                    gender = row.get("gender")

                    if not audio_bytes or not text:
                        skipped += 1
                        continue

                    original_stem = Path(source_path).stem
                    filename = f"{original_stem}.wav"
                    destination = AUDIO_DIR / filename

                    # Keep same-named recordings from different shards distinct.
                    if filename in existing or destination.exists():
                        filename = f"shard{shard_num}_{original_stem}.wav"
                        destination = AUDIO_DIR / filename

                    try:
                        # Check the WAV header before writing.
                        if not audio_bytes.startswith(b"RIFF") or audio_bytes[8:12] != b"WAVE":
                            skipped += 1
                            continue

                        destination.write_bytes(audio_bytes)

                        record = {
                            "file_name": filename,
                            "text": text,
                            "gender": gender,
                            "source_path": source_path,
                        }
                        writer.writerow(record)
                        existing[filename] = record
                        count += 1

                    except OSError as exc:
                        print(f"\nDisk/write error: {exc}")
                        raise

                    if count % 500 == 0:
                        f.flush()
                        PROGRESS.write_text(
                            json.dumps({
                                "extracted": count,
                                "skipped": skipped,
                                "total": total,
                                "complete": False,
                            }, indent=2),
                            encoding="utf-8",
                        )
                        print(f"Extracted {count:,}/{total:,}")

except KeyboardInterrupt:
    print("\nInterrupted. Run again to resume.")
    sys.exit(1)

PROGRESS.write_text(
    json.dumps({
        "extracted": count,
        "skipped_this_run": skipped,
        "total": total,
        "complete": count + skipped == total,
    }, indent=2),
    encoding="utf-8",
)

print("\n========== EXTRACTION REPORT ==========")
print(f"Source records: {total:,}")
print(f"Audio extracted/preserved: {count:,}")
print(f"Skipped this run: {skipped:,}")
print(f"Audio folder: {AUDIO_DIR}")
print(f"Metadata CSV: {MANIFEST}")
print("======================================")
