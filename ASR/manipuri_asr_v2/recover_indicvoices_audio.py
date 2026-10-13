import os
import ntpath
from collections import defaultdict

import pandas as pd
import pyarrow.parquet as pq

ASR_ROOT = (
    r"E:\manipuri_stt1\manipuri_stt4"
    r"\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt\ASR"
)
PARQUET_DIR = os.path.join(
    os.path.dirname(ASR_ROOT), "manipuri_10k", "Manipuri"
)
OUT_ROOT = os.path.join(ASR_ROOT, "recovered_indicvoices")

SPLITS = {
    split: os.path.join(ASR_ROOT, "combined_dataset", f"{split}.csv")
    for split in ("train", "val", "test")
}

for split, path in SPLITS.items():
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing manifest: {path}")

if not os.path.isdir(PARQUET_DIR):
    raise FileNotFoundError(f"Missing Parquet folder: {PARQUET_DIR}")

os.makedirs(OUT_ROOT, exist_ok=True)

manifests = {}
requests = defaultdict(lambda: defaultdict(list))
missing_shards = set()
failures = []

print("=" * 60)
print("INDICVOICES AUDIO RECOVERY - FIXED")
print("=" * 60)

# Load the CSVs and force path columns to string-compatible object dtype.
for split, csv_path in SPLITS.items():
    df = pd.read_csv(csv_path, low_memory=False)

    for column in ("audio_path", "audio_file"):
        if column not in df.columns:
            df[column] = pd.Series([None] * len(df), dtype=object)
        else:
            df[column] = df[column].astype(object)

    manifests[split] = df
    print(f"{split}: {len(df):,} rows")

    for idx, row in df.iterrows():
        source = str(row.get("source", "")).strip().lower()
        if source not in ("indicvoices", "indicvoices_test"):
            continue

        pv, rv = row.get("parquet_file"), row.get("parquet_row")
        if pd.isna(pv) or pd.isna(rv):
            failures.append((split, idx, "Missing Parquet mapping"))
            continue

        shard = ntpath.basename(str(pv).strip())
        shard_path = os.path.join(PARQUET_DIR, shard)

        try:
            row_num = int(rv)
        except (ValueError, TypeError, OverflowError):
            failures.append((split, idx, "Invalid Parquet row"))
            continue

        if not os.path.isfile(shard_path):
            missing_shards.add(shard)
            continue

        requests[shard][row_num].append((split, idx))

print(f"\nRequired shards: {len(requests)}")
print(f"Unique requested rows: {sum(len(v) for v in requests.values()):,}")

written = 0
existing = 0
processed = 0

for shard in sorted(requests):
    path = os.path.join(PARQUET_DIR, shard)
    row_map = requests[shard]
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows

    for r in list(row_map):
        if r < 0 or r >= total:
            failures.append((shard, r, "Row index outside shard"))
            del row_map[r]

    print(f"\n[{processed + 1}/{len(requests)}] {shard}")

    offset = 0
    for batch in pf.iter_batches(batch_size=16, columns=["audio"]):
        records = batch.to_pylist()

        for j, record in enumerate(records):
            row_num = offset + j
            targets = row_map.get(row_num)
            if not targets:
                continue

            audio = record.get("audio") or {}
            data = audio.get("bytes")
            if not data:
                failures.append((shard, row_num, "Missing audio bytes"))
                continue

            for split, idx in targets:
                audio_dir = os.path.join(OUT_ROOT, split, "audio")
                os.makedirs(audio_dir, exist_ok=True)

                stem = os.path.splitext(shard)[0]
                filename = f"{stem}_row_{row_num:06d}.wav"
                dest = os.path.abspath(os.path.join(audio_dir, filename))

                try:
                    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                        existing += 1
                    else:
                        with open(dest, "wb") as f:
                            f.write(data)
                        written += 1

                    manifests[split].at[idx, "audio_path"] = dest
                    manifests[split].at[idx, "audio_file"] = filename

                except OSError as exc:
                    failures.append((shard, row_num, str(exc)))

        offset += len(records)

    processed += 1

# Save new CSVs only. Original manifests remain untouched.
print("\nSaving recovered manifests...")
for split, df in manifests.items():
    output = os.path.join(OUT_ROOT, f"{split}.csv")
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"{split}: {output}")

if missing_shards:
    print("\nMissing shards:")
    for name in sorted(missing_shards):
        print(" ", name)

if failures:
    report = os.path.join(OUT_ROOT, "recovery_failures.csv")
    pd.DataFrame(
        failures, columns=["source_or_split", "row", "reason"]
    ).to_csv(report, index=False, encoding="utf-8-sig")
    print(f"Failure report: {report}")

print("\n" + "=" * 60)
print("RECOVERY SUMMARY")
print("=" * 60)
print("Shards processed:            ", processed)
print("New audio files written:     ", written)
print("Existing files reused:       ", existing)
print("Failures recorded:           ", len(failures))
print("Missing shards:              ", len(missing_shards))
print("Output folder:               ", OUT_ROOT)
print("=" * 60)
