from pathlib import Path
from collections import defaultdict
import hashlib
import pandas as pd

report = Path("audio_audit/all_splits_silence_report.csv")
out = Path("audio_audit/cleaning_plan")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(report)
df["audio_path"] = df["audio_path"].astype(str)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

hash_to_rows = defaultdict(list)
missing = []

print("Hashing audio files. This may take a while...")

for i, row in df.iterrows():
    path = Path(row["audio_path"])
    if not path.is_file():
        missing.append({
            "audio_path": str(path),
            "source": row.get("source", ""),
            "split": row.get("split", "")
        })
        continue

    try:
        digest = sha256_file(path)
        hash_to_rows[digest].append({
            "audio_path": str(path),
            "source": row.get("source", ""),
            "split": row.get("split", ""),
            "text": row.get("text", "")
        })
    except Exception as e:
        missing.append({
            "audio_path": str(path),
            "source": row.get("source", ""),
            "split": row.get("split", ""),
            "error": str(e)
        })

duplicates = []
leakage = []

for digest, rows in hash_to_rows.items():
    if len(rows) > 1:
        for item in rows:
            duplicates.append({"sha256": digest, **item})

        splits = {str(r["split"]).lower() for r in rows}
        if "train" in splits and "test" in splits:
            for item in rows:
                leakage.append({"sha256": digest, **item})

pd.DataFrame(duplicates).to_csv(
    out / "exact_audio_duplicates.csv", index=False
)
pd.DataFrame(leakage).to_csv(
    out / "train_test_audio_leakage.csv", index=False
)
pd.DataFrame(missing).to_csv(
    out / "missing_or_unreadable_audio.csv", index=False
)

print("\n========== AUDIO INTEGRITY RESULTS ==========")
print("Manifest entries:", len(df))
print("Unique audio hashes:", len(hash_to_rows))
print("Entries in duplicate groups:", len(duplicates))
print("Train-test leakage entries:", len(leakage))
print("Missing/unreadable entries:", len(missing))
print("\nReports saved in:", out.resolve())
print("Original audio and manifests were not changed.")
