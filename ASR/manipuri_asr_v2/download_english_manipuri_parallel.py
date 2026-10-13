from datasets import load_dataset
from pathlib import Path
import json

OUT = Path("..") / "external_text"
OUT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("bible", "bible"),
    ("pib-pmi", "pib_pmi"),
]

for config, split in CONFIGS:
    print(f"\nLoading {config}...")
    ds = load_dataset(
        "joyson117/english-manipuri-parallel-corpus",
        name=config,
        split=split,
    )

    print(f"Records: {len(ds):,}")
    print(f"Columns: {ds.column_names}")

    output = OUT / f"english_manipuri_{config}.jsonl"
    count = 0

    with output.open("w", encoding="utf-8", newline="\n") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

            if count % 25000 == 0:
                print(f"Saved {count:,} pairs")

    print(f"Saved {count:,} pairs to {output.resolve()}")
    print(f"File size: {output.stat().st_size:,} bytes")

print("\nBoth corpus downloads completed.")
