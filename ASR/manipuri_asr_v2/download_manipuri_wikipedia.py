import json
from pathlib import Path
from datasets import load_dataset

DATASET_ID = "marsh-mellow/manipuri_wikipedia"
OUTPUT_DIR = Path("..") / "external_text"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

txt_path = OUTPUT_DIR / "manipuri_wikipedia.txt"
jsonl_path = OUTPUT_DIR / "manipuri_wikipedia.jsonl"

print(f"Downloading {DATASET_ID}...")
dataset = load_dataset(DATASET_ID)

if hasattr(dataset, "keys") and "train" in dataset:
    split = dataset["train"]
else:
    split = next(iter(dataset.values())) if hasattr(dataset, "values") else dataset

print(f"Records: {len(split):,}")
print(f"Columns: {split.column_names}")

count = 0
mayek_chars = 0
other_alpha = 0

with txt_path.open("w", encoding="utf-8", newline="\n") as txt, \
     jsonl_path.open("w", encoding="utf-8", newline="\n") as jf:

    for i, row in enumerate(split):
        text = row.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue

        text = " ".join(text.split())
        txt.write(text + "\n")
        jf.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

        count += 1
        mayek_chars += sum(0xABC0 <= ord(c) <= 0xABFF for c in text)
        other_alpha += sum(c.isalpha() and not 0xABC0 <= ord(c) <= 0xABFF for c in text)

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1:,}/{len(split):,} records")

print("\nDownload and export completed.")
print(f"Non-empty records: {count:,}")
print(f"Meetei Mayek characters: {mayek_chars:,}")
print(f"Other alphabetic characters: {other_alpha:,}")
print(f"Text file: {txt_path.resolve()}")
print(f"JSONL file: {jsonl_path.resolve()}")
print(f"Text size: {txt_path.stat().st_size:,} bytes")
