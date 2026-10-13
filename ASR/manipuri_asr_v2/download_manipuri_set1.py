from datasets import load_dataset
from pathlib import Path

output_dir = Path("..") / "external_text"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "manipuri_monolingual_set1.txt"
progress_file = output_dir / "manipuri_monolingual_set1_progress.txt"

print("Loading Manipuri Monolingual Corpus — Set 1...")

dataset = load_dataset(
    "joyson117/manipuri-monolingual-corpus",
    name="set1",
    split="set1",
)

print("Dataset loaded.")
print("Records:", len(dataset))
print("Columns:", dataset.column_names)

with output_file.open("w", encoding="utf-8", newline="\n") as out:
    with progress_file.open("w", encoding="utf-8") as log:
        for i, row in enumerate(dataset):
            # Inspect the dataset's fields without assuming its schema.
            text = row.get("text")

            if not isinstance(text, str):
                # Fallback for datasets using a different text column.
                text = next(
                    (v for v in row.values() if isinstance(v, str)),
                    None,
                )

            if text and text.strip():
                out.write(text.strip().replace("\r", " ").replace("\n", " ") + "\n")

            if (i + 1) % 10000 == 0:
                log.write(f"{i + 1}\n")
                log.flush()
                print(f"Processed {i + 1:,} records")

print("\nDownload/extraction completed.")
print("Output:", output_file.resolve())
print("Size:", f"{output_file.stat().st_size:,} bytes")
print("Progress log:", progress_file.resolve())
