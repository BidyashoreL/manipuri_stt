from datasets import load_dataset
from pathlib import Path
import time

OUT = Path("..") / "external_text" / "sangraha_mni_verified.txt"
TARGET = 100_000
OUT.parent.mkdir(parents=True, exist_ok=True)

# Keep the existing corpus; don't overwrite it.
existing = set()
if OUT.exists():
    with OUT.open("r", encoding="utf-8") as f:
        existing = {line.rstrip("\r\n") for line in f if line.strip()}

print(f"Existing unique records: {len(existing):,}")

for attempt in range(1, 6):
    try:
        ds = load_dataset(
            "ai4bharat/sangraha",
            data_dir="verified/mni",
            split="train",
            streaming=True,
        )

        added = 0
        seen_this_run = set()

        with OUT.open("a", encoding="utf-8", newline="\n") as f:
            for row in ds:
                text = row.get("text", "")
                if not isinstance(text, str):
                    continue
                text = text.strip().replace("\r", " ").replace("\n", " ")
                if not text or text in existing or text in seen_this_run:
                    continue

                f.write(text + "\n")
                seen_this_run.add(text)
                added += 1

                if added % 500 == 0:
                    f.flush()
                    print(f"New unique records this attempt: {added:,}")

                if len(existing) + added >= TARGET:
                    break

        existing.update(seen_this_run)
        print(f"Attempt {attempt}: added {added:,}; total unique: {len(existing):,}")

        if len(existing) >= TARGET:
            print("TARGET REACHED")
            break

        # Streaming may have ended normally before the target.
        print("Stream ended before target; retrying to check connectivity.")
    except Exception as exc:
        print(f"Attempt {attempt} failed: {type(exc).__name__}: {exc}")

    if attempt < 5:
        wait = attempt * 5
        print(f"Waiting {wait}s before retry...")
        time.sleep(wait)

print(f"Final unique records: {len(existing):,}")
print(f"File: {OUT.resolve()}")
