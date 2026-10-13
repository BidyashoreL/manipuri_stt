import json
from pathlib import Path

p = Path(r".\manipuri_stt\ASR\v2\logs\duration_analysis.json")

with open(p, encoding="utf-8") as f:
    data = json.load(f)

for split in ["train", "val", "test"]:
    print()
    print("=" * 70)
    print(split.upper())
    print("=" * 70)

    items = data["datasets"][split]["missing_audio_examples"]

    for item in items:
        print(item["original_path"])

