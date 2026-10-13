import os
import requests
from datasets import load_dataset
from normalizer import clean_text

OUTPUT_DIR = r"..\external_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Fetch IndicCorp V2 Manipuri Subset
print("[1/2] Fetching IndicCorp V2 Manipuri text...")
try:
    indic_corp = load_dataset("ai4bharat/IndicCorpV2", "mni", split="train", streaming=True)
    out_file = os.path.join(OUTPUT_DIR, "indiccorp_mni.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        count = 0
        for sample in indic_corp:
            cleaned = clean_text(sample["text"])
            if len(cleaned) > 5:
                f.write(cleaned + "\n")
                count += 1
                if count >= 100000:  # Cap at 100k lines for initial sweep
                    break
    print(f"✓ Saved {count} lines to {out_file}")
except Exception as e:
    print(f"IndicCorp fetch error: {e}")

# 2. Fetch Sangraha Manipuri Subset
print("\n[2/2] Fetching Sangraha Manipuri subset...")
try:
    sangraha = load_dataset("ai4bharat/sangraha", "mni", split="train", streaming=True)
    out_file = os.path.join(OUTPUT_DIR, "sangraha_mni.txt")
    with open(out_file, "w", encoding="utf-8") as f:
        count = 0
        for sample in sangraha:
            cleaned = clean_text(sample["text"])
            if len(cleaned) > 5:
                f.write(cleaned + "\n")
                count += 1
                if count >= 100000:
                    break
    print(f"✓ Saved {count} lines to {out_file}")
except Exception as e:
    print(f"Sangraha fetch error: {e}")