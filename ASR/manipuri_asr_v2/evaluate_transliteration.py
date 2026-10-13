

from pathlib import Path
import sys
import csv
import time
import unicodedata
from collections import defaultdict

import pandas as pd

# The converter source is in the sibling xlit directory.
XLIT_DIR = Path(__file__).resolve().parent.parent / "xlit"
sys.path.insert(0, str(XLIT_DIR))

from src.mt_ import MTransliteration

# Run relative file paths from the xlit project directory.
import os
os.chdir(XLIT_DIR)


REFERENCE_FILE = Path("data/transcribed.txt")
OUTPUT_FILE = Path("transliteration_evaluation_results.csv")

def normalize(text):
    return unicodedata.normalize("NFC", str(text)).strip()

def edit_distance(a, b):
    """Levenshtein edit distance using O(len(b)) memory."""
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current = [i]

        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )

        previous = current

    return previous[-1]

def main():
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(
            f"Reference file not found: {REFERENCE_FILE.resolve()}"
        )

    data = pd.read_csv(
        REFERENCE_FILE,
        sep="\t",
        header=None,
        names=["bengali", "reference", "category"],
        dtype=str,
        encoding="utf-8",
        keep_default_na=False,
    )

    converter = MTransliteration()
    results = []
    errors = []
    start = time.time()

    total_distance = 0
    total_reference_chars = 0
    exact_matches = 0
    total = len(data)

    print(f"Loaded {total:,} reference pairs.")
    print("Starting evaluation. This may take a while...")

    for index, row in data.iterrows():
        source = normalize(row["bengali"])
        reference = normalize(row["reference"])
        category = row["category"]

        try:
            prediction = normalize(converter.transliterate(source))
            distance = edit_distance(prediction, reference)
            exact = prediction == reference

            total_distance += distance
            total_reference_chars += len(reference)
            exact_matches += int(exact)

            results.append({
                "bengali": source,
                "reference": reference,
                "prediction": prediction,
                "category": category,
                "exact_match": exact,
                "edit_distance": distance,
                "reference_length": len(reference),
                "error": "",
            })

        except Exception as exc:
            errors.append({
                "row": index + 1,
                "bengali": source,
                "reference": reference,
                "category": category,
                "error": repr(exc),
            })

            results.append({
                "bengali": source,
                "reference": reference,
                "prediction": "",
                "category": category,
                "exact_match": False,
                "edit_distance": len(reference),
                "reference_length": len(reference),
                "error": repr(exc),
            })

        if (index + 1) % 1000 == 0:
            elapsed = time.time() - start
            print(
                f"Processed {index + 1:,}/{total:,} | "
                f"elapsed: {elapsed:.1f}s"
            )

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    successful = total - len(errors)

    print("\n========== OVERALL RESULTS ==========")
    print(f"Total reference pairs: {total:,}")
    print(f"Successful conversions: {successful:,}")
    print(f"Conversion exceptions: {len(errors):,}")

    if successful:
        print(
            f"Exact word accuracy: "
            f"{100 * exact_matches / successful:.2f}%"
        )

    if total_reference_chars:
        print(
            f"Character error rate (CER): "
            f"{100 * total_distance / total_reference_chars:.2f}%"
        )

    print("\n========== RESULTS BY CATEGORY ==========")

    for category, group in result_df.groupby("category", sort=True):
        count = len(group)
        accuracy = 100 * group["exact_match"].sum() / count
        distance = group["edit_distance"].sum()
        reference_chars = group["reference_length"].sum()
        cer = 100 * distance / reference_chars if reference_chars else 0

        print(
            f"Category {category}: "
            f"pairs={count:,}, "
            f"exact_accuracy={accuracy:.2f}%, "
            f"CER={cer:.2f}%"
        )

    if errors:
        pd.DataFrame(errors).to_csv(
            "transliteration_runtime_errors.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print("\nRuntime errors saved to transliteration_runtime_errors.csv")

    print(f"\nDetailed predictions saved to: {OUTPUT_FILE.resolve()}")
    print(f"Total time: {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
