
from pathlib import Path
import re
import pandas as pd
from tqdm import tqdm
from src.mt_ import MTransliteration

SOURCE = Path(
    r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt\Manipuri_Datasets\02_IndicTTS\metadata.csv"
)
OUTPUT = SOURCE.with_name("metadata_meitei_mayek_preserved.csv")
ERROR_LOG = SOURCE.with_name("transliteration_errors.csv")

BENGALI = re.compile(r"[\u0980-\u09FF]+")
SEGMENTS = re.compile(r"[\u0980-\u09FF]+|[^\u0980-\u09FF]+")


def main():
    df = pd.read_csv(SOURCE, dtype=str).fillna("")
    converter = MTransliteration()
    errors = []

    def convert_text(text, row_index):
        result = []

        for segment in SEGMENTS.findall(text):
            if not BENGALI.fullmatch(segment):
                result.append(segment)
                continue

            try:
                converted = converter.transliterate(segment)

                # Empty output must not silently erase source text.
                if not converted or not converted.strip():
                    errors.append({
                        "row": row_index,
                        "segment": segment,
                        "reason": "Empty conversion",
                    })
                    result.append(segment)
                else:
                    result.append(converted)

            except (IndexError, ValueError, TypeError) as exc:
                errors.append({
                    "row": row_index,
                    "segment": segment,
                    "reason": f"{type(exc).__name__}: {exc}",
                })
                result.append(segment)

        return "".join(result)

    df["text_meitei_mayek"] = [
        convert_text(text, i)
        for i, text in enumerate(
            tqdm(df["text"], desc="Converting transcripts"), start=2
        )
    ]

    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    error_df = pd.DataFrame(
        errors, columns=["row", "segment", "reason"]
    )
    error_df.to_csv(ERROR_LOG, index=False, encoding="utf-8-sig")

    print("\n--- Conversion report ---")
    print("Rows:", len(df))
    print("Conversion errors:", len(errors))
    print("Original rows with spaces:",
          df["text"].str.contains(r"\s").sum())
    print("Output rows with spaces:",
          df["text_meitei_mayek"].str.contains(r"\s").sum())
    print("Original rows with punctuation:",
          df["text"].str.contains(r"[,.;!?।]").sum())
    print("Output rows with punctuation:",
          df["text_meitei_mayek"].str.contains(r"[,.;!?।]").sum())

    print("\n--- Examples ---")
    for _, row in df.head(5).iterrows():
        print("\nOriginal :", row["text"])
        print("Converted:", row["text_meitei_mayek"])

    print("\nConverted metadata:", OUTPUT)
    print("Error report:", ERROR_LOG)
    print("Review the error report before using the labels for training.")


if __name__ == "__main__":
    main()
