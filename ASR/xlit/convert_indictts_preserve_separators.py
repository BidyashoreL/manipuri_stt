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

    def convert_text(text, row_number):
        result = []

        for segment in SEGMENTS.findall(text):
            if not BENGALI.fullmatch(segment):
                result.append(segment)
                continue

            try:
                converted = converter.transliterate(segment)

                if not converted or not converted.strip():
                    errors.append({
                        "row": row_number,
                        "segment": segment,
                        "reason": "Empty conversion"
                    })
                    result.append(segment)
                else:
                    result.append(converted)

            except Exception as exc:
                errors.append({
                    "row": row_number,
                    "segment": segment,
                    "reason": f"{type(exc).__name__}: {exc}"
                })
                result.append(segment)

        return "".join(result)

    converted_texts = []

    for row_number, text in enumerate(
        tqdm(df["text"], desc="Converting transcripts"), start=2
    ):
        converted_texts.append(convert_text(text, row_number))

    df["text_meitei_mayek"] = converted_texts
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    pd.DataFrame(
        errors, columns=["row", "segment", "reason"]
    ).to_csv(ERROR_LOG, index=False, encoding="utf-8-sig")

    print("\n--- Conversion report ---")
    print("Rows:", len(df))
    print("Failed segments:", len(errors))
    print("Original rows with spaces:",
          df["text"].str.contains(r"\s").sum())
    print("Converted rows with spaces:",
          df["text_meitei_mayek"].str.contains(r"\s").sum())
    print("Original rows with punctuation:",
          df["text"].str.contains(r"[,.;!?।]").sum())
    print("Converted rows with punctuation:",
          df["text_meitei_mayek"].str.contains(r"[,.;!?।]").sum())

    for _, row in df.head(5).iterrows():
        print("\nOriginal :", row["text"])
        print("Converted:", row["text_meitei_mayek"])

    print("\nOutput:", OUTPUT)
    print("Errors:", ERROR_LOG)


if __name__ == "__main__":
    main()
