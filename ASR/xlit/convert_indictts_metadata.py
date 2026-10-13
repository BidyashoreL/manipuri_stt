from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.mt_ import MTransliteration

SOURCE = Path(r"E:\manipuri_indictts_extracted\metadata.csv")
OUTPUT = Path(
    r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt\Manipuri_Datasets\02_IndicTTS\metadata_meitei_mayek.csv"
)

def has_bengali(text):
    return any("\u0980" <= ch <= "\u09FF" for ch in text)

def has_meitei_mayek(text):
    return any("\uABC0" <= ch <= "\uABFF" for ch in text)

def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Metadata not found: {SOURCE}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SOURCE, dtype={"text": str}).fillna({"text": ""})

    converter = MTransliteration()
    converted = []
    statuses = []

    for text in tqdm(df["text"], desc="Converting transcripts"):
        text = str(text)

        if has_meitei_mayek(text) and not has_bengali(text):
            converted.append(text)
            statuses.append("already_meitei_mayek")
        elif has_bengali(text):
            result = converter.transliterate(text)
            converted.append(result)
            statuses.append(
                "converted" if result != text else "unchanged"
            )
        else:
            converted.append(text)
            statuses.append("no_bengali_script")

    df["text_meitei_mayek"] = converted
    df["conversion_status"] = statuses
    df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print("\n=== CONVERSION SUMMARY ===")
    print(f"Input rows: {len(df):,}")
    print(pd.Series(statuses).value_counts().to_string())
    print(f"\nOutput: {OUTPUT}")

    print("\n=== SAMPLE RESULTS ===")
    for _, row in df.head(10).iterrows():
        print(f"\nOriginal: {row['text']}")
        print(f"Mayek:    {row['text_meitei_mayek']}")
        print(f"Status:   {row['conversion_status']}")

if __name__ == "__main__":
    main()
