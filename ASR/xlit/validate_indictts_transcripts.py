from pathlib import Path
import re
import pandas as pd

BASE = Path(
    r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt\Manipuri_Datasets\02_IndicTTS"
)

METADATA = BASE / "metadata_meitei_mayek_preserved.csv"
ERRORS = BASE / "transliteration_errors.csv"
REPORT = BASE / "transcript_validation_report.csv"

# Bengali letters and combining marks. Digits are checked separately.
BENGALI_LETTERS = re.compile(
    r"[\u0985-\u09B9\u09BC-\u09CD\u09DC-\u09DF\u09E0-\u09E1]"
)
BENGALI_DIGITS = re.compile(r"[\u09E6-\u09EF]")
MEITEI_MAYEK = re.compile(r"[\uABC0-\uABFF]")
REPLACEMENT = "\uFFFD"

PUNCTUATION = re.compile(r"[,.;!?।৷ঃ৳৹]")

def main():
    df = pd.read_csv(METADATA, dtype=str).fillna("")
    errors = pd.read_csv(ERRORS, dtype=str).fillna("")

    required = {"file_name", "text", "text_meitei_mayek"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    # Error-log row numbers refer to CSV line numbers, including the header.
    failed_rows = set()
    for value in errors["row"]:
        try:
            failed_rows.add(int(value) - 2)
        except ValueError:
            pass

    report = []

    for i, row in df.iterrows():
        original = row["text"]
        converted = row["text_meitei_mayek"]

        bengali_letters = len(BENGALI_LETTERS.findall(converted))
        bengali_digits = len(BENGALI_DIGITS.findall(converted))
        meitei_chars = len(MEITEI_MAYEK.findall(converted))

        flags = []

        if not original.strip():
            flags.append("EMPTY_ORIGINAL")
        if not converted.strip():
            flags.append("EMPTY_CONVERTED")
        if bengali_letters:
            flags.append("BENGALI_LETTERS_REMAIN")
        if REPLACEMENT in converted:
            flags.append("UNICODE_REPLACEMENT_CHARACTER")
        if original and not meitei_chars:
            flags.append("NO_MEITEI_MAYEK_CHARACTERS")
        if i in failed_rows:
            flags.append("ROW_HAS_LOGGED_CONVERSION_FAILURE")

        if bool(re.search(r"\s", original)) != bool(
            re.search(r"\s", converted)
        ):
            flags.append("WHITESPACE_PRESENCE_CHANGED")

        if len(PUNCTUATION.findall(original)) != len(
            PUNCTUATION.findall(converted)
        ):
            flags.append("PUNCTUATION_COUNT_CHANGED")

        report.append({
            "row_index": i,
            "file_name": row["file_name"],
            "original": original,
            "converted": converted,
            "meitei_mayek_characters": meitei_chars,
            "remaining_bengali_letters": bengali_letters,
            "remaining_bengali_digits": bengali_digits,
            "flags": ";".join(flags),
        })

    result = pd.DataFrame(report)
    result.to_csv(REPORT, index=False, encoding="utf-8-sig")

    flagged = result[result["flags"] != ""]

    print("\n========== TRANSCRIPT VALIDATION ==========")
    print("Total rows:", len(df))
    print("Unique audio filenames:", df["file_name"].nunique())
    print("Empty original transcripts:", (df["text"].str.strip() == "").sum())
    print("Empty converted transcripts:",
          (df["text_meitei_mayek"].str.strip() == "").sum())
    print("Rows retaining Bengali letters:",
          (result["remaining_bengali_letters"] > 0).sum())
    print("Rows retaining Bengali digits:",
          (result["remaining_bengali_digits"] > 0).sum())
    print("Rows containing Meitei Mayek characters:",
          (result["meitei_mayek_characters"] > 0).sum())
    print("Rows with logged conversion failures:",
          result["flags"].str.contains(
              "ROW_HAS_LOGGED_CONVERSION_FAILURE", regex=False
          ).sum())
    print("Rows flagged for review:", len(flagged))
    print("Report:", REPORT)

    if len(flagged):
        print("\nFirst 20 flagged rows:")
        print(flagged[["row_index", "file_name", "flags",
                       "original", "converted"]].head(20).to_string(index=False))

    print("\nNote: flags identify candidates for review, not necessarily errors.")
    print("Bengali digits may be intentionally retained.")

if __name__ == "__main__":
    main()
