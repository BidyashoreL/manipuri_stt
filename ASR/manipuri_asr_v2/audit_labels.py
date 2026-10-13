from pathlib import Path
import pandas as pd
import unicodedata
import json
from collections import Counter

ROOT = Path(__file__).resolve().parent
ASR = ROOT.parent

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"

OUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# CHARACTER CLASSIFICATION
# ---------------------------------------------------------

def classify_char(ch):

    cp = ord(ch)

    # whitespace
    if ch.isspace():
        return "SPACE"

    # Meetei Mayek block
    if 0xABC0 <= cp <= 0xABFF:
        return "MEITEI_MAYEK"

    # ASCII Latin
    if "A" <= ch <= "Z" or "a" <= ch <= "z":
        return "LATIN"

    # ASCII digits
    if "0" <= ch <= "9":
        return "DIGIT"

    # angle brackets
    if ch in "<>":
        return "ANGLE_BRACKET"

    # punctuation
    if unicodedata.category(ch).startswith("P"):
        return "PUNCTUATION"

    # symbols
    if unicodedata.category(ch).startswith("S"):
        return "SYMBOL"

    return "OTHER"


def analyze_text(text):

    counts = Counter()

    for ch in text:
        counts[classify_char(ch)] += 1

    return counts


# ---------------------------------------------------------
# ANALYZE MANIFEST
# ---------------------------------------------------------

def analyze_manifest(csv_path, name):

    df = pd.read_csv(csv_path)

    print()
    print("=" * 80)
    print(f"{name.upper()} LABEL AUDIT")
    print("=" * 80)

    print(f"Rows: {len(df)}")

    records = []

    global_counts = Counter()

    for idx, row in df.iterrows():

        text = str(row.get("text", ""))

        counts = analyze_text(text)

        global_counts.update(counts)

        categories = set(counts.keys())

        has_latin = "LATIN" in categories
        has_digit = "DIGIT" in categories
        has_angle = "ANGLE_BRACKET" in categories
        has_other = "OTHER" in categories
        has_punct = "PUNCTUATION" in categories
        has_symbol = "SYMBOL" in categories

        # clean means:
        # Meitei Mayek + spaces only
        clean = not (
            has_latin
            or has_digit
            or has_angle
            or has_other
            or has_punct
            or has_symbol
        )

        records.append(
            {
                "index": idx,
                "text": text,
                "length": len(text),
                "meitei_mayek": counts["MEITEI_MAYEK"],
                "spaces": counts["SPACE"],
                "latin": counts["LATIN"],
                "digits": counts["DIGIT"],
                "angle_brackets": counts["ANGLE_BRACKET"],
                "punctuation": counts["PUNCTUATION"],
                "symbols": counts["SYMBOL"],
                "other": counts["OTHER"],
                "clean": clean,
                "source": row.get("source", ""),
            }
        )

    result = pd.DataFrame(records)

    print()
    print("SAMPLE COUNTS")
    print("-" * 80)

    print(f"Clean labels       : {result['clean'].sum()}")
    print(f"Contaminated       : {(~result['clean']).sum()}")
    print(
        f"Contamination %    : "
        f"{100 * (~result['clean']).mean():.4f}%"
    )

    print()
    print("CONTAMINATION TYPES")
    print("-" * 80)

    for col in [
        "latin",
        "digits",
        "angle_brackets",
        "punctuation",
        "symbols",
        "other",
    ]:

        samples = int((result[col] > 0).sum())
        chars = int(result[col].sum())

        print(
            f"{col:18s}: "
            f"{samples:5d} samples | "
            f"{chars:6d} characters"
        )

    print()
    print("GLOBAL CHARACTER COUNTS")
    print("-" * 80)

    for key, value in global_counts.most_common():

        print(f"{key:20s}: {value:,}")

    output = OUT_DIR / f"{name}_label_audit.csv"

    result.to_csv(
        output,
        index=False,
        encoding="utf-8-sig",
    )

    # Save contaminated samples separately
    contaminated = result[~result["clean"]]

    contaminated_output = (
        OUT_DIR / f"{name}_contaminated_labels.csv"
    )

    contaminated.to_csv(
        contaminated_output,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("FIRST CONTAMINATED SAMPLES")
    print("-" * 80)

    if len(contaminated) > 0:

        print(
            contaminated[
                [
                    "index",
                    "source",
                    "latin",
                    "digits",
                    "angle_brackets",
                    "punctuation",
                    "symbols",
                    "other",
                    "text",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    return result, global_counts


train_result, train_counts = analyze_manifest(
    TRAIN_CSV,
    "train",
)

val_result, val_counts = analyze_manifest(
    VAL_CSV,
    "val",
)


# ---------------------------------------------------------
# SAVE SUMMARY
# ---------------------------------------------------------

summary = {
    "train_rows": len(train_result),
    "train_clean": int(train_result["clean"].sum()),
    "train_contaminated": int((~train_result["clean"]).sum()),

    "val_rows": len(val_result),
    "val_clean": int(val_result["clean"].sum()),
    "val_contaminated": int((~val_result["clean"]).sum()),

    "train_latin_samples": int(
        (train_result["latin"] > 0).sum()
    ),

    "val_latin_samples": int(
        (val_result["latin"] > 0).sum()
    ),

    "train_angle_samples": int(
        (train_result["angle_brackets"] > 0).sum()
    ),

    "val_angle_samples": int(
        (val_result["angle_brackets"] > 0).sum()
    ),
}


summary_path = OUT_DIR / "label_audit_summary.json"

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print("=" * 80)
print("FINAL LABEL AUDIT SUMMARY")
print("=" * 80)

print(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    )
)

print()
print("DONE.")