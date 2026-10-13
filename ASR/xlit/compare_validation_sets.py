from pathlib import Path
import pandas as pd
import re
import unicodedata

root = Path.cwd()
old_path = root.parent / "combined_dataset" / "val.csv"
clean_path = root / "audio_audit" / "final_candidate" / "validation_16k.csv"

old = pd.read_csv(old_path, encoding="utf-8-sig").fillna("")
clean = pd.read_csv(clean_path, encoding="utf-8-sig").fillna("")

def norm_text(x):
    x = unicodedata.normalize("NFC", str(x))
    return re.sub(r"\s+", " ", x).strip()

def norm_path(x):
    x = str(x).strip()
    if not x:
        return ""
    return str(Path(x).resolve()).casefold()

def show_info(name, df):
    print(f"\n--- {name} ---")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

show_info("ORIGINAL VALIDATION", old)
show_info("CLEANED 16K VALIDATION", clean)

print("\n--- ORIGINAL SOURCE DISTRIBUTION ---")
if "source" in old.columns:
    print(old["source"].value_counts(dropna=False).to_string())

print("\n--- CLEANED SOURCE DISTRIBUTION ---")
if "source" in clean.columns:
    print(clean["source"].value_counts(dropna=False).to_string())

# Compare stable source identities when the fields are available.
identity_cols = ["source", "parquet_file", "parquet_row"]
if all(c in old.columns and c in clean.columns for c in identity_cols):
    def identity(df):
        return df[identity_cols].astype(str).apply(
            lambda r: "|".join(v.strip().casefold() for v in r), axis=1
        )
    old_ids, clean_ids = set(identity(old)), set(identity(clean))
    print("\n--- SOURCE IDENTITY OVERLAP ---")
    print("Original unique identities:", len(old_ids))
    print("Cleaned unique identities:", len(clean_ids))
    print("Identities in both:", len(old_ids & clean_ids))
    print("Original-only identities:", len(old_ids - clean_ids))
    print("Cleaned-only identities:", len(clean_ids - old_ids))

# Compare audio paths where they refer to the same files.
if "audio_path" in old.columns and "audio_path" in clean.columns:
    old_paths = {norm_path(x) for x in old["audio_path"] if str(x).strip()}
    clean_paths = {norm_path(x) for x in clean["audio_path"] if str(x).strip()}
    print("\n--- AUDIO PATH OVERLAP ---")
    print("Original unique paths:", len(old_paths))
    print("Cleaned unique paths:", len(clean_paths))
    print("Paths in both:", len(old_paths & clean_paths))

# Transcript overlap is informative, but is not proof of identical audio.
if "text" in old.columns and "text" in clean.columns:
    old_texts = {norm_text(x) for x in old["text"] if norm_text(x)}
    clean_texts = {norm_text(x) for x in clean["text"] if norm_text(x)}
    print("\n--- NORMALIZED TRANSCRIPT OVERLAP ---")
    print("Distinct original transcripts:", len(old_texts))
    print("Distinct cleaned transcripts:", len(clean_texts))
    print("Transcripts appearing in both:", len(old_texts & clean_texts))

print("\nComparison complete. No files were modified.")
