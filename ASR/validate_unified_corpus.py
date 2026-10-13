"""
Scans unified_ground_truth.csv and surfaces the transliterated (orig_script=
bengali) rows most likely to be broken, so a native speaker reviews a
targeted ~30-50 rows instead of randomly sampled ones.

Flags, per row:
  - residual_frac: fraction of characters that are NOT Meetei Mayek and not
    whitespace (leftover digits, Latin, untransliterated Bengali, punctuation
    the tool didn't recognize, etc.) -- should be ~0 if transliteration fully
    succeeded.
  - len_ratio: output character count / input Bengali character count. Real
    Bengali->Meetei Mayek transliteration is roughly 1:1 to 1:1.3 in length
    for this script pair; rows far outside that band are suspicious.
  - empty: transliteration produced nothing or near-nothing from non-trivial
    input.

Run from the manipuri_stt root:
    (venv) PS ...\\manipuri_stt> python ASR\\validate_unified_corpus.py

Writes flagged_for_review.csv (open in VS Code / Notepad++, UTF-8) with the
worst rows first. Also re-reads the ORIGINAL Bengali text from
manipuri_benchmark_extracted/labels.csv so you can see input vs output
side by side without hunting for it.
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
UNIFIED_CSV = ROOT / "unified_ground_truth.csv"
ORIGINAL_BENGALI_CSV = ROOT / "ASR" / "manipuri_benchmark_extracted" / "labels.csv"

# Reasonable band for Bengali -> Meetei Mayek character-count ratio.
# Adjust if your own spot-checks show typical ratios outside this.
LEN_RATIO_LOW, LEN_RATIO_HIGH = 0.5, 2.0


def is_meetei_mayek(ch):
    if ch.isspace():
        return True
    return "MEETEI MAYEK" in unicodedata.name(ch, "")


def residual_fraction(text):
    if not text:
        return 1.0
    non_mm = sum(1 for ch in text if not is_meetei_mayek(ch))
    return non_mm / len(text)


def find_file_col(cols):
    for c in ("file", "file_name", "filename"):
        if c in cols:
            return c
    return cols[0]


def find_text_col(cols):
    for c in ("text", "transcript", "transcription"):
        if c in cols:
            return c
    return cols[-1]


def main():
    unified = pd.read_csv(UNIFIED_CSV, encoding="utf-8-sig")
    bn = pd.read_csv(ORIGINAL_BENGALI_CSV, encoding="utf-8-sig")
    bn_file_col = find_file_col(bn.columns.tolist())
    bn_text_col = find_text_col(bn.columns.tolist())
    bn_lookup = dict(zip(bn[bn_file_col].astype(str), bn[bn_text_col].astype(str)))

    translit = unified[unified["orig_script"] == "bengali"].copy()
    print(f"Checking {len(translit)} transliterated rows...")

    records = []
    for _, row in translit.iterrows():
        fname = Path(row["audio_path"]).name
        original_bn = bn_lookup.get(fname, "")
        out_text = str(row["text"])

        r_frac = residual_fraction(out_text)
        len_ratio = (len(out_text) / len(original_bn)) if original_bn else float("nan")
        is_empty = len(out_text.strip()) == 0 and len(original_bn.strip()) > 0
        out_of_band = (
            not pd.isna(len_ratio) and not (LEN_RATIO_LOW <= len_ratio <= LEN_RATIO_HIGH)
        )

        # A simple composite risk score -- doesn't need to be precise, just
        # good enough to sort worst-first.
        risk = r_frac * 3 + (1.0 if is_empty else 0.0) + (0.5 if out_of_band else 0.0)

        records.append({
            "audio_file": fname,
            "original_bengali": original_bn,
            "transliterated": out_text,
            "residual_frac": round(r_frac, 3),
            "len_ratio": round(len_ratio, 2) if not pd.isna(len_ratio) else None,
            "empty_output": is_empty,
            "risk_score": round(risk, 3),
        })

    report = pd.DataFrame(records).sort_values("risk_score", ascending=False)

    n_clean = (report["risk_score"] < 0.05).sum()
    n_flagged = len(report) - n_clean

    print(f"\nClean (residual~0, normal length ratio): {n_clean} / {len(report)}")
    print(f"Flagged for review: {n_flagged}")
    print(f"Empty outputs from non-empty input: {report['empty_output'].sum()}")

    out_path = ROOT / "flagged_for_review.csv"
    report.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nFull sorted report (worst first) written to: {out_path}")
    print("Open in VS Code / Notepad++ (UTF-8). Review the top ~30-50 rows --")
    print("that's a native-speaker judgment call, not something I can check.")

    # Print just the top 10 filenames + risk scores (ASCII-safe) so there's
    # something immediately visible without opening the file.
    print("\nTop 10 riskiest rows (see the CSV for the actual text):")
    for _, r in report.head(10).iterrows():
        print(f"  {r['audio_file']}: residual_frac={r['residual_frac']}, "
              f"len_ratio={r['len_ratio']}, risk={r['risk_score']}")


if __name__ == "__main__":
    main()