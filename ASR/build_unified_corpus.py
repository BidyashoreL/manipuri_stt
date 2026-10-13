"""
Builds ONE unified ground-truth CSV, in Meetei Mayek throughout, from all
three data pools:

  - ASR/manipuri_1k_project/labels.csv           (native Meetei Mayek, 1000 rows)
  - benchmark/ground_truth.csv                    (native Meetei Mayek, 100 rows)
  - ASR/manipuri_benchmark_extracted/labels.csv   (Bengali script, 10907 rows)
                                                   -> transliterated to Meetei Mayek

Requires: https://github.com/hoomexsun/xlit cloned somewhere on disk first:
    git clone https://github.com/hoomexsun/xlit.git
Only needs `tqdm` and `numpy` installed for the transliteration path used here
(pyenchant/matplotlib are for xlit's glyph-correction / plotting features,
not needed for this script).

Run from the manipuri_stt root:
    (venv) PS ...\\manipuri_stt> python ASR\\build_unified_corpus.py

Writes: unified_ground_truth.csv  (columns: audio_path, text, source, orig_script)
Does NOT modify or move any original files.
"""

import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# EDIT THESE to match your machine
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent  # .../manipuri_stt
XLIT_REPO_PATH = ROOT / "ASR" / "xlit"          # where you `git clone`d hoomexsun/xlit

POOLS = [
    # (csv_path,                                          audio_dir,          source_tag,     needs_transliteration)
    (ROOT / "ASR" / "manipuri_1k_project" / "labels.csv",       ROOT / "ASR" / "manipuri_1k_project",       "1k_project",   False),
    (ROOT / "benchmark" / "ground_truth.csv",                    ROOT / "benchmark",                          "benchmark100", False),
    (ROOT / "ASR" / "manipuri_benchmark_extracted" / "labels.csv", ROOT / "ASR" / "manipuri_benchmark_extracted", "extracted_bn", True),
]

OUTPUT_CSV = ROOT / "unified_ground_truth.csv"

# ---------------------------------------------------------------------------

sys.path.insert(0, str(XLIT_REPO_PATH))
from src.mt_ import MTransliteration  # noqa: E402

mt = MTransliteration()

# Punctuation normalization: xlit's transliteration drops punctuation, so to
# avoid the model learning "punctuation present == native pool" as a spurious
# signal, strip common punctuation from ALL pools uniformly, not just the
# transliterated one. Adjust this set after checking dataset_audit_report.txt
# for what punctuation actually appears in your native-script rows.
PUNCT_RE = re.compile(r"[।,.!?;:\"'()\[\]—–\-]")


def normalize_punct(text):
    return re.sub(r"\s+", " ", PUNCT_RE.sub(" ", text)).strip()


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


rows_out = []
skipped_missing_audio = 0

for csv_path, audio_dir, source_tag, needs_xlit in POOLS:
    print(f"\n=== {source_tag}: {csv_path} ===")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    file_col = find_file_col(df.columns.tolist())
    text_col = find_text_col(df.columns.tolist())
    print(f"  {len(df)} rows, file_col='{file_col}', text_col='{text_col}', transliterate={needs_xlit}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc=source_tag):
        fname = str(row[file_col]).strip()
        raw_text = str(row[text_col]).strip()
        audio_path = audio_dir / fname

        if not audio_path.exists():
            skipped_missing_audio += 1
            continue

        if needs_xlit:
            try:
                mm_text = mt.transliterate(raw_text)
            except Exception as e:
                print(f"  transliteration failed for {fname}: {e}")
                continue
            orig_script = "bengali"
        else:
            mm_text = raw_text
            orig_script = "meetei_mayek"

        mm_text = normalize_punct(mm_text)
        if not mm_text:
            continue

        rows_out.append({
            "audio_path": str(audio_path),   # FULL path -- filenames collide across pools, bare names are not safe
            "text": mm_text,
            "source": source_tag,
            "orig_script": orig_script,
        })

unified = pd.DataFrame(rows_out)
unified.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print("\n" + "=" * 60)
print(f"Wrote {len(unified)} rows to {OUTPUT_CSV}")
print(f"Skipped (audio file not found): {skipped_missing_audio}")
print("\nBy source:")
print(unified["source"].value_counts().to_string())
print("\nSpot-check up to 5 random transliterated rows (orig_script=bengali) --")
print("PLEASE eyeball these yourself, you're a native speaker and I'm not:")
bengali_rows = unified[unified["orig_script"] == "bengali"]
sample = bengali_rows.sample(min(5, len(bengali_rows))) if len(bengali_rows) else bengali_rows
for _, r in sample.iterrows():
    print(f"  {Path(r['audio_path']).name}: {r['text']}")