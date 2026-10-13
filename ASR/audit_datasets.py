"""
Run this from the manipuri_stt root:

    (venv) PS ...\\manipuri_stt> python ASR\\audit_datasets.py

Writes a clean UTF-8 report to dataset_audit_report.txt (open it in VS Code /
Notepad++ / anything UTF-8-aware, NOT the raw PowerShell console) and prints
a short ASCII-only summary here so nothing gets mangled on screen.

This does NOT touch or move any of your files. Read-only.
"""

import csv
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # .../manipuri_stt
REPORT_PATH = ROOT / "dataset_audit_report.txt"

CANDIDATE_DIRS = [
    ROOT / "ASR" / "manipuri_1k_project",
    ROOT / "ASR" / "manipuri_benchmark_extracted",
    ROOT / "benchmark",
]

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a"}


def script_of_char(ch):
    if ch.isspace() or unicodedata.category(ch).startswith("P") or ch.isdigit():
        return None
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"
    if "MEETEI MAYEK" in name:
        return "Meetei Mayek"
    if "BENGALI" in name:
        return "Bengali"
    if "LATIN" in name:
        return "Latin"
    if "DEVANAGARI" in name:
        return "Devanagari"
    return name.split()[0]


def classify_text_sample(rows, text_col, n=30):
    counter = Counter()
    for row in rows[:n]:
        txt = row.get(text_col, "") or ""
        for ch in txt:
            s = script_of_char(ch)
            if s:
                counter[s] += 1
    return counter


def find_text_col(fieldnames):
    for cand in ("text", "transcript", "transcription", "sentence"):
        if cand in fieldnames:
            return cand
    return fieldnames[-1] if fieldnames else None


def find_file_col(fieldnames):
    for cand in ("file", "file_name", "filename", "path", "audio_path"):
        if cand in fieldnames:
            return cand
    return fieldnames[0] if fieldnames else None


def audit_csv(csv_path, report_lines):
    report_lines.append(f"\n--- CSV: {csv_path.relative_to(ROOT)} ---")
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except UnicodeDecodeError as e:
        report_lines.append(f"  FAILED to read as UTF-8: {e}")
        return None

    if not rows:
        report_lines.append("  EMPTY (0 data rows)")
        return None

    fieldnames = reader.fieldnames or []
    file_col = find_file_col(fieldnames)
    text_col = find_text_col(fieldnames)

    report_lines.append(f"  columns: {fieldnames}")
    report_lines.append(f"  row count: {len(rows)}")
    report_lines.append(f"  using file column: '{file_col}', text column: '{text_col}'")

    scripts = classify_text_sample(rows, text_col)
    report_lines.append(f"  script distribution (first 30 rows, char counts): {dict(scripts)}")

    # Cross-check: do the referenced audio files actually exist next to this CSV?
    parent = csv_path.parent
    existing, missing = 0, 0
    missing_examples = []
    for row in rows:
        fname = row.get(file_col, "")
        if not fname:
            continue
        candidate = parent / fname
        if candidate.exists():
            existing += 1
        else:
            missing += 1
            if len(missing_examples) < 3:
                missing_examples.append(fname)
    report_lines.append(f"  audio files found next to CSV: {existing} / {existing + missing}")
    if missing_examples:
        report_lines.append(f"  example missing filenames: {missing_examples}")

    report_lines.append("  first 3 rows (raw):")
    for row in rows[:3]:
        report_lines.append(f"    {row}")

    return {
        "path": csv_path,
        "n_rows": len(rows),
        "file_col": file_col,
        "text_col": text_col,
        "scripts": scripts,
        "audio_found": existing,
        "audio_missing": missing,
    }


def main():
    report_lines = [f"Dataset audit for: {ROOT}", "=" * 70]
    summary = []

    for d in CANDIDATE_DIRS:
        report_lines.append(f"\n\n===== DIRECTORY: {d.relative_to(ROOT) if d.exists() else d} =====")
        if not d.exists():
            report_lines.append("  NOT FOUND")
            continue

        audio_files = [p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
        report_lines.append(f"  audio files found (recursive): {len(audio_files)}")
        if audio_files:
            report_lines.append(f"  example filenames: {[p.name for p in audio_files[:3]]}")

        csv_files = list(d.rglob("*.csv"))
        report_lines.append(f"  CSV files found: {[str(p.relative_to(ROOT)) for p in csv_files]}")

        for csv_path in csv_files:
            result = audit_csv(csv_path, report_lines)
            if result:
                summary.append((d.relative_to(ROOT), result))

    # Also scan the whole tree once for a ground-truth total, in case of pools
    # not covered by CANDIDATE_DIRS (e.g. anything extracted from the parquet files)
    all_audio = [p for p in ROOT.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    report_lines.append(f"\n\n===== WHOLE-TREE TOTAL =====")
    report_lines.append(f"Total audio files anywhere under {ROOT}: {len(all_audio)}")

    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")

    # ASCII-only console summary -- safe to print regardless of console codepage
    print("=" * 60)
    print("SUMMARY (full detail + non-ASCII text written to:")
    print(f"  {REPORT_PATH})")
    print("=" * 60)
    for rel_dir, result in summary:
        print(f"{rel_dir} :: {result['path'].name}")
        print(f"  rows={result['n_rows']}  audio_found={result['audio_found']}  audio_missing={result['audio_missing']}")
        script_summary = {k: v for k, v in result["scripts"].items()}
        print(f"  scripts seen: {list(script_summary.keys())}")
        print()
    print(f"Total audio files under project root: {len(all_audio)}")
    print("\nOpen dataset_audit_report.txt in VS Code / Notepad++ (UTF-8) to read")
    print("the actual transcript text safely -- do not `type` it in PowerShell.")


if __name__ == "__main__":
    main()