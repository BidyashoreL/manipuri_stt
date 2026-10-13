#!/usr/bin/env python3
"""
Phase 2 — Manipuri ASR Character Confusion & Error Analysis

Reads the paired B2/B3 validation predictions produced by
bootstrap_b2_vs_b3.py and performs character-level Levenshtein
alignment on the 864 validation utterances.

Outputs:
  1. phase2_error_report.txt
  2. character_substitutions.csv
  3. character_deletions.csv
  4. character_insertions.csv
  5. b3_utterance_error_summary.csv
  6. tone_diacritic_analysis.csv
  7. b3_alignment_operations.csv

The script is deliberately independent of the model/checkpoint code.
It only analyzes the already-generated paired_predictions.csv.

Expected input columns:
  id, reference, b2_prediction, b3_prediction,
  b2_cer, b3_cer, b2_wer, b3_wer

Usage from the ASR/manipuri_asr_v2 directory:
  python .\phase2_character_error_analysis.py

Optional:
  python .\phase2_character_error_analysis.py --csv "PATH\\paired_predictions.csv"
  python .\phase2_character_error_analysis.py --top 20

Notes:
- Alignment is character-level.
- Whitespace is retained by default because the existing CER was computed
  on normalized strings. For the diagnostic report, whitespace operations
  are separated so they do not dominate the character confusion table.
- Unicode is handled as Python Unicode code points.
- NFC normalization is used consistently for alignment.
- "Tone/diacritic" analysis is heuristic: it identifies Unicode combining
  marks and a configurable set of tone/diacritic-like characters. It does
  NOT establish a linguistic claim about Manipuri tone without a validated
  phonological mapping.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any

import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_CSV = Path(
    r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt\ASR\manipuri_asr_v2\bootstrap_b2_vs_b3"
    r"\paired_predictions.csv"
)

DEFAULT_OUT = DEFAULT_CSV.parent / "phase2_error_analysis"

EXPECTED_COLUMNS = [
    "id",
    "reference",
    "b2_prediction",
    "b3_prediction",
    "b2_cer",
    "b3_cer",
    "b2_wer",
    "b3_wer",
]

# Common punctuation/spacing characters are not useful as "phoneme" confusion
# candidates. We keep them in the alignment but report them separately.
SPACE_CHARS = {" ", "\t", "\n", "\r", "\u00a0", "\u200b"}

# Unicode combining marks. These are the strongest general-purpose signal for
# diacritic-like errors.
COMBINING_CATEGORIES = {"Mn", "Mc", "Me"}

# A small explicit set of common Latin/Indic tone/diacritic marks. This is
# diagnostic only; the report will clearly distinguish this heuristic from a
# validated Manipuri phonological inventory.
EXPLICIT_DIACRITIC_CHARS = set(
    """
    ̀ ́ ̂ ̃ ̄ ̆ ̇ ̈ ̉ ̊ ̌ ̍ ̎ ̐ ̑ ̒ ̓ ̔ ̕ ̚ ̛ ̽
    ॑ ॒ ॓ ॔
    ং ঃ ঁ
    ે ૈ
    ் ஂ ஃ
    """
)


# ---------------------------------------------------------------------------
# Unicode / text helpers
# ---------------------------------------------------------------------------

def nfc(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    return unicodedata.normalize("NFC", str(text))


def is_space(ch: str) -> bool:
    return ch in SPACE_CHARS or ch.isspace()


def is_combining_or_diacritic(ch: str) -> bool:
    return unicodedata.category(ch) in COMBINING_CATEGORIES or ch in EXPLICIT_DIACRITIC_CHARS


def char_display(ch: str) -> str:
    """Make invisible characters readable in CSV/TXT reports."""
    if ch == "":
        return "∅"
    if ch in SPACE_CHARS or ch.isspace():
        return "<SPACE>"
    return ch


def codepoints(text: str) -> str:
    if not text:
        return ""
    return " ".join(f"U+{ord(ch):04X}" for ch in text)


def script_name(ch: str) -> str:
    if not ch:
        return ""
    try:
        return unicodedata.name(ch, "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def is_likely_diacritic(ch: str) -> bool:
    return bool(ch) and is_combining_or_diacritic(ch)


# ---------------------------------------------------------------------------
# Levenshtein alignment
# ---------------------------------------------------------------------------

# Operation tuple:
#   ("equal" | "sub" | "del" | "ins", ref_char, pred_char)
#
# Dynamic-programming backtrace uses deterministic tie-breaking:
#   equal > sub > del > ins
#
# This makes the report reproducible.
def align_chars(reference: str, prediction: str) -> List[Tuple[str, str, str]]:
    ref = list(nfc(reference))
    hyp = list(nfc(prediction))

    n, m = len(ref), len(hyp)

    # Rows of integer edit distances.
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ri = ref[i - 1]
        row = dp[i]
        prev = dp[i - 1]
        for j in range(1, m + 1):
            hj = hyp[j - 1]
            cost = 0 if ri == hj else 1
            row[j] = min(
                prev[j] + 1,       # deletion
                row[j - 1] + 1,    # insertion
                prev[j - 1] + cost # equal/substitution
            )

    ops: List[Tuple[str, str, str]] = []
    i, j = n, m

    while i > 0 or j > 0:
        # Prefer diagonal equal.
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append(("equal", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
            continue

        # Then diagonal substitution.
        if (
            i > 0
            and j > 0
            and dp[i][j] == dp[i - 1][j - 1] + 1
        ):
            ops.append(("sub", ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
            continue

        # Then deletion.
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], ""))
            i -= 1
            continue

        # Finally insertion.
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", "", hyp[j - 1]))
            j -= 1
            continue

        raise RuntimeError("Alignment backtrace failed.")

    ops.reverse()
    return ops


# ---------------------------------------------------------------------------
# Aggregate counters
# ---------------------------------------------------------------------------

def add_operation(
    counters: Dict[str, Counter],
    op: str,
    ref_char: str,
    pred_char: str,
):
    if op == "sub":
        counters["substitutions"][(ref_char, pred_char)] += 1
    elif op == "del":
        counters["deletions"][ref_char] += 1
    elif op == "ins":
        counters["insertions"][pred_char] += 1


def normalized_error_rate_from_ops(ops: List[Tuple[str, str, str]]) -> float:
    ref_len = sum(1 for op, _, _ in ops if op in {"equal", "sub", "del"})
    errors = sum(1 for op, _, _ in ops if op != "equal")
    return errors / ref_len if ref_len else 0.0


# ---------------------------------------------------------------------------
# Tone / diacritic diagnostics
# ---------------------------------------------------------------------------

def tone_flags_for_sub(ref_char: str, pred_char: str) -> Tuple[bool, str]:
    ref_d = is_likely_diacritic(ref_char)
    pred_d = is_likely_diacritic(pred_char)

    if ref_d and pred_d:
        return True, "diacritic_to_diacritic"
    if ref_d and not pred_d:
        return True, "diacritic_deleted_as_substitution"
    if not ref_d and pred_d:
        return True, "diacritic_inserted_as_substitution"
    return False, ""


def tone_flags_for_del(ref_char: str) -> Tuple[bool, str]:
    if is_likely_diacritic(ref_char):
        return True, "reference_diacritic_deletion"
    return False, ""


def tone_flags_for_ins(pred_char: str) -> Tuple[bool, str]:
    if is_likely_diacritic(pred_char):
        return True, "predicted_diacritic_insertion"
    return False, ""


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(df: pd.DataFrame, top_n: int, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    substitutions = Counter()
    deletions = Counter()
    insertions = Counter()

    # B3-specific operation counts.
    tone_subs = Counter()
    tone_dels = Counter()
    tone_ins = Counter()

    operation_rows = []

    total_ref_chars = 0
    total_errors = 0
    total_sub = 0
    total_del = 0
    total_ins = 0
    total_tone_related = 0

    b3_better = 0
    b3_worse = 0
    ties = 0

    for idx, row in df.iterrows():
        sample_id = str(row.get("id", idx))
        reference = nfc(row.get("reference", ""))
        prediction = nfc(row.get("b3_prediction", ""))

        ops = align_chars(reference, prediction)

        ref_chars = len(reference)
        errors = sum(1 for op, _, _ in ops if op != "equal")
        subs = sum(1 for op, _, _ in ops if op == "sub")
        dels = sum(1 for op, _, _ in ops if op == "del")
        ins = sum(1 for op, _, _ in ops if op == "ins")

        total_ref_chars += ref_chars
        total_errors += errors
        total_sub += subs
        total_del += dels
        total_ins += ins

        tone_count = 0

        for op, rc, pc in ops:
            if op == "sub":
                substitutions[(rc, pc)] += 1
                flagged, reason = tone_flags_for_sub(rc, pc)
                if flagged:
                    tone_count += 1
                    tone_subs[reason] += 1

            elif op == "del":
                deletions[rc] += 1
                flagged, reason = tone_flags_for_del(rc)
                if flagged:
                    tone_count += 1
                    tone_dels[reason] += 1

            elif op == "ins":
                insertions[pc] += 1
                flagged, reason = tone_flags_for_ins(pc)
                if flagged:
                    tone_count += 1
                    tone_ins[reason] += 1

            if op != "equal":
                operation_rows.append({
                    "id": sample_id,
                    "operation": op,
                    "reference_char": char_display(rc),
                    "prediction_char": char_display(pc),
                    "reference_codepoint": codepoints(rc),
                    "prediction_codepoint": codepoints(pc),
                    "reference_unicode_name": script_name(rc),
                    "prediction_unicode_name": script_name(pc),
                    "tone_or_diacritic_flag": (
                        bool(
                            (op == "sub" and tone_flags_for_sub(rc, pc)[0])
                            or (op == "del" and tone_flags_for_del(rc)[0])
                            or (op == "ins" and tone_flags_for_ins(pc)[0])
                        )
                    ),
                })

        total_tone_related += tone_count

        b3_cer = float(row["b3_cer"]) if pd.notna(row.get("b3_cer")) else math.nan
        b2_cer = float(row["b2_cer"]) if pd.notna(row.get("b2_cer")) else math.nan

        if not math.isnan(b2_cer) and not math.isnan(b3_cer):
            if b3_cer < b2_cer:
                b3_better += 1
            elif b3_cer > b2_cer:
                b3_worse += 1
            else:
                ties += 1

        rows.append({
            "id": sample_id,
            "reference": reference,
            "b3_prediction": prediction,
            "reference_chars": ref_chars,
            "b3_edit_errors": errors,
            "b3_substitutions": subs,
            "b3_deletions": dels,
            "b3_insertions": ins,
            "b3_alignment_cer": errors / ref_chars if ref_chars else 0.0,
            "b2_cer_from_bootstrap": b2_cer,
            "b3_cer_from_bootstrap": b3_cer,
            "cer_change_b3_minus_b2": b3_cer - b2_cer
            if not math.isnan(b2_cer) and not math.isnan(b3_cer)
            else math.nan,
            "tone_diacritic_related_errors": tone_count,
            "tone_share_of_errors": tone_count / errors if errors else 0.0,
        })

    # -----------------------------------------------------------------------
    # Write ranked tables
    # -----------------------------------------------------------------------

    sub_rows = []
    for (ref, pred), count in substitutions.most_common():
        sub_rows.append({
            "reference_char": char_display(ref),
            "prediction_char": char_display(pred),
            "count": count,
            "reference_codepoint": codepoints(ref),
            "prediction_codepoint": codepoints(pred),
            "reference_unicode_name": script_name(ref),
            "prediction_unicode_name": script_name(pred),
            "tone_or_diacritic_related": tone_flags_for_sub(ref, pred)[0],
            "tone_reason": tone_flags_for_sub(ref, pred)[1],
        })

    del_rows = []
    for ch, count in deletions.most_common():
        del_rows.append({
            "reference_char": char_display(ch),
            "count": count,
            "reference_codepoint": codepoints(ch),
            "reference_unicode_name": script_name(ch),
            "tone_or_diacritic_related": is_likely_diacritic(ch),
        })

    ins_rows = []
    for ch, count in insertions.most_common():
        ins_rows.append({
            "prediction_char": char_display(ch),
            "count": count,
            "prediction_codepoint": codepoints(ch),
            "prediction_unicode_name": script_name(ch),
            "tone_or_diacritic_related": is_likely_diacritic(ch),
        })

    pd.DataFrame(sub_rows).to_csv(
        out_dir / "character_substitutions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(del_rows).to_csv(
        out_dir / "character_deletions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(ins_rows).to_csv(
        out_dir / "character_insertions.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(rows).to_csv(
        out_dir / "b3_utterance_error_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(operation_rows).to_csv(
        out_dir / "b3_alignment_operations.csv", index=False, encoding="utf-8-sig"
    )

    # Tone summary.
    total_sub_del_ins = total_sub + total_del + total_ins
    tone_rows = [
        {
            "category": "tone_or_diacritic_related",
            "count": total_tone_related,
            "share_of_all_B3_alignment_errors": (
                total_tone_related / total_sub_del_ins
                if total_sub_del_ins else 0.0
            ),
        },
        {
            "category": "all_substitutions",
            "count": total_sub,
            "share_of_all_B3_alignment_errors": (
                total_sub / total_sub_del_ins if total_sub_del_ins else 0.0
            ),
        },
        {
            "category": "all_deletions",
            "count": total_del,
            "share_of_all_B3_alignment_errors": (
                total_del / total_sub_del_ins if total_sub_del_ins else 0.0
            ),
        },
        {
            "category": "all_insertions",
            "count": total_ins,
            "share_of_all_B3_alignment_errors": (
                total_ins / total_sub_del_ins if total_sub_del_ins else 0.0
            ),
        },
    ]

    for reason, count in tone_subs.items():
        tone_rows.append({
            "category": reason,
            "count": count,
            "share_of_all_B3_alignment_errors": (
                count / total_sub_del_ins if total_sub_del_ins else 0.0
            ),
        })

    for reason, count in tone_dels.items():
        tone_rows.append({
            "category": reason,
            "count": count,
            "share_of_all_B3_alignment_errors": (
                count / total_sub_del_ins if total_sub_del_ins else 0.0
            ),
        })

    for reason, count in tone_ins.items():
        tone_rows.append({
            "category": reason,
            "count": count,
            "share_of_all_B3_alignment_errors": (
                count / total_sub_del_ins if total_sub_del_ins else 0.0
            ),
        })

    pd.DataFrame(tone_rows).to_csv(
        out_dir / "tone_diacritic_analysis.csv", index=False, encoding="utf-8-sig"
    )

    # -----------------------------------------------------------------------
    # Text report
    # -----------------------------------------------------------------------

    n = len(df)
    micro_cer = total_errors / total_ref_chars if total_ref_chars else 0.0

    lines = []
    lines.append("=" * 78)
    lines.append("MANIPURI ASR — PHASE 2 CHARACTER CONFUSION & ERROR ANALYSIS")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Validation utterances analyzed : {n}")
    lines.append("Model analyzed                 : B3-NARROW Epoch 11")
    lines.append("Alignment                      : character-level Levenshtein")
    lines.append("Unicode normalization          : NFC")
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append(
        "Tone/diacritic classification is a Unicode-based diagnostic heuristic."
    )
    lines.append(
        "It does not by itself prove that a particular error is a Manipuri tonal error."
    )
    lines.append("")

    lines.append("-" * 78)
    lines.append("MICRO-AVERAGED ALIGNMENT SUMMARY")
    lines.append("-" * 78)
    lines.append(f"Reference characters           : {total_ref_chars:,}")
    lines.append(f"Total B3 edit errors            : {total_errors:,}")
    lines.append(f"Substitutions                   : {total_sub:,}")
    lines.append(f"Deletions                       : {total_del:,}")
    lines.append(f"Insertions                      : {total_ins:,}")
    lines.append(f"Micro CER from alignment        : {micro_cer * 100:.4f}%")
    lines.append(
        f"Tone/diacritic-flagged errors  : {total_tone_related:,} "
        f"({(total_tone_related / total_sub_del_ins * 100 if total_sub_del_ins else 0):.2f}%)"
    )
    lines.append("")

    lines.append("-" * 78)
    lines.append(f"TOP {top_n} CHARACTER SUBSTITUTIONS")
    lines.append("-" * 78)

    for rank, ((ref, pred), count) in enumerate(
        substitutions.most_common(top_n), start=1
    ):
        tone, reason = tone_flags_for_sub(ref, pred)
        suffix = f"  [{reason}]" if tone else ""
        lines.append(
            f"{rank:2d}. {char_display(ref)} -> {char_display(pred)}"
            f"  count={count:5d}  "
            f"{script_name(ref)} -> {script_name(pred)}{suffix}"
        )
    lines.append("")

    lines.append("-" * 78)
    lines.append(f"TOP {top_n} DELETIONS")
    lines.append("-" * 78)

    for rank, (ch, count) in enumerate(deletions.most_common(top_n), start=1):
        suffix = "  [DIACRITIC FLAG]" if is_likely_diacritic(ch) else ""
        lines.append(
            f"{rank:2d}. {char_display(ch)}"
            f"  count={count:5d}  {script_name(ch)}{suffix}"
        )
    lines.append("")

    lines.append("-" * 78)
    lines.append(f"TOP {top_n} INSERTIONS")
    lines.append("-" * 78)

    for rank, (ch, count) in enumerate(insertions.most_common(top_n), start=1):
        suffix = "  [DIACRITIC FLAG]" if is_likely_diacritic(ch) else ""
        lines.append(
            f"{rank:2d}. {char_display(ch)}"
            f"  count={count:5d}  {script_name(ch)}{suffix}"
        )
    lines.append("")

    lines.append("-" * 78)
    lines.append("TONE / DIACRITIC DIAGNOSTIC")
    lines.append("-" * 78)
    lines.append(
        f"Flagged substitution errors      : {sum(tone_subs.values()):,}"
    )
    lines.append(
        f"Flagged deletion errors          : {sum(tone_dels.values()):,}"
    )
    lines.append(
        f"Flagged insertion errors         : {sum(tone_ins.values()):,}"
    )
    lines.append(
        f"Total flagged errors             : {total_tone_related:,}"
    )
    if total_sub_del_ins:
        lines.append(
            f"Share of all B3 edit errors     : "
            f"{total_tone_related / total_sub_del_ins * 100:.2f}%"
        )
    lines.append("")
    lines.append(
        "Interpretation rule: a high flagged share is evidence that Unicode "
        "diacritic/combining-mark errors deserve inspection; it is not by itself "
        "evidence of a phonological tone error."
    )
    lines.append("")

    lines.append("-" * 78)
    lines.append("PAIRED B2 vs B3 UTTERANCE COUNTS")
    lines.append("-" * 78)
    lines.append(f"B3 lower CER than B2           : {b3_better:,}")
    lines.append(f"B3 higher CER than B2          : {b3_worse:,}")
    lines.append(f"Equal utterance CER             : {ties:,}")
    lines.append("")

    lines.append("-" * 78)
    lines.append("OUTPUT FILES")
    lines.append("-" * 78)
    for filename in [
        "phase2_error_report.txt",
        "character_substitutions.csv",
        "character_deletions.csv",
        "character_insertions.csv",
        "b3_utterance_error_summary.csv",
        "tone_diacritic_analysis.csv",
        "b3_alignment_operations.csv",
    ]:
        lines.append(str(out_dir / filename))

    report = "\n".join(lines)
    (out_dir / "phase2_error_report.txt").write_text(report, encoding="utf-8")

    summary = {
        "validation_utterances": n,
        "model": "B3-NARROW Epoch 11",
        "normalization": "NFC",
        "alignment": "character-level Levenshtein",
        "reference_characters": total_ref_chars,
        "edit_errors": total_errors,
        "substitutions": total_sub,
        "deletions": total_del,
        "insertions": total_ins,
        "micro_cer": micro_cer,
        "tone_diacritic_flagged_errors": total_tone_related,
        "tone_diacritic_share_of_errors": (
            total_tone_related / total_sub_del_ins
            if total_sub_del_ins else 0.0
        ),
        "b3_lower_cer_utterances": b3_better,
        "b3_higher_cer_utterances": b3_worse,
        "equal_cer_utterances": ties,
        "note": (
            "Tone/diacritic analysis is a Unicode heuristic, not a validated "
            "Manipuri phonological tone classifier."
        ),
    }

    (out_dir / "phase2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "report": report,
        "summary": summary,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "output_dir": out_dir,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 2 B3-NARROW character confusion and error analysis."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to paired_predictions.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to <csv-dir>/phase2_error_analysis",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of ranked errors to print (default: 20)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = args.csv.expanduser().resolve()
    out_dir = (
        args.out.expanduser().resolve()
        if args.out
        else csv_path.parent / "phase2_error_analysis"
    )

    print("=" * 78)
    print("MANIPURI ASR — PHASE 2 ERROR ANALYSIS")
    print("=" * 78)
    print(f"[INPUT] {csv_path}")
    print(f"[OUTPUT] {out_dir}")
    print()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"paired_predictions.csv not found:\n{csv_path}\n\n"
            "Use --csv to provide the correct path."
        )

    df = pd.read_csv(csv_path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + f"\n\nFound columns:\n{df.columns.tolist()}"
        )

    if len(df) != 864:
        print(
            f"[WARNING] Expected 864 validation rows, found {len(df)}. "
            "The script will continue, but verify that this is the intended "
            "validation set."
        )

    # Normalize text columns before analysis.
    for col in ["reference", "b2_prediction", "b3_prediction"]:
        df[col] = df[col].map(nfc)

    print(f"[DATA] Rows: {len(df)}")
    print("[DATA] Required columns: PASS")
    print("[ALIGN] NFC + character-level Levenshtein")
    print()

    result = analyze(df, args.top, out_dir)

    print(result["report"])
    print()
    print("=" * 78)
    print("PHASE 2 COMPLETE")
    print("=" * 78)
    print(f"Report: {out_dir / 'phase2_error_report.txt'}")
    print(f"Summary: {out_dir / 'phase2_summary.json'}")
    print(f"Substitutions: {out_dir / 'character_substitutions.csv'}")
    print(f"Deletions: {out_dir / 'character_deletions.csv'}")
    print(f"Insertions: {out_dir / 'character_insertions.csv'}")
    print(f"Tone analysis: {out_dir / 'tone_diacritic_analysis.csv'}")
    print(f"All operations: {out_dir / 'b3_alignment_operations.csv'}")


if __name__ == "__main__":
    main()
