from pathlib import Path
import pandas as pd
import math
import wave
import json

ROOT = Path(__file__).resolve().parent
ASR = ROOT.parent

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"

OUTPUT_DIR = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_wav(path):
    result = {
        "exists": False,
        "readable": False,
        "frames": None,
        "sample_rate": None,
        "duration": None,
        "error": None,
    }

    if not path:
        result["error"] = "EMPTY_PATH"
        return result

    path = Path(path)

    if not path.exists():
        result["error"] = "FILE_NOT_FOUND"
        return result

    result["exists"] = True

    try:
        with wave.open(str(path), "rb") as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()

            result["frames"] = frames
            result["sample_rate"] = sr
            result["duration"] = frames / sr if sr else None
            result["readable"] = True

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result


def inspect_manifest(csv_path, name):
    print()
    print("=" * 80)
    print(f"CHECKING {name.upper()}")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    results = []

    for idx, row in df.iterrows():

        source = str(row.get("source", "")).strip().lower()

        audio_path = row.get("audio_path", "")
        parquet_file = row.get("parquet_file", "")
        parquet_row = row.get("parquet_row", "")

        record = {
            "manifest": name,
            "row_index": idx,
            "source": source,
            "text": str(row.get("text", "")),
            "audio_path": audio_path,
            "parquet_file": parquet_file,
            "parquet_row": parquet_row,
        }

        # IndicVoices samples are loaded from parquet by the B3 script.
        if source == "indicvoices":
            if (
                pd.isna(parquet_file)
                or str(parquet_file).strip() == ""
                or pd.isna(parquet_row)
            ):
                record["status"] = "BAD_PARQUET_REFERENCE"
            else:
                p = Path(str(parquet_file))

                if not p.exists():
                    record["status"] = "PARQUET_FILE_NOT_FOUND"
                else:
                    record["status"] = "PARQUET_REFERENCE_OK"

            record["exists"] = None
            record["readable"] = None
            record["duration_actual"] = None
            record["error"] = None

        else:
            if pd.isna(audio_path):
                audio_path = ""

            check = check_wav(str(audio_path))

            record["status"] = (
                "OK"
                if check["exists"] and check["readable"]
                else check["error"]
            )

            record["exists"] = check["exists"]
            record["readable"] = check["readable"]
            record["duration_actual"] = check["duration"]
            record["sample_rate"] = check["sample_rate"]
            record["frames"] = check["frames"]
            record["error"] = check["error"]

        results.append(record)

        if (idx + 1) % 500 == 0:
            print(f"Checked {idx + 1}/{len(df)}")

    out = pd.DataFrame(results)

    output_csv = OUTPUT_DIR / f"{name}_audio_diagnostic.csv"
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print()
    print("RESULT SUMMARY")
    print("-" * 80)

    print(out["status"].value_counts(dropna=False).to_string())

    print()
    print(f"Saved:")
    print(output_csv)

    return out


train_result = inspect_manifest(TRAIN_CSV, "train")
val_result = inspect_manifest(VAL_CSV, "val")


summary = {
    "train_rows": len(train_result),
    "val_rows": len(val_result),

    "train_file_not_found": int(
        (train_result["status"] == "FILE_NOT_FOUND").sum()
    ),

    "val_file_not_found": int(
        (val_result["status"] == "FILE_NOT_FOUND").sum()
    ),

    "train_parquet_missing": int(
        train_result["status"].isin(
            ["PARQUET_FILE_NOT_FOUND", "BAD_PARQUET_REFERENCE"]
        ).sum()
    ),

    "val_parquet_missing": int(
        val_result["status"].isin(
            ["PARQUET_FILE_NOT_FOUND", "BAD_PARQUET_REFERENCE"]
        ).sum()
    ),
}


summary_path = OUTPUT_DIR / "missing_audio_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(json.dumps(summary, indent=2, ensure_ascii=False))

print()
print("DONE.")