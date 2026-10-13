from pathlib import Path
import pandas as pd

root = Path.cwd()
outdir = root / "audio_audit" / "final_candidate"
outdir.mkdir(parents=True, exist_ok=True)

mapping_file = root / "audio_audit/safe_cleaned/validation_checks/original_path_mapping_audit.csv"
review_file = root / "audio_audit/safe_cleaned/validation_checks/manual_review_decisions.csv"
review_details_file = root / "audio_audit/safe_cleaned/validation_checks/combined_manual_review.csv"

mapping = pd.read_csv(mapping_file).fillna("")
decisions = pd.read_csv(review_file).fillna("")
review_details = pd.read_csv(review_details_file).fillna("")

reason_by_path = (
    review_details.drop_duplicates(subset=["path"])
    .set_index("path")["review_reason"]
    .to_dict()
)

# Use the verified filename mapping, not transcript-only guesses.
if mapping["candidate_exists"].astype(str).str.lower().isin(["false", "0"]).any():
    raise RuntimeError("Some original paths failed validation. Stopping.")

if mapping["original_path_candidate"].astype(str).str.strip().eq("").any():
    raise RuntimeError("Some original paths are unresolved. Stopping.")

if mapping["mapping_method"].astype(str).ne("filename").any():
    raise RuntimeError("Unexpected mapping method. Inspect the mapping report first.")

# Verify that the saved decisions and audit mappings refer to the same reviewed clips.
if set(mapping["cleaned_path"]) != set(decisions["cleaned_path"]):
    raise RuntimeError("Review and mapping files contain different clip sets.")

decision_by_cleaned = decisions.set_index("cleaned_path").to_dict("index")
mapping_by_cleaned = mapping.set_index("cleaned_path").to_dict("index")

for split in ["train", "validation"]:
    source = root / "combined_dataset" / f"{split}.csv"
    df = pd.read_csv(source).fillna("")
    df["audio_path"] = df["audio_path"].astype(str)
    df["text"] = df["text"].astype(str)

    output_rows = []
    quarantine_rows = []
    replacement_count = 0

    for _, row in df.iterrows():
        original_path = row["audio_path"]
        cleaned_matches = mapping[
            (mapping["split"].astype(str) == split) &
            (mapping["original_path_candidate"].astype(str) == original_path)
        ]

        if len(cleaned_matches) == 0:
            # Unreviewed recordings remain unchanged.
            output_rows.append(row.to_dict())
            continue

        if len(cleaned_matches) != 1:
            raise RuntimeError(
                f"Expected one reviewed mapping for {original_path}; "
                f"found {len(cleaned_matches)}"
            )

        m = cleaned_matches.iloc[0]
        decision = str(m["decision"])
        cleaned_path = str(m["cleaned_path"])

        if str(m["text"]).strip() != str(row["text"]).strip():
            raise RuntimeError(
                f"Transcript mismatch for reviewed clip: {original_path}"
            )

        if decision == "Exclude":
            q = row.to_dict()
            q["review_decision"] = decision
            q["cleaned_path"] = cleaned_path
            q["review_reason"] = reason_by_path.get(cleaned_path, "")
            quarantine_rows.append(q)

        elif decision == "Keep cleaned":
            if not Path(cleaned_path).is_file():
                raise RuntimeError(f"Cleaned audio missing: {cleaned_path}")
            new_row = row.to_dict()
            new_row["audio_path"] = cleaned_path
            output_rows.append(new_row)
            replacement_count += 1

        elif decision == "Use original":
            output_rows.append(row.to_dict())

        elif decision == "Needs expert review":
            q = row.to_dict()
            q["review_decision"] = decision
            q["cleaned_path"] = cleaned_path
            q["review_reason"] = reason_by_path.get(cleaned_path, "")
            quarantine_rows.append(q)

        else:
            raise RuntimeError(
                f"Unexpected decision {decision!r} for {original_path}"
            )

    final_df = pd.DataFrame(output_rows, columns=df.columns)
    final_path = outdir / f"{split}.csv"
    final_df.to_csv(final_path, index=False)

    quarantine_df = pd.DataFrame(quarantine_rows)
    quarantine_df.to_csv(outdir / f"{split}_quarantined.csv", index=False)

    # Verify all candidate audio paths and transcript fields.
    missing = [
        p for p in final_df["audio_path"].astype(str)
        if not Path(p).is_file()
    ]
    empty_text = int(final_df["text"].astype(str).str.strip().eq("").sum())

    if missing:
        raise RuntimeError(
            f"{split}: {len(missing)} missing audio paths. See {final_path}"
        )

    print(f"\n{split.upper()}")
    print("Original entries:", len(df))
    print("Candidate entries:", len(final_df))
    print("Cleaned replacements:", replacement_count)
    print("Quarantined entries:", len(quarantine_df))
    print("Missing audio:", len(missing))
    print("Empty transcripts:", empty_text)
    print("Manifest:", final_path)

print("\nTest benchmark was not modified.")
print("Candidate files saved in:", outdir)
