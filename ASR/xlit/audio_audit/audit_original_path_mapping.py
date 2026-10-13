from pathlib import Path
import pandas as pd
import re

root = Path.cwd()
review_file = root / "audio_audit/safe_cleaned/validation_checks/combined_manual_review.csv"
decision_file = root / "audio_audit/safe_cleaned/validation_checks/manual_review_decisions.csv"
out_file = root / "audio_audit/safe_cleaned/validation_checks/original_path_mapping_audit.csv"

review = pd.read_csv(review_file).fillna("")
decisions = pd.read_csv(decision_file).fillna("")

manifests = {}
for split in ["train", "validation"]:
    p = root / "combined_dataset" / f"{split}.csv"
    if p.exists():
        df = pd.read_csv(p).fillna("")
        df["audio_path"] = df["audio_path"].astype(str)
        df["text"] = df["text"].astype(str)
        df["_basename"] = df["audio_path"].apply(
            lambda x: Path(x).name.lower()
        )
        manifests[split] = df

decision_map = decisions.set_index("cleaned_path").to_dict("index")
rows = []

for _, r in review.iterrows():
    cleaned = str(r["path"])
    split = str(r["split"])
    text = str(r["text"]).strip()
    basename = Path(cleaned).name.lower()
    # Cleaned filenames commonly have a numeric prefix, e.g.
    # 000737_sample_053.wav -> sample_053.wav
    stripped = re.sub(r"^\d+_", "", basename)

    df = manifests.get(split)
    candidates = pd.DataFrame()
    method = "unresolved"

    if df is not None:
        candidates = df[df["_basename"] == stripped]
        if len(candidates) == 1:
            method = "filename"
        else:
            candidates = df[df["_basename"] == basename]
            if len(candidates) == 1:
                method = "exact_filename"
            else:
                candidates = df[df["text"].str.strip() == text]
                if len(candidates) == 1:
                    method = "unique_transcript"
                else:
                    method = "ambiguous" if len(candidates) > 1 else "no_match"

    original = str(candidates.iloc[0]["audio_path"]) if len(candidates) == 1 else ""
    saved = decision_map.get(cleaned, {})
    rows.append({
        "cleaned_path": cleaned,
        "split": split,
        "text": text,
        "decision": saved.get("decision", ""),
        "original_path_saved": saved.get("original_path", ""),
        "original_path_candidate": original,
        "mapping_method": method,
        "candidate_count": len(candidates),
        "candidate_exists": bool(original and Path(original).is_file()),
    })

out = pd.DataFrame(rows)
out.to_csv(out_file, index=False)

print("\n========== ORIGINAL PATH MAPPING AUDIT ==========")
print("Reviewed clips:", len(out))
print("\nMapping methods:")
print(out["mapping_method"].value_counts().to_string())
print("\nCandidate paths that exist:",
      int(out["candidate_exists"].sum()))
print("\nUnresolved or ambiguous:")
print(int((out["original_path_candidate"] == "").sum()))
print("\nSaved:", out_file)
