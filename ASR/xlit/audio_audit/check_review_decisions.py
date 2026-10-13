import pandas as pd
from pathlib import Path

p = Path("audio_audit/safe_cleaned/validation_checks/manual_review_decisions.csv")
d = pd.read_csv(p).fillna("")

print("\n========== MANUAL REVIEW SUMMARY ==========")
print("Total saved decisions:", len(d))
print("\nDecisions:")
print(d["decision"].value_counts(dropna=False).to_string())

print("\nDecisions by split:")
print(pd.crosstab(d["split"], d["decision"], margins=True).to_string())

print("\nMissing original paths:", (d["original_path"].astype(str).str.strip() == "").sum())
print("Empty decisions:", (d["decision"].astype(str).str.strip() == "").sum())
print("\nSaved review file:", p.resolve())
