from pathlib import Path
import pandas as pd

p = Path(r"..\combined_dataset\val.csv").resolve()
print("File:", p)
df = pd.read_csv(p, encoding="utf-8-sig")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))
