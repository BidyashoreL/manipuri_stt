import pyarrow.parquet as pq
from pathlib import Path

folder = Path(
    r"E:\hf_cache\hub\datasets--SPRINGLab--IndicTTS_Manipuri"
    r"\datasets--SPRINGLab--IndicTTS_Manipuri\snapshots"
    r"\a7472c5ec603e22a9ab1f1d52aa44e2b23b24d14\data"
)

for path in sorted(folder.glob("train-*.parquet")):
    pf = pq.ParquetFile(path)
    print(f"{path.name}: {pf.metadata.num_rows:,} rows")
    print("Columns:", pf.schema.names)
    print("Audio schema:", pf.schema.field("audio") if "audio" in pf.schema.names else "Audio field not found")
