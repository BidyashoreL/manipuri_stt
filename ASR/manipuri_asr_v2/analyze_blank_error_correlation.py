
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent

frame_file = BASE / "diagnostics" / "ctc_logit_forensics" / "frame_logits_important_samples.csv"
error_file = BASE.parent / "checkpoints" / "v3_ctc" / "b3_narrow" / "diagnostics" / "sample_diagnostics.csv"

frames = pd.read_csv(frame_file)
errors = pd.read_csv(error_file)

# Only samples actually present in the frame-level CSV.
ids = sorted(frames["sample_index"].unique())
print("Selected samples:", ids)

# Summarize the frames present for each selected sample.
summary = frames.groupby("sample_index").agg(
    frames_in_file=("frame", "count"),
    blank_wins=("predicted_id", lambda x: (x == 0).sum()),
    mean_blank_margin=("blank_margin", "mean"),
    mean_blank_probability=("blank_probability", "mean"),
    median_blank_probability=("blank_probability", "median"),
    mean_entropy=("entropy", "mean"),
).reset_index()

summary["blank_win_fraction"] = (
    summary["blank_wins"] / summary["frames_in_file"]
)

# Add existing greedy and beam evaluation metrics.
metrics = errors[
    ["sample_index", "greedy_cer", "greedy_wer",
     "beam4g_cer", "beam4g_wer", "category"]
]

summary = summary.merge(
    metrics, on="sample_index", how="left", validate="one_to_one"
)

print("\nSELECTED-SAMPLE COMPARISON")
print(summary.round(4).to_string(index=False))

print("\nNote: frame metrics describe only frames saved in the CSV.")
print("They are not full-utterance statistics unless every frame is present.")
