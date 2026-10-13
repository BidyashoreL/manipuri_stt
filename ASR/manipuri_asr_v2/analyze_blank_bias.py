from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ASR = ROOT.parent

CACHE_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "val_logit_cache"
)

OUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# LOAD CACHE
# =========================================================

print("=" * 80)
print("B3 CTC BLANK / EMISSION BIAS ANALYSIS")
print("=" * 80)

logits = np.load(
    CACHE_DIR / "logits.npy",
    mmap_mode="r"
)

offsets = np.load(
    CACHE_DIR / "offsets.npy"
)

texts = json.loads(
    (CACHE_DIR / "texts.json").read_text(
        encoding="utf-8"
    )
)

print(f"Logits shape : {logits.shape}")
print(f"Samples      : {len(texts)}")


# =========================================================
# LOAD VOCAB
# =========================================================

VOCAB_PATH = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as f:
    vocab = json.load(f)


# vocab can be either token -> id or id -> token
if all(str(k).isdigit() for k in vocab.keys()):
    id_to_token = {
        int(k): v
        for k, v in vocab.items()
    }
else:
    id_to_token = {
        int(v): k
        for k, v in vocab.items()
    }


BLANK_ID = id_to_token.get(
    0,
    "<blank>"
)

print(f"Blank ID     : 0")
print(f"Blank token  : {BLANK_ID}")


# =========================================================
# SOFTMAX
# =========================================================

def softmax(x):
    x = x.astype(np.float64)

    x = x - np.max(
        x,
        axis=-1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return exp_x / exp_x.sum(
        axis=-1,
        keepdims=True
    )


# =========================================================
# ANALYZE EACH SAMPLE
# =========================================================

records = []

global_total_frames = 0
global_blank_frames = 0
global_nonblank_frames = 0

global_blank_prob_sum = 0.0
global_nonblank_prob_sum = 0.0

for i, text in enumerate(texts):

    start = int(offsets[i])
    end = int(offsets[i + 1])

    sample_logits = np.asarray(
        logits[start:end]
    )

    probs = softmax(sample_logits)

    pred_ids = np.argmax(
        sample_logits,
        axis=-1
    )

    blank_probs = probs[:, 0]

    max_probs = probs.max(axis=1)

    blank_mask = pred_ids == 0

    blank_frames = int(
        blank_mask.sum()
    )

    total_frames = len(pred_ids)

    nonblank_frames = (
        total_frames - blank_frames
    )

    blank_fraction = (
        blank_frames / total_frames
        if total_frames
        else 0.0
    )

    nonblank_fraction = (
        nonblank_frames / total_frames
        if total_frames
        else 0.0
    )

    mean_blank_probability = float(
        blank_probs.mean()
    )

    median_blank_probability = float(
        np.median(blank_probs)
    )

    mean_max_probability = float(
        max_probs.mean()
    )

    # Collapse CTC output
    collapsed = []

    previous = None

    for token_id in pred_ids:

        if token_id == 0:
            previous = token_id
            continue

        if token_id == previous:
            continue

        collapsed.append(
            int(token_id)
        )

        previous = token_id

    predicted_length = len(
        collapsed
    )

    reference_length = len(text)

    length_ratio = (
        predicted_length / reference_length
        if reference_length
        else np.nan
    )

    length_error_ratio = (
        predicted_length / reference_length
        if reference_length
        else np.nan
    )

    records.append(
        {
            "index": i,
            "T": total_frames,
            "reference_length": reference_length,
            "predicted_length": predicted_length,
            "pred_ref_length_ratio": length_ratio,
            "blank_frames": blank_frames,
            "nonblank_frames": nonblank_frames,
            "blank_fraction": blank_fraction,
            "nonblank_fraction": nonblank_fraction,
            "mean_blank_probability":
                mean_blank_probability,
            "median_blank_probability":
                median_blank_probability,
            "mean_max_probability":
                mean_max_probability,
            "text": text,
        }
    )

    global_total_frames += total_frames
    global_blank_frames += blank_frames
    global_nonblank_frames += nonblank_frames

    global_blank_prob_sum += (
        blank_probs.sum()
    )

    global_nonblank_prob_sum += (
        1.0 - blank_probs
    ).sum()


df = pd.DataFrame(records)


# =========================================================
# GLOBAL SUMMARY
# =========================================================

print()
print("=" * 80)
print("GLOBAL BLANK / EMISSION SUMMARY")
print("=" * 80)

print(
    f"Total frames             : "
    f"{global_total_frames:,}"
)

print(
    f"Blank argmax frames      : "
    f"{global_blank_frames:,}"
)

print(
    f"Non-blank argmax frames  : "
    f"{global_nonblank_frames:,}"
)

print(
    f"Blank argmax fraction    : "
    f"{global_blank_frames / global_total_frames:.4f}"
)

print(
    f"Non-blank argmax fraction: "
    f"{global_nonblank_frames / global_total_frames:.4f}"
)

print(
    f"Mean blank probability   : "
    f"{global_blank_prob_sum / global_total_frames:.4f}"
)

print(
    f"Mean nonblank probability: "
    f"{global_nonblank_prob_sum / global_total_frames:.4f}"
)


# =========================================================
# SAMPLE DISTRIBUTIONS
# =========================================================

print()
print("=" * 80)
print("SAMPLE-LEVEL DISTRIBUTIONS")
print("=" * 80)

for column in [
    "blank_fraction",
    "mean_blank_probability",
    "pred_ref_length_ratio",
]:

    print()
    print(column)

    print(
        df[column].describe(
            percentiles=[
                0.01,
                0.05,
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        ).to_string()
    )


# =========================================================
# BLANK-FRACTION ZONES
# =========================================================

print()
print("=" * 80)
print("BLANK-FRACTION ZONES")
print("=" * 80)

zones = [
    ("< 0.50", df["blank_fraction"] < 0.50),
    ("0.50-0.60",
     (df["blank_fraction"] >= 0.50) &
     (df["blank_fraction"] < 0.60)),
    ("0.60-0.70",
     (df["blank_fraction"] >= 0.60) &
     (df["blank_fraction"] < 0.70)),
    ("0.70-0.80",
     (df["blank_fraction"] >= 0.70) &
     (df["blank_fraction"] < 0.80)),
    ("0.80-0.90",
     (df["blank_fraction"] >= 0.80) &
     (df["blank_fraction"] < 0.90)),
    (">= 0.90",
     df["blank_fraction"] >= 0.90),
]

for name, mask in zones:

    count = int(mask.sum())

    print(
        f"{name:10s}: "
        f"{count:4d} / {len(df)} "
        f"({100 * count / len(df):6.2f}%)"
    )


# =========================================================
# VERY HIGH BLANK SAMPLES
# =========================================================

print()
print("=" * 80)
print("20 HIGHEST BLANK-FRACTION SAMPLES")
print("=" * 80)

worst_blank = (
    df.sort_values(
        "blank_fraction",
        ascending=False
    )
    .head(20)
)

for _, r in worst_blank.iterrows():

    print()
    print(
        f"INDEX       : {int(r['index'])}"
    )

    print(
        f"T           : {int(r['T'])}"
    )

    print(
        f"Reference L : {int(r['reference_length'])}"
    )

    print(
        f"Predicted L : {int(r['predicted_length'])}"
    )

    print(
        f"Length ratio: "
        f"{r['pred_ref_length_ratio']:.4f}"
    )

    print(
        f"Blank frac  : "
        f"{r['blank_fraction']:.4f}"
    )

    print(
        f"Mean blank P: "
        f"{r['mean_blank_probability']:.4f}"
    )

    print(
        f"Text        : {r['text']}"
    )


# =========================================================
# LOW PREDICTION-LENGTH SAMPLES
# =========================================================

print()
print("=" * 80)
print("20 LOWEST PREDICTED/REFERENCE LENGTH RATIOS")
print("=" * 80)

worst_length = (
    df.sort_values(
        "pred_ref_length_ratio",
        ascending=True
    )
    .head(20)
)

for _, r in worst_length.iterrows():

    print()
    print(
        f"INDEX       : {int(r['index'])}"
    )

    print(
        f"T           : {int(r['T'])}"
    )

    print(
        f"Reference L : {int(r['reference_length'])}"
    )

    print(
        f"Predicted L : {int(r['predicted_length'])}"
    )

    print(
        f"Length ratio: "
        f"{r['pred_ref_length_ratio']:.4f}"
    )

    print(
        f"Blank frac  : "
        f"{r['blank_fraction']:.4f}"
    )

    print(
        f"Mean blank P: "
        f"{r['mean_blank_probability']:.4f}"
    )

    print(
        f"Text        : {r['text']}"
    )


# =========================================================
# LENGTH BINS
# =========================================================

bins = [
    0,
    5,
    10,
    20,
    40,
    80,
    120,
    10000,
]

labels = [
    "1-5",
    "6-10",
    "11-20",
    "21-40",
    "41-80",
    "81-120",
    "121+",
]

df["length_bin"] = pd.cut(
    df["reference_length"],
    bins=bins,
    labels=labels,
    include_lowest=True,
)


length_summary = (
    df.groupby(
        "length_bin",
        observed=False
    )
    .agg(
        samples=("index", "count"),
        mean_blank_fraction=(
            "blank_fraction",
            "mean"
        ),
        median_blank_fraction=(
            "blank_fraction",
            "median"
        ),
        mean_blank_probability=(
            "mean_blank_probability",
            "mean"
        ),
        mean_pred_ref_ratio=(
            "pred_ref_length_ratio",
            "mean"
        ),
        median_pred_ref_ratio=(
            "pred_ref_length_ratio",
            "median"
        ),
    )
    .reset_index()
)


print()
print("=" * 80)
print("BLANK BIAS BY REFERENCE LENGTH")
print("=" * 80)

print(
    length_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

length_summary.to_csv(
    OUT_DIR / "blank_bias_length_summary.csv",
    index=False
)


# =========================================================
# CORRELATIONS
# =========================================================

print()
print("=" * 80)
print("CORRELATIONS")
print("=" * 80)

corr_columns = [
    "reference_length",
    "T",
    "blank_fraction",
    "mean_blank_probability",
    "pred_ref_length_ratio",
]

print(
    df[corr_columns]
    .corr()
    .to_string(
        float_format=lambda x: f"{x:.4f}"
    )
)


# =========================================================
# SAVE
# =========================================================

output_csv = (
    OUT_DIR
    / "blank_bias_per_sample.csv"
)

df.to_csv(
    output_csv,
    index=False,
    encoding="utf-8-sig"
)

summary = {
    "total_frames": int(
        global_total_frames
    ),

    "blank_argmax_frames": int(
        global_blank_frames
    ),

    "nonblank_argmax_frames": int(
        global_nonblank_frames
    ),

    "blank_argmax_fraction":
        float(
            global_blank_frames
            / global_total_frames
        ),

    "nonblank_argmax_fraction":
        float(
            global_nonblank_frames
            / global_total_frames
        ),

    "mean_blank_probability":
        float(
            global_blank_prob_sum
            / global_total_frames
        ),

    "mean_nonblank_probability":
        float(
            global_nonblank_prob_sum
            / global_total_frames
        ),
}

with open(
    OUT_DIR / "blank_bias_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


print()
print("=" * 80)
print("FILES CREATED")
print("=" * 80)

print(output_csv)
print(OUT_DIR / "blank_bias_length_summary.csv")
print(OUT_DIR / "blank_bias_summary.json")

print()
print("DONE.")