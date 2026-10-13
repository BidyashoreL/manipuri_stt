
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

REP_FILE = ROOT / "diagnostics" / "representation_forensics" / "representation_all_864.csv"
ERR_FILE = (
    ROOT / ".." / "checkpoints" / "v3_ctc" / "b3_narrow"
    / "diagnostics" / "sample_diagnostics.csv"
)

OUT_DIR = ROOT / "diagnostics" / "representation_forensics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Exact samples used in the blank-margin decomposition
SEVERE = [187, 349, 491, 568, 803, 814]
NORMAL = [16, 26, 59, 76, 81, 97, 126, 156, 195, 205, 207, 241]

FEATURES = [
    "hidden_norm_mean",
    "hidden_norm_std",
    "hidden_frame_std_mean",
    "adapted_norm_mean",
    "adapted_norm_std",
    "adapted_frame_std_mean",
    "temporal_change",
    "mean_blank_probability",
    "blank_fraction",
    "active_fraction",
    "max_nonblank_mean",
    "max_nonblank_max",
    "entropy_mean",
    "entropy_std",
    "active_span_fraction",
]

def cohens_d(a, b):
    """Severe minus normal, using pooled sample standard deviation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or len(b) < 2:
        return np.nan

    pooled_var = (
        (len(a) - 1) * np.var(a, ddof=1)
        + (len(b) - 1) * np.var(b, ddof=1)
    ) / (len(a) + len(b) - 2)

    if pooled_var <= 0:
        return np.nan

    return (np.mean(a) - np.mean(b)) / np.sqrt(pooled_var)


def main():
    if not REP_FILE.exists():
        raise FileNotFoundError(f"Representation CSV not found: {REP_FILE}")

    rep = pd.read_csv(REP_FILE)
    rep["sample_index"] = pd.to_numeric(
        rep["sample_index"], errors="raise"
    ).astype(int)

    required = {"sample_index", *FEATURES}
    missing = required - set(rep.columns)
    if missing:
        raise ValueError(f"Missing columns in representation CSV: {sorted(missing)}")

    if rep["sample_index"].duplicated().any():
        raise ValueError("Duplicate sample_index values found in representation CSV.")

    selected_ids = SEVERE + NORMAL
    selected = rep[rep["sample_index"].isin(selected_ids)].copy()

    found = set(selected["sample_index"])
    missing_ids = sorted(set(selected_ids) - found)
    if missing_ids:
        raise ValueError(f"These required samples are missing: {missing_ids}")

    selected["comparison_group"] = np.where(
        selected["sample_index"].isin(SEVERE),
        "SEVERE",
        "NORMAL_COMPARISON",
    )

    # Add error metrics if the diagnostic file is available.
    if ERR_FILE.exists():
        err = pd.read_csv(ERR_FILE)
        if "sample_index" in err.columns:
            err["sample_index"] = pd.to_numeric(
                err["sample_index"], errors="coerce"
            )
            err = err.dropna(subset=["sample_index"])
            err["sample_index"] = err["sample_index"].astype(int)

            useful = [
                c for c in [
                    "sample_index", "beam4g_cer", "beam4g_wer",
                    "greedy_cer", "greedy_wer", "category",
                    "reference"
                ]
                if c in err.columns
            ]
            selected = selected.merge(
                err[useful].drop_duplicates("sample_index"),
                on="sample_index",
                how="left",
                suffixes=("", "_error"),
            )

    selected = selected.sort_values(
        ["comparison_group", "sample_index"]
    )
    per_sample_path = OUT_DIR / "matched_18_representation_samples.csv"
    selected.to_csv(per_sample_path, index=False)

    rows = []
    for feature in FEATURES:
        severe_values = selected.loc[
            selected["comparison_group"] == "SEVERE", feature
        ].dropna().astype(float)

        normal_values = selected.loc[
            selected["comparison_group"] == "NORMAL_COMPARISON", feature
        ].dropna().astype(float)

        rows.append({
            "feature": feature,
            "severe_n": len(severe_values),
            "normal_n": len(normal_values),
            "severe_mean": severe_values.mean(),
            "normal_mean": normal_values.mean(),
            "delta_severe_minus_normal":
                severe_values.mean() - normal_values.mean(),
            "cohens_d_severe_minus_normal":
                cohens_d(severe_values, normal_values),
            "severe_median": severe_values.median(),
            "normal_median": normal_values.median(),
        })

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "matched_18_representation_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nMATCHED SAMPLE COUNTS")
    print("Severe:", int((selected["comparison_group"] == "SEVERE").sum()))
    print(
        "Normal comparison:",
        int((selected["comparison_group"] == "NORMAL_COMPARISON").sum())
    )

    print("\nPER-SAMPLE FEATURES")
    show_cols = [
        "sample_index", "comparison_group", "collapse_class",
        "hidden_norm_mean", "adapted_norm_mean", "temporal_change",
        "blank_fraction", "active_fraction", "entropy_mean",
    ]
    show_cols = [c for c in show_cols if c in selected.columns]
    print(selected[show_cols].to_string(index=False))

    print("\nMATCHED GROUP COMPARISON")
    print(summary.round(4).to_string(index=False))

    print("\nSaved files:")
    print(per_sample_path)
    print(summary_path)

    if len(selected) != 18:
        raise RuntimeError(
            f"Expected 18 unique samples, found {len(selected)}."
        )

    print("\nCHECK PASSED: all 18 requested samples were found.")


if __name__ == "__main__":
    main()
