from pathlib import Path
import pandas as pd
import streamlit as st
import soundfile as sf

ROOT = Path.cwd()
REVIEW_CSV = ROOT / "audio_audit/safe_cleaned/validation_checks/combined_manual_review.csv"
ORIGINAL_DIR = ROOT / "combined_dataset"
OUTPUT_CSV = ROOT / "audio_audit/safe_cleaned/validation_checks/manual_review_decisions.csv"

st.set_page_config(page_title="Manipuri Audio Review", layout="wide")
st.title("Manipuri ASR — Audio Cleaning Review")

review = pd.read_csv(REVIEW_CSV).fillna("")
decisions = (
    pd.read_csv(OUTPUT_CSV).fillna("")
    if OUTPUT_CSV.exists()
    else pd.DataFrame(columns=[
        "cleaned_path", "original_path", "text", "split",
        "review_reason", "decision", "notes"
    ])
)

if "idx" not in st.session_state:
    st.session_state.idx = 0

# Load the matching original manifest for each split.
originals = {}
for split in ["train", "validation"]:
    manifest = ORIGINAL_DIR / f"{split}.csv"
    if manifest.exists():
        df = pd.read_csv(manifest).fillna("")
        df["audio_path"] = df["audio_path"].astype(str)
        originals[split] = df

def find_original(row):
    split = str(row["split"])
    text = str(row["text"]).strip()
    cleaned_path = Path(str(row["path"]))
    candidates = originals.get(split)

    if candidates is None:
        return ""

    # First try an exact transcript match.
    matches = candidates[candidates["text"].astype(str).str.strip() == text]
    if len(matches) == 1:
        return str(matches.iloc[0]["audio_path"])

    # Prefer a filename match if the cleaned filename was preserved.
    filename_matches = candidates[
        candidates["audio_path"].apply(
            lambda p: Path(str(p)).name == cleaned_path.name
        )
    ]
    if len(filename_matches) == 1:
        return str(filename_matches.iloc[0]["audio_path"])

    # Do not guess if matching is ambiguous.
    return ""

st.write(f"Review clips: **{len(review)}**")
st.write("Choose a decision for each clip, then click Save decision.")

if not len(review):
    st.success("No clips to review.")
    st.stop()

idx = min(st.session_state.idx, len(review) - 1)
row = review.iloc[idx]
cleaned_path = str(row["path"])
original_path = find_original(row)

st.progress((idx + 1) / len(review))
st.subheader(f"Clip {idx + 1} of {len(review)} — {row['split']}")
st.write("**Transcript (Meitei Mayek):**")
st.code(str(row["text"]) or "[EMPTY TRANSCRIPT]", language=None)
st.write("**Review reason:**", row["review_reason"])
st.write("**Cleaned duration:**", row["duration_sec"], "seconds")
st.write("**Seconds removed:**", row["seconds_removed"] or "Not recorded")

left, right = st.columns(2)
with left:
    st.markdown("### Original audio")
    if original_path and Path(original_path).is_file():
        st.audio(str(original_path))
        st.caption(original_path)
    else:
        st.warning("Original audio could not be matched safely.")
with right:
    st.markdown("### Cleaned audio")
    if Path(cleaned_path).is_file():
        st.audio(cleaned_path)
        st.caption(cleaned_path)
    else:
        st.error("Cleaned file is missing.")

existing = decisions[
    decisions["cleaned_path"].astype(str) == cleaned_path
]
default_decision = (
    str(existing.iloc[-1]["decision"])
    if len(existing) else "Keep cleaned"
)
options = ["Keep cleaned", "Use original", "Exclude", "Needs expert review"]
selected = st.radio(
    "Decision",
    options,
    index=options.index(default_decision)
    if default_decision in options else 0,
    horizontal=True
)
notes = st.text_input("Notes (optional)")

if st.button("Save decision"):
    if selected == "Use original" and not original_path:
        st.error("Original path is unresolved. Do not save this decision yet.")
    else:
        record = {
            "cleaned_path": cleaned_path,
            "original_path": original_path,
            "text": str(row["text"]),
            "split": str(row["split"]),
            "review_reason": str(row["review_reason"]),
            "decision": selected,
            "notes": notes,
        }
        decisions = decisions[
            decisions["cleaned_path"].astype(str) != cleaned_path
        ]
        decisions = pd.concat(
            [decisions, pd.DataFrame([record])], ignore_index=True
        )
        decisions.to_csv(OUTPUT_CSV, index=False)
        st.success(f"Saved: {selected}")
        st.rerun()

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("⬅ Previous", disabled=(idx == 0)):
        st.session_state.idx -= 1
        st.rerun()
with col2:
    st.write(f"Saved decisions: {len(decisions)} / {len(review)}")
with col3:
    if st.button("Next ➡", disabled=(idx >= len(review) - 1)):
        st.session_state.idx += 1
        st.rerun()

st.caption(f"Decision file: {OUTPUT_CSV}")
