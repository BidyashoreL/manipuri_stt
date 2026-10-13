# ============================================================
# 06_evaluate_final_test_beam500.py
#
# FINAL B2 OFFICIAL TEST EVALUATION
#
# Model:
#   FINAL_B2_MODEL.pt
#
# Decoder:
#   pyctcdecode unigram-assisted CTC beam search
#   Beam width = 500
#
# Dataset:
#   Official untouched test set
#   410 samples
#
# IMPORTANT:
#   DO NOT tune anything using this test set.
#   This is the final B2 held-out evaluation.
# ============================================================

import sys
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pyctcdecode import build_ctcdecoder
from transformers import AutoFeatureExtractor


# ============================================================
# CONFIGURATION
# ============================================================

# This script is located inside:
#
# ...\ASR\manipuri_asr_v2\
#
SCRIPT_DIR = Path(__file__).resolve().parent

# ...\ASR\
ASR_DIR = SCRIPT_DIR.parent

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

MODEL_PATH = (
    ASR_DIR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "regularized"
    / "FINAL_B2_MODEL.pt"
)

VOCAB_PATH = (
    ASR_DIR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

TEST_CSV = (
    ASR_DIR
    / "combined_dataset"
    / "test.csv"
)

TEST_CACHE = (
    ASR_DIR
    / "test_audio_cache"
)

UNIGRAM_PATH = (
    SCRIPT_DIR
    / "kenlm_experiment"
    / "manipuri_unigrams.txt"
)

OUTPUT_DIR = (
    ASR_DIR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "regularized"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "FINAL_B2_MODEL_beam500_test_results.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "FINAL_B2_MODEL_beam500_test_summary.json"
)

PREDICTIONS_CSV = (
    OUTPUT_DIR
    / "FINAL_B2_MODEL_beam500_test_predictions.csv"
)

PROCESSOR_PATH = r"E:\hf_models\w2v-bert-2.0"

BEAM_WIDTH = 500

EXPECTED_TEST_SAMPLES = 410


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 75)
print("FINAL B2 OFFICIAL TEST — UNIGRAM BEAM 500")
print("=" * 75)

print(f"[INFO] Device          : {DEVICE}")
print(f"[INFO] Model           : {MODEL_PATH}")
print(f"[INFO] Test CSV        : {TEST_CSV}")
print(f"[INFO] Test cache      : {TEST_CACHE}")
print(f"[INFO] Beam width      : {BEAM_WIDTH}")


# ============================================================
# SAFETY CHECKS
# ============================================================

print()
print("[INFO] Running safety checks...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"FINAL_B2_MODEL.pt not found:\n{MODEL_PATH}"
    )

if not VOCAB_PATH.exists():
    raise FileNotFoundError(
        f"Vocabulary not found:\n{VOCAB_PATH}"
    )

if not TEST_CSV.exists():
    raise FileNotFoundError(
        f"Official test CSV not found:\n{TEST_CSV}"
    )

if not TEST_CACHE.exists():
    raise FileNotFoundError(
        f"Test audio cache not found:\n{TEST_CACHE}"
    )

if not UNIGRAM_PATH.exists():
    raise FileNotFoundError(
        f"Unigram dictionary not found:\n{UNIGRAM_PATH}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("[PASS] Required files exist.")


# ============================================================
# LOAD VOCABULARY
# ============================================================

print()
print("[INFO] Loading vocabulary...")

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8",
) as f:
    vocab = json.load(f)

# JSON keys may be strings
vocab = {
    str(k): int(v)
    for k, v in vocab.items()
}

id_to_token = {
    idx: token
    for token, idx in vocab.items()
}

blank_id = vocab["<blank>"]
unk_id = vocab["<unk>"]

print(f"[INFO] Vocabulary size : {len(vocab)}")
print(f"[INFO] Blank ID        : {blank_id}")
print(f"[INFO] UNK ID          : {unk_id}")


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("[INFO] Loading FINAL_B2_MODEL.pt...")

# Existing B2 architecture.
sys.path.insert(
    0,
    str(SCRIPT_DIR),
)

from train_b2_partial_encoder import B2Model


model = B2Model(
    vocab_size=len(vocab)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    state_dict = checkpoint["model_state_dict"]

elif (
    isinstance(checkpoint, dict)
    and "state_dict" in checkpoint
):
    state_dict = checkpoint["state_dict"]

else:
    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True,
)

model.to(DEVICE)
model.eval()

print("[PASS] FINAL_B2_MODEL loaded successfully.")


# ============================================================
# MODEL PARAMETER INFORMATION
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print("[MODEL]")
print(f"  Total parameters     : {total_params:,}")
print(f"  Trainable parameters : {trainable_params:,}")


# ============================================================
# LOAD PROCESSOR
# ============================================================

print()
print("[INFO] Loading W2V-BERT 2.0 processor...")

processor = AutoFeatureExtractor.from_pretrained(
    PROCESSOR_PATH
)

print("[PASS] Processor loaded.")


# ============================================================
# LOAD OFFICIAL TEST DATA
# ============================================================

print()
print("[INFO] Loading official test CSV...")

df = pd.read_csv(
    TEST_CSV
)

if len(df) != EXPECTED_TEST_SAMPLES:
    raise RuntimeError(
        f"Expected exactly {EXPECTED_TEST_SAMPLES} "
        f"official test samples, got {len(df)}"
    )

print(
    f"[PASS] Official test samples: {len(df)}"
)


# ============================================================
# VERIFY TEST CACHE
# ============================================================

print()
print("[INFO] Checking official test cache...")

missing_cache = []

for i in range(len(df)):

    cache_path = (
        TEST_CACHE
        / f"{i:06d}.pt"
    )

    if not cache_path.exists():
        missing_cache.append(
            str(cache_path)
        )

if missing_cache:

    print()
    print("[ERROR] Missing test cache files:")

    for path in missing_cache[:20]:
        print(f"  {path}")

    raise RuntimeError(
        f"Missing {len(missing_cache)} "
        f"test cache files."
    )

print(
    f"[PASS] {len(df)} / {len(df)} "
    f"test cache files found."
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize reference/hypothesis text.

    NFC normalization + whitespace normalization.
    """

    if text is None:
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFC",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# CHARACTER ERROR RATE
# ============================================================

def edit_distance(
    reference,
    hypothesis,
):
    """
    Standard Levenshtein edit distance.
    """

    reference = list(reference)
    hypothesis = list(hypothesis)

    n = len(reference)
    m = len(hypothesis)

    if n == 0:
        return m

    if m == 0:
        return n

    previous = list(range(m + 1))

    for i in range(1, n + 1):

        current = [i]

        ref_char = reference[i - 1]

        for j in range(1, m + 1):

            hyp_char = hypothesis[j - 1]

            insertion = (
                current[j - 1] + 1
            )

            deletion = (
                previous[j] + 1
            )

            substitution = (
                previous[j - 1]
                + (ref_char != hyp_char)
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution,
                )
            )

        previous = current

    return previous[-1]


def cer(
    reference,
    hypothesis,
):
    """
    Character Error Rate.
    """

    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    if len(reference) == 0:
        return (
            0.0
            if len(hypothesis) == 0
            else 1.0
        )

    distance = edit_distance(
        reference,
        hypothesis,
    )

    return distance / len(reference)


# ============================================================
# WORD ERROR RATE
# ============================================================

def wer(
    reference,
    hypothesis,
):
    """
    Word Error Rate.
    """

    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    ref_words = (
        reference.split()
        if reference
        else []
    )

    hyp_words = (
        hypothesis.split()
        if hypothesis
        else []
    )

    if len(ref_words) == 0:
        return (
            0.0
            if len(hyp_words) == 0
            else 1.0
        )

    distance = edit_distance(
        ref_words,
        hyp_words,
    )

    return distance / len(ref_words)


# ============================================================
# LOAD UNIGRAM DICTIONARY
# ============================================================

print()
print("[INFO] Loading Manipuri unigram dictionary...")

with open(
    UNIGRAM_PATH,
    "r",
    encoding="utf-8",
) as f:

    unigrams = [
        line.strip()
        for line in f
        if line.strip()
    ]

print(
    f"[PASS] Unigrams loaded: "
    f"{len(unigrams):,}"
)


# ============================================================
# BUILD CTC DECODER
# ============================================================

print()
print("[INFO] Building pyctcdecode unigram decoder...")

# pyctcdecode needs the CTC blank represented
# as an empty label.
#
# <unk> must remain distinct because an empty
# string would create a duplicate decoder label.

decoder_labels = []

for token_id in range(len(vocab)):

    token = id_to_token[token_id]

    if token == "<blank>":

        decoder_labels.append("")

    elif token == "<unk>":

        decoder_labels.append("<unk>")

    else:

        decoder_labels.append(token)


# Safety check
if len(decoder_labels) != len(vocab):
    raise RuntimeError(
        "Decoder label count does not match vocabulary."
    )

if len(set(decoder_labels)) != len(decoder_labels):
    duplicates = [
        label
        for label in set(decoder_labels)
        if decoder_labels.count(label) > 1
    ]

    raise RuntimeError(
        f"Duplicate decoder labels found: {duplicates}"
    )


decoder = build_ctcdecoder(
    labels=decoder_labels,
    kenlm_model_path=None,
    unigrams=unigrams,
)

print("[PASS] Decoder constructed.")
print(
    f"[INFO] Decoder labels : "
    f"{len(decoder_labels)}"
)
print(
    f"[INFO] Beam width     : "
    f"{BEAM_WIDTH}"
)


# ============================================================
# GREEDY CTC DECODER
# ============================================================

def greedy_decode(logits):
    """
    Standard CTC greedy decoding.
    """

    pred_ids = (
        torch.argmax(
            logits,
            dim=-1,
        )[0]
        .detach()
        .cpu()
        .tolist()
    )

    output = []

    previous = None

    for token_id in pred_ids:

        # Blank
        if token_id == blank_id:

            previous = token_id

            continue

        # CTC repeated token
        if token_id == previous:

            continue

        # UNK is ignored in the existing B2 decoder
        if token_id == unk_id:

            previous = token_id

            continue

        output.append(
            id_to_token[token_id]
        )

        previous = token_id

    return "".join(output)


# ============================================================
# START FINAL TEST
# ============================================================

print()
print("=" * 75)
print("STARTING FINAL OFFICIAL TEST")
print("=" * 75)

print()
print("IMPORTANT:")
print("  Dataset        : Official test")
print("  Samples        : 410")
print("  Model          : FINAL_B2_MODEL.pt")
print("  Decoder        : Unigram CTC Beam Search")
print("  Beam width     : 500")
print("  Test set       : NEVER used for tuning")
print()


results = []

total_loss = 0.0


# ============================================================
# TEST LOOP
# ============================================================

for i in range(len(df)):

    # --------------------------------------------------------
    # LOAD CACHED AUDIO
    # --------------------------------------------------------

    cache_path = (
        TEST_CACHE
        / f"{i:06d}.pt"
    )

    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=True,
    )

    if "audio" not in payload:
        raise RuntimeError(
            f"'audio' key missing from "
            f"{cache_path}"
        )

    audio = payload["audio"].float()

    # --------------------------------------------------------
    # AUDIO SHAPE
    # --------------------------------------------------------

    if audio.ndim == 1:

        audio_np = (
            audio
            .detach()
            .cpu()
            .numpy()
        )

    elif audio.ndim == 2:

        audio_np = (
            audio[0]
            .detach()
            .cpu()
            .numpy()
        )

    else:

        raise RuntimeError(
            f"Unexpected cached audio shape "
            f"at index {i}: "
            f"{tuple(audio.shape)}"
        )

    # Force:
    #   float32
    #   1-D
    audio_np = np.asarray(
        audio_np,
        dtype=np.float32,
    ).reshape(-1)

    # --------------------------------------------------------
    # REFERENCE
    # --------------------------------------------------------

    reference = normalize_text(
        df.iloc[i]["text"]
    )

    # --------------------------------------------------------
    # FEATURE EXTRACTION
    # --------------------------------------------------------

    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    if "input_features" in inputs:

        input_features = (
            inputs["input_features"]
            .to(
                DEVICE,
                dtype=torch.float32,
            )
        )

    elif "input_values" in inputs:

        input_features = (
            inputs["input_values"]
            .to(
                DEVICE,
                dtype=torch.float32,
            )
        )

    else:

        raise RuntimeError(
            "Unexpected processor output keys: "
            f"{list(inputs.keys())}"
        )

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            input_features
        )

        log_probs = F.log_softmax(
            logits,
            dim=-1,
        )

        # ----------------------------------------------------
        # GREEDY DECODING
        # ----------------------------------------------------

        greedy_hypothesis = (
            greedy_decode(logits)
        )

        # ----------------------------------------------------
        # BEAM SEARCH
        # ----------------------------------------------------

        decoder_input = (
            log_probs[0]
            .detach()
            .cpu()
            .numpy()
        )

        beam_hypothesis = (
            decoder.decode(
                decoder_input,
                beam_width=BEAM_WIDTH,
            )
        )

        beam_hypothesis = normalize_text(
            beam_hypothesis
        )

    # --------------------------------------------------------
    # CTC LOSS
    # --------------------------------------------------------

    targets = []

    for ch in reference:

        if ch not in vocab:

            targets.append(
                unk_id
            )

        else:

            targets.append(
                vocab[ch]
            )

    target_tensor = torch.tensor(
        targets,
        dtype=torch.long,
        device=DEVICE,
    ).unsqueeze(0)

    target_lengths = torch.tensor(
        [len(targets)],
        dtype=torch.long,
        device=DEVICE,
    )

    input_lengths = torch.tensor(
        [logits.shape[1]],
        dtype=torch.long,
        device=DEVICE,
    )

    log_probs_ctc = (
        log_probs.transpose(0, 1)
    )

    loss = F.ctc_loss(
        log_probs_ctc,
        target_tensor,
        input_lengths,
        target_lengths,
        blank=blank_id,
        zero_infinity=True,
    )

    loss_value = float(
        loss.item()
    )

    total_loss += loss_value

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    greedy_cer = cer(
        reference,
        greedy_hypothesis,
    )

    greedy_wer = wer(
        reference,
        greedy_hypothesis,
    )

    beam_cer = cer(
        reference,
        beam_hypothesis,
    )

    beam_wer = wer(
        reference,
        beam_hypothesis,
    )

    # --------------------------------------------------------
    # SAVE RESULT
    # --------------------------------------------------------

    results.append(
        {
            "index": i,
            "reference": reference,
            "greedy_hypothesis": greedy_hypothesis,
            "beam500_hypothesis": beam_hypothesis,
            "greedy_cer": greedy_cer,
            "greedy_wer": greedy_wer,
            "beam500_cer": beam_cer,
            "beam500_wer": beam_wer,
            "loss": loss_value,
        }
    )

    # --------------------------------------------------------
    # SHOW FIRST 10
    # --------------------------------------------------------

    if i < 10:

        print()
        print(
            f"[{i + 1}/{len(df)}]"
        )

        print(
            f"REF    : {reference}"
        )

        print(
            f"GREEDY : {greedy_hypothesis}"
        )

        print(
            f"BEAM500: {beam_hypothesis}"
        )

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        (i + 1) % 25 == 0
        or i == len(df) - 1
    ):

        current_df = pd.DataFrame(
            results
        )

        current_greedy_cer = (
            current_df["greedy_cer"]
            .mean()
        )

        current_greedy_wer = (
            current_df["greedy_wer"]
            .mean()
        )

        current_beam_cer = (
            current_df["beam500_cer"]
            .mean()
        )

        current_beam_wer = (
            current_df["beam500_wer"]
            .mean()
        )

        current_loss = (
            current_df["loss"]
            .mean()
        )

        print()
        print(
            f"[{i + 1}/{len(df)}] "
            f"Loss={current_loss:.6f} | "
            f"Greedy CER="
            f"{current_greedy_cer * 100:.4f}% | "
            f"Beam500 CER="
            f"{current_beam_cer * 100:.4f}% | "
            f"Greedy WER="
            f"{current_greedy_wer * 100:.4f}% | "
            f"Beam500 WER="
            f"{current_beam_wer * 100:.4f}%"
        )


# ============================================================
# FINAL METRICS
# ============================================================

result_df = pd.DataFrame(
    results
)

mean_loss = (
    result_df["loss"]
    .mean()
)

mean_greedy_cer = (
    result_df["greedy_cer"]
    .mean()
)

mean_greedy_wer = (
    result_df["greedy_wer"]
    .mean()
)

mean_beam_cer = (
    result_df["beam500_cer"]
    .mean()
)

mean_beam_wer = (
    result_df["beam500_wer"]
    .mean()
)


# ============================================================
# IMPROVEMENT
# ============================================================

cer_change = (
    mean_beam_cer
    - mean_greedy_cer
)

wer_change = (
    mean_beam_wer
    - mean_greedy_wer
)


# ============================================================
# SAVE DETAILED RESULTS
# ============================================================

result_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)

print()
print(
    f"[SAVED] Detailed results:"
)
print(
    f"        {OUTPUT_CSV}"
)


# ============================================================
# SAVE PREDICTIONS-ONLY FILE
# ============================================================

prediction_df = result_df[
    [
        "index",
        "reference",
        "beam500_hypothesis",
        "beam500_cer",
        "beam500_wer",
    ]
].copy()

prediction_df.to_csv(
    PREDICTIONS_CSV,
    index=False,
    encoding="utf-8-sig",
)

print(
    f"[SAVED] Predictions:"
)
print(
    f"        {PREDICTIONS_CSV}"
)


# ============================================================
# SUMMARY JSON
# ============================================================

summary = {
    "experiment": "B2_final_official_test_beam500",

    "model": "FINAL_B2_MODEL.pt",

    "model_path": str(
        MODEL_PATH.resolve()
    ),

    "decoder": (
        "pyctcdecode "
        "unigram_assisted_ctc_beam_search"
    ),

    "beam_width": BEAM_WIDTH,

    "unigram_count": len(
        unigrams
    ),

    "test_samples": len(
        result_df
    ),

    "official_test_set": True,

    "test_set_used_for_training": False,

    "test_set_used_for_decoder_tuning": False,

    "validation_selected_decoder": (
        "unigram_beam_500"
    ),

    "validation_selected_by": (
        "minimum_validation_CER"
    ),

    "test_loss": mean_loss,

    "greedy_cer": mean_greedy_cer,

    "greedy_wer": mean_greedy_wer,

    "beam500_cer": mean_beam_cer,

    "beam500_wer": mean_beam_wer,

    "beam500_minus_greedy_cer": cer_change,

    "beam500_minus_greedy_wer": wer_change,

    "beam500_cer_improvement_pp": (
        -cer_change * 100
    ),

    "beam500_wer_improvement_pp": (
        -wer_change * 100
    ),

    "vocab_size": len(
        vocab
    ),

    "total_model_parameters": (
        total_params
    ),

    "trainable_model_parameters": (
        trainable_params
    ),
}

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        ensure_ascii=False,
        indent=2,
    )

print(
    f"[SAVED] Summary:"
)
print(
    f"        {OUTPUT_JSON}"
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 75)
print("FINAL B2 OFFICIAL TEST RESULT")
print("=" * 75)

print()
print(
    f"Test samples       : "
    f"{len(result_df)}"
)

print(
    f"Test loss          : "
    f"{mean_loss:.6f}"
)

print()
print("GREEDY CTC")
print(
    f"  CER              : "
    f"{mean_greedy_cer * 100:.4f}%"
)

print(
    f"  WER              : "
    f"{mean_greedy_wer * 100:.4f}%"
)

print()
print("UNIGRAM BEAM SEARCH — BEAM 500")

print(
    f"  CER              : "
    f"{mean_beam_cer * 100:.4f}%"
)

print(
    f"  WER              : "
    f"{mean_beam_wer * 100:.4f}%"
)

print()
print("BEAM 500 − GREEDY")

print(
    f"  CER change       : "
    f"{cer_change * 100:+.4f} pp"
)

print(
    f"  WER change       : "
    f"{wer_change * 100:+.4f} pp"
)

print()
print("IMPROVEMENT")

print(
    f"  CER improvement  : "
    f"{-cer_change * 100:+.4f} pp"
)

print(
    f"  WER improvement  : "
    f"{-wer_change * 100:+.4f} pp"
)

print()
print("=" * 75)
print("B2 FINAL TEST COMPLETE")
print("=" * 75)

print()
print("[OUTPUTS]")
print(
    f"  Results   : {OUTPUT_CSV}"
)
print(
    f"  Predictions: {PREDICTIONS_CSV}"
)
print(
    f"  Summary   : {OUTPUT_JSON}"
)

print()
print(
    "[IMPORTANT] Official test set remains "
    "untouched for all future B3 tuning."
)

print()
print("[DONE]")