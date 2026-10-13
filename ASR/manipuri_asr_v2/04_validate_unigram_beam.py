from pathlib import Path
import sys
import json
import re
import unicodedata
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pyctcdecode import build_ctcdecoder


# ============================================================
# PATHS
# ============================================================

ASR_DIR = Path(__file__).resolve().parent

VAL_CSV = ASR_DIR.parent / "combined_dataset" / "val.csv"
VAL_CACHE = ASR_DIR.parent / "val_audio_cache"

MODEL_DIR = (
    ASR_DIR.parent
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "regularized"
)

MODEL_PATH = MODEL_DIR / "FINAL_B2_MODEL.pt"

VOCAB_PATH = (
    ASR_DIR.parent
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

UNIGRAM_PATH = (
    ASR_DIR
    / "kenlm_experiment"
    / "manipuri_unigrams.txt"
)

OUTPUT_CSV = MODEL_DIR / "FINAL_B2_unigram_validation_results.csv"
OUTPUT_JSON = MODEL_DIR / "FINAL_B2_unigram_validation_summary.json"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("B2 + UNIGRAM CTC BEAM SEARCH VALIDATION")
print("=" * 70)

print(f"[INFO] Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# CHARACTER METRICS
# ============================================================

def edit_distance(a, b):
    m = len(a)
    n = len(b)

    prev = list(range(n + 1))

    for i in range(1, m + 1):

        curr = [i] + [0] * n

        for j in range(1, n + 1):

            cost = 0 if a[i - 1] == b[j - 1] else 1

            curr[j] = min(
                curr[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost,
            )

        prev = curr

    return prev[n]


def cer(reference, hypothesis):
    if len(reference) == 0:
        return 0.0 if len(hypothesis) == 0 else 1.0

    return edit_distance(reference, hypothesis) / len(reference)


def wer(reference, hypothesis):
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return edit_distance(ref_words, hyp_words) / len(ref_words)


# ============================================================
# LOAD VOCAB
# ============================================================

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

if len(vocab) != 60:
    raise RuntimeError(
        f"Expected vocab size 60, got {len(vocab)}"
    )

id_to_token = {
    idx: token
    for token, idx in vocab.items()
}

raw_labels = [
    id_to_token[i]
    for i in range(len(id_to_token))
]

# pyctcdecode:
# blank -> ""
# unk remains literal "<unk>"
decoder_labels = [
    "" if token == "<blank>" else token
    for token in raw_labels
]

# Make sure no duplicate labels remain.
seen = {}

for idx, label in enumerate(decoder_labels):

    if label in seen:
        raise RuntimeError(
            f"Duplicate decoder label {label!r} "
            f"at IDs {seen[label]} and {idx}"
        )

    seen[label] = idx

print(f"[INFO] Vocabulary size: {len(vocab)}")


# ============================================================
# LOAD UNIGRAMS
# ============================================================

with open(UNIGRAM_PATH, "r", encoding="utf-8") as f:

    unigrams = [
        line.rstrip("\n")
        for line in f
        if line.strip()
    ]

print(f"[INFO] Unigrams: {len(unigrams):,}")


# ============================================================
# BUILD DECODER
# ============================================================

print("[INFO] Building pyctcdecode decoder...")

decoder = build_ctcdecoder(
    labels=decoder_labels,
    unigrams=unigrams,
)

print("[PASS] Decoder constructed.")


# ============================================================
# LOAD MODEL
# ============================================================

print("[INFO] Loading FINAL_B2_MODEL.pt...")

# Import your existing model definition.
sys.path.insert(0, str(ASR_DIR))

from train_b2_partial_encoder import B2Model  # noqa: E402


model = B2Model(vocab_size=len(vocab))

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]

elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]

else:
    state_dict = checkpoint

model.load_state_dict(state_dict, strict=True)

model.to(DEVICE)
model.eval()

print("[PASS] Model loaded.")


# ============================================================
# LOAD PROCESSOR
# ============================================================

from transformers import AutoFeatureExtractor  # noqa: E402

PROCESSOR_PATH = r"E:\hf_models\w2v-bert-2.0"

processor = AutoFeatureExtractor.from_pretrained(
    PROCESSOR_PATH
)

print("[INFO] Processor loaded.")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

df = pd.read_csv(VAL_CSV)

if len(df) != 864:
    raise RuntimeError(
        f"Expected 864 validation samples, got {len(df)}"
    )

print(f"[INFO] Validation samples: {len(df)}")


# ============================================================
# GREEDY DECODER
# ============================================================

blank_id = vocab["<blank>"]
unk_id = vocab["<unk>"]


def greedy_decode(logits):

    pred_ids = torch.argmax(logits, dim=-1)[0].detach().cpu().tolist()

    output = []
    previous = None

    for token_id in pred_ids:

        if token_id == blank_id:
            previous = token_id
            continue

        if token_id == previous:
            continue

        if token_id == unk_id:
            previous = token_id
            continue

        output.append(id_to_token[token_id])

        previous = token_id

    return "".join(output)


# ============================================================
# VALIDATION LOOP
# ============================================================

results = []

total_loss = 0.0

print()
print("=" * 70)
print("STARTING VALIDATION")
print("=" * 70)

for i in range(len(df)):

    # --------------------------------------------------------
    # LOAD CACHED AUDIO
    # --------------------------------------------------------

    cache_path = VAL_CACHE / f"{i:06d}.pt"

    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=True,
    )

    audio = payload["audio"].float()

    reference = normalize_text(
        df.iloc[i]["text"]
    )

    # --------------------------------------------------------
    # ROBUST AUDIO SHAPE HANDLING
    #
    # Expected:
    #   [T]
    #   [1, T]
    # --------------------------------------------------------

    if audio.ndim == 1:

        audio_np = audio.detach().cpu().numpy()

    elif audio.ndim == 2:

        audio_np = audio[0].detach().cpu().numpy()

    else:

        raise RuntimeError(
            f"Unexpected cached audio shape at index {i}: "
            f"{tuple(audio.shape)}"
        )

    # Force:
    #   float32
    #   1-D
    audio_np = np.asarray(
        audio_np,
        dtype=np.float32
    ).reshape(-1)

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

        input_features = inputs["input_features"].to(
            DEVICE,
            dtype=torch.float32,
        )

    elif "input_values" in inputs:

        input_features = inputs["input_values"].to(
            DEVICE,
            dtype=torch.float32,
        )

    else:

        raise RuntimeError(
            f"Unexpected processor output keys: "
            f"{list(inputs.keys())}"
        )

    # --------------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(input_features)

        log_probs = F.log_softmax(
            logits,
            dim=-1,
        )

        # ----------------------------------------------------
        # GREEDY DECODING
        # ----------------------------------------------------

        greedy_hypothesis = greedy_decode(logits)

        # ----------------------------------------------------
        # UNIGRAM BEAM SEARCH
        # ----------------------------------------------------

        decoder_input = (
            log_probs[0]
            .detach()
            .cpu()
            .numpy()
        )

        beam_hypothesis = decoder.decode(
            decoder_input,
            beam_width=100,
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
            targets.append(unk_id)

        else:
            targets.append(vocab[ch])

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

    log_probs_ctc = log_probs.transpose(0, 1)

    loss = F.ctc_loss(
        log_probs_ctc,
        target_tensor,
        input_lengths,
        target_lengths,
        blank=blank_id,
        zero_infinity=True,
    )

    loss_value = float(loss.item())

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

    results.append(
        {
            "index": i,
            "reference": reference,
            "greedy_hypothesis": greedy_hypothesis,
            "beam_hypothesis": beam_hypothesis,
            "greedy_cer": greedy_cer,
            "greedy_wer": greedy_wer,
            "beam_cer": beam_cer,
            "beam_wer": beam_wer,
            "loss": loss_value,
        }
    )

    # --------------------------------------------------------
    # FIRST 10 EXAMPLES
    # --------------------------------------------------------

    if i < 10:

        print()
        print(f"[{i + 1}/{len(df)}]")
        print(f"REF    : {reference}")
        print(f"GREEDY : {greedy_hypothesis}")
        print(f"BEAM   : {beam_hypothesis}")

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (i + 1) % 25 == 0 or i == len(df) - 1:

        current_greedy_cer = (
            sum(r["greedy_cer"] for r in results)
            / len(results)
        )

        current_beam_cer = (
            sum(r["beam_cer"] for r in results)
            / len(results)
        )

        current_greedy_wer = (
            sum(r["greedy_wer"] for r in results)
            / len(results)
        )

        current_beam_wer = (
            sum(r["beam_wer"] for r in results)
            / len(results)
        )

        print(
            f"[{i + 1}/{len(df)}] "
            f"GREEDY CER={current_greedy_cer:.4f} "
            f"BEAM CER={current_beam_cer:.4f} "
            f"GREEDY WER={current_greedy_wer:.4f} "
            f"BEAM WER={current_beam_wer:.4f}"
        )
# ============================================================
# FINAL METRICS
# ============================================================

result_df = pd.DataFrame(results)

mean_greedy_cer = result_df["greedy_cer"].mean()
mean_greedy_wer = result_df["greedy_wer"].mean()

mean_beam_cer = result_df["beam_cer"].mean()
mean_beam_wer = result_df["beam_wer"].mean()

mean_loss = total_loss / len(result_df)


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)

summary = {
    "model": "FINAL_B2_MODEL.pt",
    "decoder": "pyctcdecode_ctc_beam_search_unigram",
    "beam_width": 100,
    "unigram_count": len(unigrams),
    "validation_samples": len(result_df),
    "validation_loss": mean_loss,
    "greedy_cer": mean_greedy_cer,
    "greedy_wer": mean_greedy_wer,
    "beam_cer": mean_beam_cer,
    "beam_wer": mean_beam_wer,
    "beam_minus_greedy_cer": mean_beam_cer - mean_greedy_cer,
    "beam_minus_greedy_wer": mean_beam_wer - mean_greedy_wer,
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


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)

print(f"Validation samples : {len(result_df)}")
print(f"Validation loss    : {mean_loss:.6f}")

print()
print("GREEDY CTC")
print(f"  CER : {mean_greedy_cer * 100:.4f}%")
print(f"  WER : {mean_greedy_wer * 100:.4f}%")

print()
print("UNIGRAM BEAM SEARCH")
print(f"  CER : {mean_beam_cer * 100:.4f}%")
print(f"  WER : {mean_beam_wer * 100:.4f}%")

print()
print("CHANGE: BEAM - GREEDY")

print(
    f"  CER : "
    f"{(mean_beam_cer - mean_greedy_cer) * 100:+.4f} pp"
)

print(
    f"  WER : "
    f"{(mean_beam_wer - mean_greedy_wer) * 100:+.4f} pp"
)

print()
print(f"[SAVED] {OUTPUT_CSV}")
print(f"[SAVED] {OUTPUT_JSON}")
print()
print("[DONE]")