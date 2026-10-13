# ============================================================
# 05_tune_beam_width_val.py
#
# B2 + UNIGRAM CTC BEAM WIDTH ABLATION
#
# Validation ONLY
# Official 410-sample test set is NEVER touched.
#
# Beam widths:
#   50
#   100
#   200
#   500
#
# Neural inference is performed ONCE.
# The same logits are decoded using every beam width.
# ============================================================

import sys
import json
import time
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from pyctcdecode import build_ctcdecoder


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent

MODEL_PATH = (
    ASR_DIR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "regularized"
    / "FINAL_B2_MODEL.pt"
)

VAL_CSV = (
    ASR_DIR
    / "combined_dataset"
    / "val.csv"
)

VAL_CACHE = (
    ASR_DIR
    / "val_audio_cache"
)

VOCAB_PATH = (
    ASR_DIR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
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
    / "beam_width_sweep_results.csv"
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "beam_width_sweep_summary.json"
)

PROCESSOR_PATH = r"E:\hf_models\w2v-bert-2.0"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("B2 + UNIGRAM CTC BEAM WIDTH ABLATION")
print("=" * 70)

print(f"[INFO] Device: {DEVICE}")

if torch.cuda.is_available():
    print(
        f"[INFO] GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# CHECK FILES
# ============================================================

required_paths = [
    MODEL_PATH,
    VAL_CSV,
    VAL_CACHE,
    VOCAB_PATH,
    UNIGRAM_PATH,
]

for path in required_paths:

    if not path.exists():

        raise FileNotFoundError(
            f"Required path not found:\n{path}"
        )


# ============================================================
# IMPORT EXACT B2 MODEL
# ============================================================
#
# This is the exact import used by your working
# 04_validate_unigram_beam.py.
#
# ============================================================

sys.path.insert(
    0,
    str(ASR_DIR)
)

from train_b2_partial_encoder import B2Model


# ============================================================
# IMPORT NORMALIZATION + METRICS
# ============================================================
#
# Your working 04 script already uses normalize_text(),
# cer(), and wer().
#
# Import them from the same module if available.
#
# ============================================================

try:

    from train_b2_partial_encoder import (
        normalize_text,
        cer,
        wer,
    )

except ImportError:

    print(
        "[INFO] Metric/normalization functions not exported "
        "by train_b2_partial_encoder."
    )

    print(
        "[INFO] Defining compatible local functions."
    )

    def normalize_text(text):

        text = str(text)

        text = " ".join(
            text.split()
        )

        return text.strip()


    def edit_distance(ref, hyp):

        n = len(ref)
        m = len(hyp)

        prev = list(range(m + 1))

        for i in range(1, n + 1):

            curr = [i]

            for j in range(1, m + 1):

                cost = (
                    0
                    if ref[i - 1] == hyp[j - 1]
                    else 1
                )

                curr.append(
                    min(
                        prev[j] + 1,
                        curr[j - 1] + 1,
                        prev[j - 1] + cost,
                    )
                )

            prev = curr

        return prev[m]


    def cer(ref, hyp):

        if len(ref) == 0:

            return (
                0.0
                if len(hyp) == 0
                else 1.0
            )

        return (
            edit_distance(
                list(ref),
                list(hyp),
            )
            / len(ref)
        )


    def wer(ref, hyp):

        ref_words = ref.split()
        hyp_words = hyp.split()

        if len(ref_words) == 0:

            return (
                0.0
                if len(hyp_words) == 0
                else 1.0
            )

        return (
            edit_distance(
                ref_words,
                hyp_words,
            )
            / len(ref_words)
        )


# ============================================================
# LOAD VOCABULARY
# ============================================================

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8",
) as f:

    vocab = json.load(f)


id_to_token = {
    idx: token
    for token, idx in vocab.items()
}


print(
    f"[INFO] Vocabulary size: "
    f"{len(vocab)}"
)


# ============================================================
# BUILD DECODER LABELS
# ============================================================

labels = [
    token
    for token, idx in sorted(
        vocab.items(),
        key=lambda x: x[1],
    )
]


labels_for_decoder = []

for label in labels:

    if label == "<blank>":

        labels_for_decoder.append("")

    else:

        labels_for_decoder.append(label)


# Ensure uniqueness
if len(labels_for_decoder) != len(
    set(labels_for_decoder)
):

    raise RuntimeError(
        "[ERROR] Duplicate decoder labels detected."
    )


print(
    "[PASS] Decoder labels validated."
)


# ============================================================
# LOAD UNIGRAMS
# ============================================================

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
    f"[INFO] Unigrams: "
    f"{len(unigrams):,}"
)


# ============================================================
# BUILD DECODER
# ============================================================

print(
    "[INFO] Building pyctcdecode decoder..."
)

decoder = build_ctcdecoder(
    labels=labels_for_decoder,
    unigrams=unigrams,
)

print(
    "[PASS] Decoder constructed."
)


# ============================================================
# LOAD B2 MODEL
# ============================================================

print(
    "[INFO] Loading FINAL_B2_MODEL.pt..."
)

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

    state_dict = checkpoint[
        "model_state_dict"
    ]

elif (
    isinstance(checkpoint, dict)
    and "state_dict" in checkpoint
):

    state_dict = checkpoint[
        "state_dict"
    ]

else:

    state_dict = checkpoint


model.load_state_dict(
    state_dict,
    strict=True,
)

model.to(DEVICE)

model.eval()

print(
    "[PASS] Model loaded."
)


# ============================================================
# LOAD PROCESSOR
# ============================================================

from transformers import AutoFeatureExtractor


print(
    "[INFO] Loading processor..."
)

processor = AutoFeatureExtractor.from_pretrained(
    PROCESSOR_PATH
)

print(
    "[PASS] Processor loaded."
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

df = pd.read_csv(
    VAL_CSV
)


if len(df) != 864:

    raise RuntimeError(
        f"Expected 864 validation samples, "
        f"got {len(df)}"
    )


print(
    f"[INFO] Validation samples: "
    f"{len(df)}"
)


# ============================================================
# GREEDY DECODER
# ============================================================

blank_id = vocab[
    "<blank>"
]

unk_id = vocab[
    "<unk>"
]


def greedy_decode(logits):

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

        # ----------------------------------------------------
        # BLANK
        # ----------------------------------------------------

        if token_id == blank_id:

            previous = token_id

            continue


        # ----------------------------------------------------
        # CTC REPEAT COLLAPSE
        # ----------------------------------------------------

        if token_id == previous:

            continue


        # ----------------------------------------------------
        # UNK
        # ----------------------------------------------------

        if token_id == unk_id:

            previous = token_id

            continue


        # ----------------------------------------------------
        # TOKEN
        # ----------------------------------------------------

        output.append(
            id_to_token[token_id]
        )

        previous = token_id


    return "".join(output)


# ============================================================
# STEP 1:
# RUN MODEL INFERENCE ONCE
# ============================================================

print()
print("=" * 70)
print("STEP 1/2: EXTRACTING VALIDATION LOGITS")
print("=" * 70)

print(
    "[INFO] Neural inference will run exactly once."
)

print(
    "[INFO] The resulting logits will be reused "
    "for all beam widths."
)


# ------------------------------------------------------------
# Store CPU log-probability matrices.
# ------------------------------------------------------------

log_probs_list = []

references = []

total_loss = 0.0

start_time = time.time()


for i in range(len(df)):

    # ========================================================
    # LOAD AUDIO CACHE
    # ========================================================

    cache_path = (
        VAL_CACHE
        / f"{i:06d}.pt"
    )


    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=True,
    )


    audio = payload[
        "audio"
    ].float()


    # ========================================================
    # REFERENCE
    # ========================================================

    reference = normalize_text(
        df.iloc[i]["text"]
    )

    references.append(
        reference
    )


    # ========================================================
    # AUDIO SHAPE
    # ========================================================

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


    audio_np = np.asarray(
        audio_np,
        dtype=np.float32,
    ).reshape(-1)


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )


    if "input_features" in inputs:

        input_features = (
            inputs[
                "input_features"
            ]
            .to(
                DEVICE,
                dtype=torch.float32,
            )
        )

    elif "input_values" in inputs:

        input_features = (
            inputs[
                "input_values"
            ]
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


    # ========================================================
    # MODEL
    # ========================================================

    with torch.no_grad():

        logits = model(
            input_features
        )


        log_probs = F.log_softmax(
            logits,
            dim=-1,
        )


    # ========================================================
    # SAVE CPU LOG PROBS
    # ========================================================

    decoder_input = (
        log_probs[0]
        .detach()
        .float()
        .cpu()
        .numpy()
    )


    log_probs_list.append(
        decoder_input
    )


    # ========================================================
    # LOSS
    # ========================================================

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
        log_probs.transpose(
            0,
            1,
        )
    )


    loss = F.ctc_loss(
        log_probs_ctc,
        target_tensor,
        input_lengths,
        target_lengths,
        blank=blank_id,
        zero_infinity=True,
    )


    total_loss += float(
        loss.item()
    )


    # ========================================================
    # CLEANUP
    # ========================================================

    del (
        payload,
        audio,
        audio_np,
        inputs,
        input_features,
        logits,
        log_probs,
        decoder_input,
        target_tensor,
        target_lengths,
        input_lengths,
        log_probs_ctc,
        loss,
    )


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        (i + 1) % 25 == 0
        or i == len(df) - 1
    ):

        elapsed = (
            time.time()
            - start_time
        )

        print(
            f"[{i + 1}/{len(df)}] "
            f"Elapsed: {elapsed / 60:.2f} min"
        )


# ============================================================
# INFERENCE COMPLETE
# ============================================================

mean_loss = (
    total_loss
    / len(df)
)


print()
print(
    "[PASS] All validation logits extracted."
)

print(
    f"[INFO] Validation loss: "
    f"{mean_loss:.6f}"
)


# ============================================================
# STEP 2:
# GREEDY BASELINE
# ============================================================

print()
print("=" * 70)
print("STEP 2/2: DECODER ABLATION")
print("=" * 70)


print(
    "[INFO] Calculating greedy baseline..."
)


greedy_predictions = []


for log_probs in log_probs_list:

    # Convert log-probabilities back into
    # a tensor shaped [1, T, V].

    logits_tensor = torch.from_numpy(
        log_probs
    ).unsqueeze(0)


    prediction = greedy_decode(
        logits_tensor
    )


    prediction = normalize_text(
        prediction
    )


    greedy_predictions.append(
        prediction
    )


# ============================================================
# GREEDY METRICS
# ============================================================

greedy_cer_values = []

greedy_wer_values = []


for reference, hypothesis in zip(
    references,
    greedy_predictions,
):

    greedy_cer_values.append(
        cer(
            reference,
            hypothesis,
        )
    )

    greedy_wer_values.append(
        wer(
            reference,
            hypothesis,
        )
    )


mean_greedy_cer = (
    sum(greedy_cer_values)
    / len(greedy_cer_values)
)

mean_greedy_wer = (
    sum(greedy_wer_values)
    / len(greedy_wer_values)
)


print()
print(
    "GREEDY BASELINE"
)

print(
    f"  CER : "
    f"{mean_greedy_cer * 100:.4f}%"
)

print(
    f"  WER : "
    f"{mean_greedy_wer * 100:.4f}%"
)


# ============================================================
# BEAM WIDTHS
# ============================================================

BEAM_WIDTHS = [
    50,
    100,
    200,
    500,
]


sweep_results = []


# ============================================================
# BEAM SWEEP
# ============================================================

for beam_width in BEAM_WIDTHS:

    print()
    print("=" * 70)

    print(
        f"UNIGRAM BEAM WIDTH = "
        f"{beam_width}"
    )

    print("=" * 70)


    beam_predictions = []

    beam_start = time.time()


    for i, log_probs in enumerate(
        log_probs_list
    ):

        hypothesis = decoder.decode(
            log_probs,
            beam_width=beam_width,
        )


        hypothesis = normalize_text(
            hypothesis
        )


        beam_predictions.append(
            hypothesis
        )


        if (
            (i + 1) % 50 == 0
            or i == len(log_probs_list) - 1
        ):

            elapsed = (
                time.time()
                - beam_start
            )

            print(
                f"[{i + 1}/{len(log_probs_list)}] "
                f"Decode elapsed: "
                f"{elapsed / 60:.2f} min"
            )


    # ========================================================
    # METRICS
    # ========================================================

    beam_cer_values = []

    beam_wer_values = []


    for reference, hypothesis in zip(
        references,
        beam_predictions,
    ):

        beam_cer_values.append(
            cer(
                reference,
                hypothesis,
            )
        )

        beam_wer_values.append(
            wer(
                reference,
                hypothesis,
            )
        )


    mean_beam_cer = (
        sum(beam_cer_values)
        / len(beam_cer_values)
    )

    mean_beam_wer = (
        sum(beam_wer_values)
        / len(beam_wer_values)
    )


    cer_change = (
        mean_beam_cer
        - mean_greedy_cer
    ) * 100


    wer_change = (
        mean_beam_wer
        - mean_greedy_wer
    ) * 100


    decode_time = (
        time.time()
        - beam_start
    ) / 60


    # ========================================================
    # STORE RESULT
    # ========================================================

    sweep_results.append(
        {
            "decoder":
                "unigram_beam",

            "beam_width":
                beam_width,

            "validation_samples":
                len(df),

            "validation_loss":
                round(
                    mean_loss,
                    6,
                ),

            "cer_percent":
                round(
                    mean_beam_cer * 100,
                    4,
                ),

            "wer_percent":
                round(
                    mean_beam_wer * 100,
                    4,
                ),

            "cer_change_vs_greedy_pp":
                round(
                    cer_change,
                    4,
                ),

            "wer_change_vs_greedy_pp":
                round(
                    wer_change,
                    4,
                ),

            "decode_time_minutes":
                round(
                    decode_time,
                    3,
                ),
        }
    )


    # ========================================================
    # PRINT RESULT
    # ========================================================

    print()
    print(
        f"Beam {beam_width}"
    )

    print(
        f"  CER : "
        f"{mean_beam_cer * 100:.4f}%"
    )

    print(
        f"  WER : "
        f"{mean_beam_wer * 100:.4f}%"
    )

    print(
        f"  CER change vs greedy : "
        f"{cer_change:+.4f} pp"
    )

    print(
        f"  WER change vs greedy : "
        f"{wer_change:+.4f} pp"
    )

    print(
        f"  Decode time : "
        f"{decode_time:.2f} min"
    )

    del beam_predictions

    gc.collect()


# ============================================================
# ADD GREEDY ROW
# ============================================================

all_results = [

    {
        "decoder":
            "greedy_ctc",

        "beam_width":
            None,

        "validation_samples":
            len(df),

        "validation_loss":
            round(
                mean_loss,
                6,
            ),

        "cer_percent":
            round(
                mean_greedy_cer * 100,
                4,
            ),

        "wer_percent":
            round(
                mean_greedy_wer * 100,
                4,
            ),

        "cer_change_vs_greedy_pp":
            0.0,

        "wer_change_vs_greedy_pp":
            0.0,

        "decode_time_minutes":
            None,
    }

]


all_results.extend(
    sweep_results
)


results_df = pd.DataFrame(
    all_results
)


# ============================================================
# FIND BEST BEAM
# ============================================================

beam_df = results_df[
    results_df[
        "decoder"
    ] == "unigram_beam"
]


best_position = beam_df[
    "cer_percent"
].idxmin()


best_row = results_df.loc[
    best_position
]


best_beam = int(
    best_row[
        "beam_width"
    ]
)


best_cer = float(
    best_row[
        "cer_percent"
    ]
)


best_wer = float(
    best_row[
        "wer_percent"
    ]
)


# ============================================================
# FINAL TABLE
# ============================================================

print()
print("=" * 70)
print("FINAL BEAM WIDTH ABLATION")
print("=" * 70)


print(
    results_df[
        [
            "decoder",
            "beam_width",
            "cer_percent",
            "wer_percent",
            "cer_change_vs_greedy_pp",
            "wer_change_vs_greedy_pp",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BEST CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("BEST VALIDATION CONFIGURATION")
print("=" * 70)

print(
    f"Decoder      : "
    f"Unigram-Assisted CTC Beam Search"
)

print(
    f"Beam width   : "
    f"{best_beam}"
)

print(
    f"Val CER      : "
    f"{best_cer:.4f}%"
)

print(
    f"Val WER      : "
    f"{best_wer:.4f}%"
)

print(
    f"Greedy CER   : "
    f"{mean_greedy_cer * 100:.4f}%"
)

print(
    f"Greedy WER   : "
    f"{mean_greedy_wer * 100:.4f}%"
)


print(
    f"CER improvement: "
    f"{(mean_greedy_cer * 100) - best_cer:.4f} pp"
)

print(
    f"WER improvement: "
    f"{(mean_greedy_wer * 100) - best_wer:.4f} pp"
)


# ============================================================
# SAVE CSV
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


results_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig",
)


print()
print(
    f"[SAVED] {OUTPUT_CSV}"
)


# ============================================================
# SAVE JSON
# ============================================================

summary = {

    "experiment":
        "B2 unigram-assisted CTC beam width ablation",

    "model":
        "FINAL_B2_MODEL.pt",

    "decoder":
        "pyctcdecode_ctc_beam_search_unigram",

    "validation_samples":
        len(df),

    "official_test_samples":
        410,

    "official_test_evaluated":
        False,

    "vocabulary_size":
        len(vocab),

    "unigram_count":
        len(unigrams),

    "beam_widths_tested":
        BEAM_WIDTHS,

    "greedy": {

        "cer":
            mean_greedy_cer,

        "wer":
            mean_greedy_wer,

    },

    "best_validation_configuration": {

        "beam_width":
            best_beam,

        "cer":
            best_cer / 100.0,

        "wer":
            best_wer / 100.0,

    },

    "results":
        sweep_results,

    "selection_criterion":
        "minimum validation CER",

    "official_test_status":
        "UNTOUCHED",
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
    f"[SAVED] {OUTPUT_JSON}"
)


# ============================================================
# CLEANUP
# ============================================================

del log_probs_list

gc.collect()

if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# DONE
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print()
print(
    "Official 410-sample test set: "
    "NOT TOUCHED"
)

print(
    f"Best validation beam width: "
    f"{best_beam}"
)

print()
print(
    "Next step is to freeze this decoder "
    "and evaluate it ONCE on the official "
    "410-sample test set."
)

print()