from pathlib import Path
import json
import unicodedata

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# FINAL CHARACTER-LEVEL GRU N-BEST RESCORING
#
# B3-NARROW
#     +
# 4-GRAM KENLM POOLED N-BEST
#     +
# CHARACTER-LEVEL GRU LANGUAGE MODEL
#
# VALIDATION SET ONLY
#
# Official 410-sample test is NOT touched.
# ============================================================


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent


NBEST_PATH = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "b3_4gram_pooled100_val.json"
)


LM_DIR = (
    SCRIPT_DIR
    / "checkpoints"
    / "lm_gru"
)


MODEL_PATH = (
    LM_DIR
    / "char_gru_lm_best.pth"
)


VOCAB_PATH = (
    LM_DIR
    / "char_gru_vocab.json"
)


META_PATH = (
    LM_DIR
    / "char_gru_metadata.json"
)


OUTPUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "gru_lm"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


OUTPUT_PATH = (
    OUTPUT_DIR
    / "gru_lm_rescoring_verification.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 20

DEFAULT_EMBED_DIM = 64
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2

BATCH_SIZE = 256

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# FUSION WEIGHTS
# ============================================================

BETAS = [
    0.0,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.80,
    1.00,
    1.50,
    2.00,
    3.00,
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = unicodedata.normalize(
        "NFC",
        str(text)
    )

    text = " ".join(
        text.split()
    )

    return text


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_distance(a, b):

    previous = list(
        range(len(b) + 1)
    )

    for i, ca in enumerate(
        a,
        start=1
    ):

        current = [i]

        for j, cb in enumerate(
            b,
            start=1
        ):

            if ca == cb:
                cost = 0
            else:
                cost = 1

            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# CER / WER
# ============================================================

def corpus_metrics(
    predictions,
    references
):

    char_errors = 0
    char_total = 0

    word_errors = 0
    word_total = 0

    for prediction, reference in zip(
        predictions,
        references
    ):

        prediction = normalize_text(
            prediction
        )

        reference = normalize_text(
            reference
        )

        # Character error
        char_errors += edit_distance(
            list(prediction),
            list(reference)
        )

        char_total += len(reference)

        # Word error
        prediction_words = prediction.split()
        reference_words = reference.split()

        word_errors += edit_distance(
            prediction_words,
            reference_words
        )

        word_total += len(
            reference_words
        )

    cer = (
        100.0
        * char_errors
        / max(char_total, 1)
    )

    wer = (
        100.0
        * word_errors
        / max(word_total, 1)
    )

    return cer, wer


# ============================================================
# CHARACTER GRU
# ============================================================

class CharGRU(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
        pad_id
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_id
        )

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            )
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(
        self,
        x
    ):

        x = self.embedding(x)

        x, _ = self.gru(x)

        logits = self.fc(x)

        return logits


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("FINAL CHARACTER GRU N-BEST RESCORING")
print("=" * 72)

print(
    f"Device : {DEVICE}"
)

print(
    f"N-best : {NBEST_PATH}"
)

print(
    f"GRU    : {MODEL_PATH}"
)

print(
    f"Vocab  : {VOCAB_PATH}"
)


# ============================================================
# CHECK FILES
# ============================================================

required_files = [
    NBEST_PATH,
    MODEL_PATH,
    VOCAB_PATH,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            "\nRequired file does not exist:\n"
            f"{path}"
        )


print()
print("All required files found.")


# ============================================================
# LOAD GRU VOCABULARY
# ============================================================

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as f:

    vocab_data = json.load(f)


stoi = vocab_data["stoi"]


PAD_ID = int(
    vocab_data["pad_id"]
)

UNK_ID = int(
    vocab_data["unk_id"]
)

BOS_ID = int(
    vocab_data["bos_id"]
)

EOS_ID = int(
    vocab_data["eos_id"]
)

VOCAB_SIZE = len(stoi)


print()
print("VOCABULARY")
print("-" * 72)

print(
    f"Vocabulary size : {VOCAB_SIZE}"
)

print(
    f"PAD ID          : {PAD_ID}"
)

print(
    f"UNK ID          : {UNK_ID}"
)

print(
    f"BOS ID          : {BOS_ID}"
)

print(
    f"EOS ID          : {EOS_ID}"
)


# ============================================================
# LOAD GRU METADATA
# ============================================================

EMBED_DIM = DEFAULT_EMBED_DIM
HIDDEN_DIM = DEFAULT_HIDDEN_DIM
NUM_LAYERS = DEFAULT_NUM_LAYERS
DROPOUT = DEFAULT_DROPOUT


if META_PATH.exists():

    with open(
        META_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        metadata = json.load(f)

    EMBED_DIM = int(
        metadata.get(
            "embedding_dim",
            DEFAULT_EMBED_DIM
        )
    )

    HIDDEN_DIM = int(
        metadata.get(
            "hidden_dim",
            DEFAULT_HIDDEN_DIM
        )
    )

    NUM_LAYERS = int(
        metadata.get(
            "num_layers",
            DEFAULT_NUM_LAYERS
        )
    )

    DROPOUT = float(
        metadata.get(
            "dropout",
            DEFAULT_DROPOUT
        )
    )


print()
print("GRU ARCHITECTURE")
print("-" * 72)

print(
    f"Embedding : {EMBED_DIM}"
)

print(
    f"Hidden    : {HIDDEN_DIM}"
)

print(
    f"Layers    : {NUM_LAYERS}"
)

print(
    f"Dropout   : {DROPOUT}"
)


# ============================================================
# BUILD MODEL
# ============================================================

model = CharGRU(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pad_id=PAD_ID
).to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


if (
    isinstance(state_dict, dict)
    and "model_state_dict" in state_dict
):

    state_dict = state_dict[
        "model_state_dict"
    ]


model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()


total_params = sum(
    p.numel()
    for p in model.parameters()
)


print()
print(
    f"GRU parameters : "
    f"{total_params:,}"
)

print(
    "Checkpoint loaded successfully."
)


# ============================================================
# LOAD POOLED N-BEST
# ============================================================

with open(
    NBEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    nbest_data = json.load(f)


samples = nbest_data["samples"]


print()
print("N-BEST DATA")
print("-" * 72)

print(
    f"Validation samples : "
    f"{len(samples):,}"
)

print(
    f"Top-K candidates   : "
    f"{TOP_K}"
)


# ============================================================
# VALIDATE ACTUAL JSON STRUCTURE
#
# Actual pooled JSON:
#
# sample
#   sample_index
#   reference
#   hypotheses
#
# hypothesis
#   source
#   original_rank
#   text
#   lm_score
#   logit_score
# ============================================================

for sample in samples:

    if "reference" not in sample:

        raise KeyError(
            "Sample is missing 'reference'."
        )

    if "hypotheses" not in sample:

        raise KeyError(
            "Sample is missing 'hypotheses'."
        )

    if not sample["hypotheses"]:

        raise ValueError(
            "Sample contains zero hypotheses."
        )


# ============================================================
# TEXT â†’ CHARACTER IDS
# ============================================================

def candidate_ids(text):

    text = normalize_text(
        text
    )

    character_ids = [
        stoi.get(
            character,
            UNK_ID
        )
        for character in text
    ]

    return (
        [BOS_ID]
        + character_ids
        + [EOS_ID]
    )


# ============================================================
# BATCH GRU SCORING
# ============================================================

@torch.no_grad()
def score_candidates_batch(
    texts
):

    sequences = [
        candidate_ids(text)
        for text in texts
    ]

    max_len = max(
        len(sequence)
        for sequence in sequences
    )

    input_ids = torch.full(
        (
            len(sequences),
            max_len
        ),
        PAD_ID,
        dtype=torch.long,
        device=DEVICE
    )

    for i, sequence in enumerate(
        sequences
    ):

        input_ids[
            i,
            :len(sequence)
        ] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=DEVICE
        )

    logits = model(
        input_ids
    )

    log_probs = torch.log_softmax(
        logits,
        dim=-1
    )

    scores = []

    for i, sequence in enumerate(
        sequences
    ):

        sequence_length = len(
            sequence
        )

        if sequence_length < 2:

            scores.append(
                0.0
            )

            continue

        target = input_ids[
            i,
            1:sequence_length
        ]

        token_log_probs = log_probs[
            i,
            :sequence_length - 1
        ].gather(
            1,
            target.unsqueeze(1)
        ).squeeze(1)

        # Average log probability.
        #
        # This reduces the strong length bias
        # of raw summed log probability.

        average_log_prob = (
            token_log_probs.mean()
        )

        scores.append(
            float(
                average_log_prob.item()
            )
        )

    return scores


# ============================================================
# SCORE TOP-K HYPOTHESES
# ============================================================

print()
print("=" * 72)
print("SCORING TOP-K HYPOTHESES WITH GRU")
print("=" * 72)


for sample_index, sample in enumerate(
    samples,
    start=1
):

    hypotheses = sample[
        "hypotheses"
    ][:TOP_K]


    texts = [
        hypothesis["text"]
        for hypothesis in hypotheses
    ]


    neural_scores = []


    for start in range(
        0,
        len(texts),
        BATCH_SIZE
    ):

        batch_texts = texts[
            start:start + BATCH_SIZE
        ]

        batch_scores = (
            score_candidates_batch(
                batch_texts
            )
        )

        neural_scores.extend(
            batch_scores
        )


    if len(neural_scores) != len(
        hypotheses
    ):

        raise RuntimeError(
            "GRU score count does not match "
            "hypothesis count."
        )


    for hypothesis, score in zip(
        hypotheses,
        neural_scores
    ):

        hypothesis[
            "neural_lm_score"
        ] = float(score)


    if (
        sample_index == 1
        or sample_index % 100 == 0
        or sample_index == len(samples)
    ):

        print(
            f"Scored "
            f"{sample_index:,}/"
            f"{len(samples):,}"
        )


# ============================================================
# REFERENCES
# ============================================================

references = [
    normalize_text(
        sample["reference"]
    )
    for sample in samples
]


# ============================================================
# PRIMARY BASELINE
#
# IMPORTANT:
#
# The primary baseline is the original
# pooled N-best Top-1 hypothesis.
#
# This is the same baseline used by
# the previous LSTM experiment.
# ============================================================

baseline_predictions = []

baseline_ranks = []


for sample in samples:

    hypotheses = sample[
        "hypotheses"
    ]

    top_hypothesis = hypotheses[0]


    baseline_predictions.append(
        normalize_text(
            top_hypothesis["text"]
        )
    )


    baseline_ranks.append(
        int(
            top_hypothesis[
                "original_rank"
            ]
        )
    )


baseline_cer, baseline_wer = (
    corpus_metrics(
        baseline_predictions,
        references
    )
)


print()
print("=" * 72)
print("PRIMARY BASELINE â€” ORIGINAL N-BEST TOP-1")
print("=" * 72)

print(
    f"CER : {baseline_cer:.4f}%"
)

print(
    f"WER : {baseline_wer:.4f}%"
)


# ============================================================
# GRU-ONLY SELECTION
# ============================================================

gru_predictions = []

gru_ranks = []


for sample in samples:

    hypotheses = sample[
        "hypotheses"
    ][:TOP_K]


    best_hypothesis = max(
        hypotheses,
        key=lambda hypothesis:
            hypothesis[
                "neural_lm_score"
            ]
    )


    gru_predictions.append(
        normalize_text(
            best_hypothesis["text"]
        )
    )


    gru_ranks.append(
        int(
            best_hypothesis[
                "original_rank"
            ]
        )
    )


gru_cer, gru_wer = (
    corpus_metrics(
        gru_predictions,
        references
    )
)


print()
print("=" * 72)
print("GRU-ONLY N-BEST SELECTION")
print("=" * 72)

print(
    f"CER : {gru_cer:.4f}%"
)

print(
    f"WER : {gru_wer:.4f}%"
)

print(
    f"Average selected rank : "
    f"{np.mean(gru_ranks):.2f}"
)

print(
    f"Median selected rank  : "
    f"{np.median(gru_ranks):.0f}"
)


# ============================================================
# Z-SCORE
# ============================================================

def zscore(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    mean = values.mean()
    std = values.std()

    if std < 1e-12:

        return np.zeros_like(
            values
        )

    return (
        values - mean
    ) / std


# ============================================================
# GRU + 4-GRAM FUSION
#
# Approximate base score:
#
# logit_score + 0.5 * lm_score
#
# IMPORTANT:
#
# This is a relative diagnostic score.
# It is NOT claimed to exactly reproduce
# pyctcdecode because the original decoder
# also contains beta / insertion effects.
# ============================================================

fusion_results = []


print()
print("=" * 72)
print("GRU + 4-GRAM N-BEST FUSION")
print("=" * 72)


for beta in BETAS:

    predictions = []

    selected_ranks = []

    changed = 0


    for sample in samples:

        hypotheses = sample[
            "hypotheses"
        ][:TOP_K]


        # ----------------------------------------------------
        # Approximate acoustic + 4-gram score
        # ----------------------------------------------------

        base_scores = np.asarray(
            [
                float(
                    hypothesis[
                        "logit_score"
                    ]
                )
                +
                0.5
                *
                float(
                    hypothesis[
                        "lm_score"
                    ]
                )
                for hypothesis in hypotheses
            ],
            dtype=np.float64
        )


        # ----------------------------------------------------
        # GRU score
        # ----------------------------------------------------

        neural_scores = np.asarray(
            [
                float(
                    hypothesis[
                        "neural_lm_score"
                    ]
                )
                for hypothesis in hypotheses
            ],
            dtype=np.float64
        )


        # ----------------------------------------------------
        # Per-utterance normalization
        # ----------------------------------------------------

        base_z = zscore(
            base_scores
        )

        neural_z = zscore(
            neural_scores
        )


        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        combined_scores = (
            base_z
            +
            beta
            *
            neural_z
        )


        best_index = int(
            np.argmax(
                combined_scores
            )
        )


        best_hypothesis = hypotheses[
            best_index
        ]


        predictions.append(
            normalize_text(
                best_hypothesis["text"]
            )
        )


        selected_rank = int(
            best_hypothesis[
                "original_rank"
            ]
        )


        selected_ranks.append(
            selected_rank
        )


        if best_index != 0:

            changed += 1


    cer, wer = corpus_metrics(
        predictions,
        references
    )


    result = {

        "beta":
            float(beta),

        "cer":
            float(cer),

        "wer":
            float(wer),

        "cer_improvement_pp":
            float(
                baseline_cer - cer
            ),

        "wer_improvement_pp":
            float(
                baseline_wer - wer
            ),

        "average_selected_rank":
            float(
                np.mean(
                    selected_ranks
                )
            ),

        "median_selected_rank":
            float(
                np.median(
                    selected_ranks
                )
            ),

        "selection_changed_pct":
            float(
                100.0
                *
                changed
                /
                len(samples)
            )
    }


    fusion_results.append(
        result
    )


    print(
        f"Beta {beta:<4.2f} "
        f"-> "
        f"CER {cer:.4f}% "
        f"WER {wer:.4f}% "
        f"AvgRank "
        f"{result['average_selected_rank']:.2f} "
        f"Changed "
        f"{result['selection_changed_pct']:.2f}%"
    )


# ============================================================
# BEST RESULTS
# ============================================================

best_cer_result = min(
    fusion_results,
    key=lambda result:
        result["cer"]
)


best_wer_result = min(
    fusion_results,
    key=lambda result:
        result["wer"]
)


# ============================================================
# SAVE OUTPUT
# ============================================================

output = {

    "experiment":
        "B3-NARROW + "
        "4-GRAM pooled N-best + "
        "Character GRU LM",

    "device":
        str(DEVICE),

    "num_validation_samples":
        len(samples),

    "top_k":
        TOP_K,

    "gru_model":
        str(MODEL_PATH),

    "gru_vocab":
        str(VOCAB_PATH),

    "baseline": {

        "type":
            "original_pooled_nbest_top1",

        "cer":
            float(baseline_cer),

        "wer":
            float(baseline_wer)
    },

    "gru_only": {

        "cer":
            float(gru_cer),

        "wer":
            float(gru_wer),

        "average_selected_rank":
            float(
                np.mean(
                    gru_ranks
                )
            ),

        "median_selected_rank":
            float(
                np.median(
                    gru_ranks
                )
            )
    },

    "betas":
        BETAS,

    "fusion_results":
        fusion_results,

    "best_cer":
        best_cer_result,

    "best_wer":
        best_wer_result
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SANITY CHECKS
# ============================================================

assert len(samples) == 864

assert all(
    len(
        sample["hypotheses"][:TOP_K]
    ) > 0
    for sample in samples
)

assert np.isfinite(
    baseline_cer
)

assert np.isfinite(
    baseline_wer
)

assert np.isfinite(
    gru_cer
)

assert np.isfinite(
    gru_wer
)

assert all(
    np.isfinite(
        result["cer"]
    )
    and
    np.isfinite(
        result["wer"]
    )
    for result in fusion_results
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 72)
print("BEST GRU RESCORING RESULTS")
print("=" * 72)

print(
    f"Baseline CER : "
    f"{baseline_cer:.4f}%"
)

print(
    f"Baseline WER : "
    f"{baseline_wer:.4f}%"
)

print()

print(
    f"GRU-only CER : "
    f"{gru_cer:.4f}%"
)

print(
    f"GRU-only WER : "
    f"{gru_wer:.4f}%"
)

print()

print(
    f"Best CER beta : "
    f"{best_cer_result['beta']}"
)

print(
    f"Best CER      : "
    f"{best_cer_result['cer']:.4f}%"
)

print(
    f"CER gain      : "
    f"{best_cer_result['cer_improvement_pp']:.4f} pp"
)

print()

print(
    f"Best WER beta : "
    f"{best_wer_result['beta']}"
)

print(
    f"Best WER      : "
    f"{best_wer_result['wer']:.4f}%"
)

print(
    f"WER gain      : "
    f"{best_wer_result['wer_improvement_pp']:.4f} pp"
)

print()
print("=" * 72)
print("SANITY CHECKS")
print("=" * 72)

print(
    "PASS: 864 validation samples"
)

print(
    "PASS: pooled N-best hypotheses loaded"
)

print(
    "PASS: GRU checkpoint loaded"
)

print(
    "PASS: GRU scores generated"
)

print(
    "PASS: CER/WER finite"
)

print(
    "PASS: Official 410-sample test untouched"
)

print()
print("RESULTS SAVED TO:")
print(
    OUTPUT_PATH
)

print()
print("=" * 72)
print("GRU N-BEST RESCORING COMPLETE")
print("=" * 72)
