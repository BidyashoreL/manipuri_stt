from pathlib import Path
import json
import unicodedata
import csv
import numpy as np
import torch
import torch.nn as nn


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

MODEL_PATH = LM_DIR / "char_gru_lm_best.pth"
VOCAB_PATH = LM_DIR / "char_gru_vocab.json"
META_PATH = LM_DIR / "char_gru_metadata.json"

OUTPUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "error_analysis"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_JSON = OUTPUT_DIR / "sample_level_analysis.json"
OUTPUT_CSV = OUTPUT_DIR / "sample_level_analysis.csv"


# ============================================================
# CONFIG
# ============================================================

TOP_K = 20
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BETA = 3.0

EMBED_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize(
        "NFC",
        str(text)
    )
    return " ".join(text.split())


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_distance(a, b):

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):

        current = [i]

        for j, cb in enumerate(b, start=1):

            cost = 0 if ca == cb else 1

            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost
                )
            )

        previous = current

    return previous[-1]


def sample_cer(reference, prediction):

    reference = normalize_text(reference)
    prediction = normalize_text(prediction)

    return (
        edit_distance(
            list(reference),
            list(prediction)
        )
        / max(len(reference), 1)
    )


def sample_wer(reference, prediction):

    reference = normalize_text(reference)
    prediction = normalize_text(prediction)

    return (
        edit_distance(
            reference.split(),
            prediction.split()
        )
        / max(len(reference.split()), 1)
    )


# ============================================================
# GRU MODEL
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
            vocab_size,
            embed_dim,
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

    def forward(self, x):

        x = self.embedding(x)

        x, _ = self.gru(x)

        return self.fc(x)


# ============================================================
# LOAD VOCAB
# ============================================================

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8"
) as f:

    vocab_data = json.load(f)

stoi = vocab_data["stoi"]

PAD_ID = int(vocab_data["pad_id"])
UNK_ID = int(vocab_data["unk_id"])
BOS_ID = int(vocab_data["bos_id"])
EOS_ID = int(vocab_data["eos_id"])

VOCAB_SIZE = len(stoi)


# ============================================================
# LOAD MODEL
# ============================================================

model = CharGRU(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT,
    pad_id=PAD_ID
).to(DEVICE)


state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(state_dict, dict)
    and "model_state_dict" in state_dict
):

    state_dict = state_dict["model_state_dict"]

model.load_state_dict(
    state_dict,
    strict=True
)

model.eval()


# ============================================================
# LOAD N-BEST
# ============================================================

with open(
    NBEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    nbest_data = json.load(f)

samples = nbest_data["samples"]


if len(samples) != 864:

    raise RuntimeError(
        f"Expected 864 validation samples, "
        f"found {len(samples)}"
    )


# ============================================================
# CHARACTER ENCODING
# ============================================================

def candidate_ids(text):

    text = normalize_text(text)

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
# GRU SCORING
# EXACTLY MATCHES ORIGINAL RESCORER
# ============================================================

@torch.no_grad()
def score_candidates(texts):

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

    for i, sequence in enumerate(sequences):

        input_ids[
            i,
            :len(sequence)
        ] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=DEVICE
        )

    logits = model(input_ids)

    log_probs = torch.log_softmax(
        logits,
        dim=-1
    )

    scores = []

    for i, sequence in enumerate(sequences):

        sequence_length = len(sequence)

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

        return np.zeros_like(values)

    return (
        values - mean
    ) / std


# ============================================================
# PROCESS
# ============================================================

results = []

print("=" * 72)
print("SAMPLE-LEVEL GRU ERROR ANALYSIS")
print("=" * 72)

print(f"Device : {DEVICE}")
print(f"Samples: {len(samples)}")
print(f"Top-K  : {TOP_K}")
print(f"Beta   : {BETA}")


for sample_number, sample in enumerate(
    samples,
    start=1
):

    reference = normalize_text(
        sample["reference"]
    )

    all_hypotheses = sample["hypotheses"]

    top20 = all_hypotheses[:TOP_K]

    # --------------------------------------------------------
    # GRU scores
    # --------------------------------------------------------

    texts = [
        h["text"]
        for h in top20
    ]

    neural_scores = score_candidates(
        texts
    )

    # --------------------------------------------------------
    # BASE SCORE
    # EXACT ORIGINAL FORMULA
    # --------------------------------------------------------

    base_scores = np.asarray(
        [
            float(h["logit_score"])
            + 0.5 * float(h["lm_score"])
            for h in top20
        ],
        dtype=np.float64
    )

    neural_scores = np.asarray(
        neural_scores,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # FUSION
    # EXACT ORIGINAL FORMULA
    # --------------------------------------------------------

    base_z = zscore(
        base_scores
    )

    neural_z = zscore(
        neural_scores
    )

    combined_scores = (
        base_z
        + BETA * neural_z
    )

    best_index = int(
        np.argmax(
            combined_scores
        )
    )

    selected = top20[best_index]

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline = top20[0]

    baseline_text = normalize_text(
        baseline["text"]
    )

    # --------------------------------------------------------
    # GRU PREDICTION
    # --------------------------------------------------------

    gru_text = normalize_text(
        selected["text"]
    )

    gru_rank = int(
        selected["original_rank"]
    )

    # --------------------------------------------------------
    # BEST CER/WER IN TOP-20
    # --------------------------------------------------------

    best20_cer_h = min(
        top20,
        key=lambda h:
            sample_cer(
                reference,
                h["text"]
            )
    )

    best20_wer_h = min(
        top20,
        key=lambda h:
            sample_wer(
                reference,
                h["text"]
            )
    )

    # --------------------------------------------------------
    # BEST CER/WER IN TOP-100
    # --------------------------------------------------------

    best100_cer_h = min(
        all_hypotheses,
        key=lambda h:
            sample_cer(
                reference,
                h["text"]
            )
    )

    best100_wer_h = min(
        all_hypotheses,
        key=lambda h:
            sample_wer(
                reference,
                h["text"]
            )
    )

    # --------------------------------------------------------
    # EXACT REFERENCE RANK
    # --------------------------------------------------------

    exact_rank = None

    for h in all_hypotheses:

        if (
            normalize_text(h["text"])
            == reference
        ):

            exact_rank = int(
                h["original_rank"]
            )

            break

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    baseline_cer = sample_cer(
        reference,
        baseline_text
    )

    baseline_wer = sample_wer(
        reference,
        baseline_text
    )

    gru_cer = sample_cer(
        reference,
        gru_text
    )

    gru_wer = sample_wer(
        reference,
        gru_text
    )

    oracle20_cer = sample_cer(
        reference,
        best20_cer_h["text"]
    )

    oracle20_wer = sample_wer(
        reference,
        best20_wer_h["text"]
    )

    oracle100_cer = sample_cer(
        reference,
        best100_cer_h["text"]
    )

    oracle100_wer = sample_wer(
        reference,
        best100_wer_h["text"]
    )

    # --------------------------------------------------------
    # CANDIDATE CATEGORY
    # --------------------------------------------------------

    if exact_rank is None:

        candidate_category = (
            "ABSENT_TOP100"
        )

    elif exact_rank <= 20:

        candidate_category = (
            "EXACT_TOP20"
        )

    elif exact_rank <= 50:

        candidate_category = (
            "EXACT_21_50"
        )

    else:

        candidate_category = (
            "EXACT_51_100"
        )

    # --------------------------------------------------------
    # GRU EFFECT
    # --------------------------------------------------------

    if gru_cer < baseline_cer:

        gru_effect = "IMPROVED"

    elif gru_cer > baseline_cer:

        gru_effect = "WORSENED"

    else:

        gru_effect = "UNCHANGED"

    # --------------------------------------------------------
    # SELECTION ERROR
    # --------------------------------------------------------

    correct_top20_but_not_selected = (
        exact_rank is not None
        and exact_rank <= 20
        and gru_text != reference
    )

    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    result = {

        "sample_index":
            int(sample["sample_index"]),

        "reference":
            reference,

        "baseline_4gram_top1":
            baseline_text,

        "gru_beta3":
            gru_text,

        "gru_selected_rank":
            gru_rank,

        "exact_reference_rank":
            exact_rank,

        "candidate_category":
            candidate_category,

        "gru_effect":
            gru_effect,

        "correct_top20_but_not_selected":
            correct_top20_but_not_selected,

        "baseline_cer":
            baseline_cer,

        "gru_cer":
            gru_cer,

        "delta_cer":
            gru_cer - baseline_cer,

        "baseline_wer":
            baseline_wer,

        "gru_wer":
            gru_wer,

        "delta_wer":
            gru_wer - baseline_wer,

        "oracle20_cer":
            oracle20_cer,

        "oracle20_wer":
            oracle20_wer,

        "oracle20_cer_rank":
            int(
                best20_cer_h[
                    "original_rank"
                ]
            ),

        "oracle20_wer_rank":
            int(
                best20_wer_h[
                    "original_rank"
                ]
            ),

        "oracle20_cer_text":
            normalize_text(
                best20_cer_h["text"]
            ),

        "oracle20_wer_text":
            normalize_text(
                best20_wer_h["text"]
            ),

        "oracle100_cer":
            oracle100_cer,

        "oracle100_wer":
            oracle100_wer,

        "oracle100_cer_rank":
            int(
                best100_cer_h[
                    "original_rank"
                ]
            ),

        "oracle100_wer_rank":
            int(
                best100_wer_h[
                    "original_rank"
                ]
            ),

        "oracle100_cer_text":
            normalize_text(
                best100_cer_h["text"]
            ),

        "oracle100_wer_text":
            normalize_text(
                best100_wer_h["text"]
            ),

        "gru_neural_score":
            float(
                neural_scores[
                    best_index
                ]
            ),

        "selected_combined_score":
            float(
                combined_scores[
                    best_index
                ]
            ),
    }

    results.append(result)

    if (
        sample_number == 1
        or sample_number % 100 == 0
        or sample_number == len(samples)
    ):

        print(
            f"Processed "
            f"{sample_number}/{len(samples)}"
        )


# ============================================================
# SUMMARY
# ============================================================

def count_category(category):

    return sum(
        r["candidate_category"]
        == category
        for r in results
    )


def count_effect(effect):

    return sum(
        r["gru_effect"]
        == effect
        for r in results
    )


summary = {

    "samples":
        len(results),

    "candidate_pool": {

        "exact_top20":
            count_category(
                "EXACT_TOP20"
            ),

        "exact_21_50":
            count_category(
                "EXACT_21_50"
            ),

        "exact_51_100":
            count_category(
                "EXACT_51_100"
            ),

        "absent_top100":
            count_category(
                "ABSENT_TOP100"
            ),
    },

    "gru_effect": {

        "improved":
            count_effect(
                "IMPROVED"
            ),

        "worsened":
            count_effect(
                "WORSENED"
            ),

        "unchanged":
            count_effect(
                "UNCHANGED"
            ),
    },

    "correct_top20_but_not_selected":
        sum(
            r[
                "correct_top20_but_not_selected"
            ]
            for r in results
        ),
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "summary": summary,
            "samples": results
        },
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    encoding="utf-8-sig",
    newline=""
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=list(
            results[0].keys()
        )
    )

    writer.writeheader()

    writer.writerows(results)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 72)
print("CANDIDATE POOL")
print("=" * 72)

print(
    f"Exact reference in Top-20 : "
    f"{summary['candidate_pool']['exact_top20']}"
)

print(
    f"Exact reference in 21-50  : "
    f"{summary['candidate_pool']['exact_21_50']}"
)

print(
    f"Exact reference in 51-100 : "
    f"{summary['candidate_pool']['exact_51_100']}"
)

print(
    f"Reference absent Top-100  : "
    f"{summary['candidate_pool']['absent_top100']}"
)


print()
print("=" * 72)
print("GRU EFFECT")
print("=" * 72)

print(
    f"Improved  : "
    f"{summary['gru_effect']['improved']}"
)

print(
    f"Worsened  : "
    f"{summary['gru_effect']['worsened']}"
)

print(
    f"Unchanged : "
    f"{summary['gru_effect']['unchanged']}"
)


print()
print(
    "Correct answer in Top-20 "
    "but GRU did not select it:"
)

print(
    summary[
        "correct_top20_but_not_selected"
    ]
)


# ============================================================
# WORST SAMPLES
# ============================================================

print()
print("=" * 72)
print("20 WORST GRU CER SAMPLES")
print("=" * 72)

worst = sorted(
    results,
    key=lambda r:
        r["gru_cer"],
    reverse=True
)[:20]


for r in worst:

    print()
    print(
        f"[{r['sample_index']}] "
        f"rank={r['gru_selected_rank']} "
        f"category={r['candidate_category']}"
    )

    print(
        "REF :",
        r["reference"]
    )

    print(
        "4G  :",
        r["baseline_4gram_top1"]
    )

    print(
        "GRU :",
        r["gru_beta3"]
    )

    print(
        f"CER : "
        f"{r['baseline_cer']*100:.2f}%"
        f" -> "
        f"{r['gru_cer']*100:.2f}%"
    )


print()
print("=" * 72)
print("20 BIGGEST GRU IMPROVEMENTS")
print("=" * 72)

best = sorted(
    results,
    key=lambda r:
        r["delta_cer"]
)[:20]


for r in best:

    print()
    print(
        f"[{r['sample_index']}] "
        f"rank={r['gru_selected_rank']}"
    )

    print(
        "REF :",
        r["reference"]
    )

    print(
        "4G  :",
        r["baseline_4gram_top1"]
    )

    print(
        "GRU :",
        r["gru_beta3"]
    )

    print(
        f"CER : "
        f"{r['baseline_cer']*100:.2f}%"
        f" -> "
        f"{r['gru_cer']*100:.2f}%"
    )


print()
print("=" * 72)
print("ANALYSIS COMPLETE")
print("=" * 72)

print(
    "\nJSON:",
    OUTPUT_JSON
)

print(
    "CSV :",
    OUTPUT_CSV
)