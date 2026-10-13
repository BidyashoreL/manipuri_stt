import json
import unicodedata
import warnings
import numpy as np

from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# CONFIGURATION
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

OUTPUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
)

OUTPUT_PATH = OUTPUT_DIR / "train_5fold_reranker_v2_results.json"

RANDOM_STATE = 42
N_SPLITS = 5

HIDDEN_LAYERS = (64, 32)

MAX_ITER = 1000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = " ".join(text.split())
    return text.strip()


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_distance(a, b):
    """
    Standard Levenshtein edit distance.
    Works with lists of characters or words.
    """

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))

    for i, x in enumerate(a, start=1):

        current = [i]

        for j, y in enumerate(b, start=1):

            substitution_cost = 0 if x == y else 1

            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + substitution_cost
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# CER / WER
# ============================================================

def calc_cer(reference, hypothesis):

    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    ref_chars = list(reference)
    hyp_chars = list(hypothesis)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0

    return (
        edit_distance(ref_chars, hyp_chars)
        / len(ref_chars)
    )


def calc_wer(reference, hypothesis):

    reference = normalize_text(reference)
    hypothesis = normalize_text(hypothesis)

    ref_words = reference.split()
    hyp_words = hypothesis.split()

    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0

    return (
        edit_distance(ref_words, hyp_words)
        / len(ref_words)
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(hypothesis):
    """
    Numerical features used by the MLP reranker.

    Features:
        0  raw acoustic/logit score
        1  acoustic score per character
        2  raw LM score
        3  LM score per character
        4  LM score per word
        5  original beam rank
        6  character length
        7  word count
        8  average word length
    """

    text = normalize_text(
        hypothesis.get("text", "")
    )

    logit_score = float(
        hypothesis.get("logit_score", 0.0)
    )

    lm_score = float(
        hypothesis.get("lm_score", 0.0)
    )

    rank = float(
        hypothesis.get("rank", 1.0)
    )

    char_len = len(text)

    words = text.split()

    word_count = len(words)

    avg_word_len = (
        char_len / max(1, word_count)
    )

    logit_per_char = (
        logit_score / max(1, char_len)
    )

    lm_per_char = (
        lm_score / max(1, char_len)
    )

    lm_per_word = (
        lm_score / max(1, word_count)
    )

    return np.asarray(
        [
            logit_score,
            logit_per_char,
            lm_score,
            lm_per_char,
            lm_per_word,
            rank,
            char_len,
            word_count,
            avg_word_len,
        ],
        dtype=np.float64
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("5-FOLD MLP N-BEST RERANKER — V2")
print("=" * 70)

print("\nN-best file:")
print(NBEST_PATH)

if not NBEST_PATH.exists():
    raise FileNotFoundError(
        f"\nN-best file not found:\n{NBEST_PATH}"
    )


with open(
    NBEST_PATH,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


if isinstance(data, dict):
    samples = data["samples"]
else:
    samples = data


print(
    f"\nLoaded utterances: {len(samples)}"
)

if len(samples) != 864:
    print(
        "\nWARNING: Expected 864 development utterances."
    )
    print(
        f"Found: {len(samples)}"
    )


# ============================================================
# PREPARE SAMPLE-LEVEL DATA
# ============================================================

sample_records = []

total_candidates = 0

for sample_id, sample in enumerate(samples):

    reference = normalize_text(
        sample.get(
            "reference",
            sample.get("text", "")
        )
    )

    hypotheses = sample.get(
        "hypotheses",
        sample.get("nbest", [])
    )

    if len(hypotheses) == 0:
        raise ValueError(
            f"Sample {sample_id} contains no hypotheses."
        )

    candidate_records = []

    for candidate_id, hypothesis in enumerate(hypotheses):

        text = normalize_text(
            hypothesis.get("text", "")
        )

        candidate_cer = calc_cer(
            reference,
            text
        )

        candidate_wer = calc_wer(
            reference,
            text
        )

        features = extract_features(
            hypothesis
        )

        candidate_records.append(
            {
                "candidate_id": candidate_id,
                "text": text,
                "features": features,
                "cer": candidate_cer,
                "wer": candidate_wer,
                "rank": float(
                    hypothesis.get(
                        "rank",
                        candidate_id + 1
                    )
                ),
                "source": hypothesis.get(
                    "source",
                    "unknown"
                ),
            }
        )

    sample_records.append(
        {
            "sample_id": sample_id,
            "reference": reference,
            "candidates": candidate_records,
        }
    )

    total_candidates += len(candidate_records)


print(
    f"Total candidates: {total_candidates:,}"
)

print(
    f"Average candidates/utterance: "
    f"{total_candidates / len(sample_records):.2f}"
)


# ============================================================
# 5-FOLD UTTERANCE-LEVEL CROSS VALIDATION
# ============================================================

sample_ids = np.arange(
    len(sample_records)
)

kf = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


fold_results = []

all_baseline_cers = []
all_baseline_wers = []

all_reranked_cers = []
all_reranked_wers = []

all_oracle_cers = []
all_oracle_wers = []

all_selected_ranks = []

changed_selection_count = 0
total_evaluated_samples = 0


print("\n" + "=" * 70)
print("STARTING 5-FOLD OUT-OF-FOLD RERANKING")
print("=" * 70)


for fold, (train_ids, test_ids) in enumerate(
    kf.split(sample_ids),
    start=1
):

    print(
        f"\n{'-' * 70}"
    )

    print(
        f"FOLD {fold}/{N_SPLITS}"
    )

    print(
        f"Training utterances: {len(train_ids)}"
    )

    print(
        f"Test utterances:     {len(test_ids)}"
    )

    # --------------------------------------------------------
    # Build training candidate matrix
    # --------------------------------------------------------

    X_train_list = []
    y_train_list = []

    for sample_id in train_ids:

        record = sample_records[sample_id]

        for candidate in record["candidates"]:

            X_train_list.append(
                candidate["features"]
            )

            y_train_list.append(
                candidate["cer"]
            )

    X_train = np.asarray(
        X_train_list,
        dtype=np.float64
    )

    y_train = np.asarray(
        y_train_list,
        dtype=np.float64
    )

    print(
        f"Training candidates: {len(X_train):,}"
    )

    # --------------------------------------------------------
    # Scale using training fold ONLY
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    # --------------------------------------------------------
    # Train MLP
    # --------------------------------------------------------

    mlp = MLPRegressor(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        learning_rate_init=0.001,
        alpha=0.0001,
        batch_size="auto",
        shuffle=True,
        tol=1e-4,
    )

    with warnings.catch_warnings():

        warnings.filterwarnings(
            "ignore",
            category=ConvergenceWarning
        )

        mlp.fit(
            X_train_scaled,
            y_train
        )

    print(
        f"MLP iterations: {mlp.n_iter_}"
    )

    # --------------------------------------------------------
    # Evaluate each held-out utterance
    # --------------------------------------------------------

    fold_baseline_cers = []
    fold_baseline_wers = []

    fold_reranked_cers = []
    fold_reranked_wers = []

    fold_oracle_cers = []
    fold_oracle_wers = []

    fold_selected_ranks = []

    fold_changed = 0

    for sample_id in test_ids:

        record = sample_records[sample_id]

        reference = record["reference"]

        candidates = record["candidates"]

        # ----------------------------------------------------
        # Candidate feature matrix
        # ----------------------------------------------------

        X_test = np.asarray(
            [
                candidate["features"]
                for candidate in candidates
            ],
            dtype=np.float64
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        # ----------------------------------------------------
        # Predict CER for every candidate
        # ----------------------------------------------------

        predicted_cers = mlp.predict(
            X_test_scaled
        )

        # ----------------------------------------------------
        # 1. BASELINE
        # ----------------------------------------------------

        baseline_index = 0

        baseline_candidate = candidates[
            baseline_index
        ]

        baseline_cer = calc_cer(
            reference,
            baseline_candidate["text"]
        )

        baseline_wer = calc_wer(
            reference,
            baseline_candidate["text"]
        )

        # ----------------------------------------------------
        # 2. MLP RERANKING
        # ----------------------------------------------------

        reranked_index = int(
            np.argmin(predicted_cers)
        )

        reranked_candidate = candidates[
            reranked_index
        ]

        reranked_cer = calc_cer(
            reference,
            reranked_candidate["text"]
        )

        reranked_wer = calc_wer(
            reference,
            reranked_candidate["text"]
        )

        selected_rank = int(
            reranked_candidate["rank"]
        )

        # ----------------------------------------------------
        # 3. ORACLE CER
        # ----------------------------------------------------

        oracle_cer_index = int(
            np.argmin(
                [
                    candidate["cer"]
                    for candidate in candidates
                ]
            )
        )

        oracle_cer_candidate = candidates[
            oracle_cer_index
        ]

        oracle_cer_value = calc_cer(
            reference,
            oracle_cer_candidate["text"]
        )

        # ----------------------------------------------------
        # 4. ORACLE WER
        # ----------------------------------------------------

        oracle_wer_index = int(
            np.argmin(
                [
                    candidate["wer"]
                    for candidate in candidates
                ]
            )
        )

        oracle_wer_candidate = candidates[
            oracle_wer_index
        ]

        oracle_wer_value = calc_wer(
            reference,
            oracle_wer_candidate["text"]
        )

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        fold_baseline_cers.append(
            baseline_cer
        )

        fold_baseline_wers.append(
            baseline_wer
        )

        fold_reranked_cers.append(
            reranked_cer
        )

        fold_reranked_wers.append(
            reranked_wer
        )

        fold_oracle_cers.append(
            oracle_cer_value
        )

        fold_oracle_wers.append(
            oracle_wer_value
        )

        fold_selected_ranks.append(
            selected_rank
        )

        # ----------------------------------------------------
        # Did reranker change the candidate?
        # ----------------------------------------------------

        if reranked_index != baseline_index:

            fold_changed += 1

        # ----------------------------------------------------
        # Global accumulation
        # ----------------------------------------------------

        all_baseline_cers.append(
            baseline_cer
        )

        all_baseline_wers.append(
            baseline_wer
        )

        all_reranked_cers.append(
            reranked_cer
        )

        all_reranked_wers.append(
            reranked_wer
        )

        all_oracle_cers.append(
            oracle_cer_value
        )

        all_oracle_wers.append(
            oracle_wer_value
        )

        all_selected_ranks.append(
            selected_rank
        )

        total_evaluated_samples += 1

    # --------------------------------------------------------
    # Fold metrics
    # --------------------------------------------------------

    fold_baseline_cer = (
        np.mean(fold_baseline_cers) * 100
    )

    fold_baseline_wer = (
        np.mean(fold_baseline_wers) * 100
    )

    fold_reranked_cer = (
        np.mean(fold_reranked_cers) * 100
    )

    fold_reranked_wer = (
        np.mean(fold_reranked_wers) * 100
    )

    fold_oracle_cer = (
        np.mean(fold_oracle_cers) * 100
    )

    fold_oracle_wer = (
        np.mean(fold_oracle_wers) * 100
    )

    fold_rank = np.mean(
        fold_selected_ranks
    )

    fold_cer_gain = (
        fold_baseline_cer
        - fold_reranked_cer
    )

    fold_wer_gain = (
        fold_baseline_wer
        - fold_reranked_wer
    )

    fold_change_rate = (
        fold_changed / len(test_ids) * 100
    )

    fold_result = {
        "fold": fold,
        "train_utterances": int(
            len(train_ids)
        ),
        "test_utterances": int(
            len(test_ids)
        ),
        "training_candidates": int(
            len(X_train)
        ),
        "baseline_cer_percent": fold_baseline_cer,
        "reranked_cer_percent": fold_reranked_cer,
        "oracle_cer_percent": fold_oracle_cer,
        "baseline_wer_percent": fold_baseline_wer,
        "reranked_wer_percent": fold_reranked_wer,
        "oracle_wer_percent": fold_oracle_wer,
        "cer_reduction_pp": fold_cer_gain,
        "wer_reduction_pp": fold_wer_gain,
        "average_selected_rank": fold_rank,
        "selection_change_rate_percent": fold_change_rate,
        "mlp_iterations": int(
            mlp.n_iter_
        ),
    }

    fold_results.append(
        fold_result
    )

    print(
        f"\nFold {fold} results:"
    )

    print(
        f"  Baseline CER : "
        f"{fold_baseline_cer:.4f}%"
    )

    print(
        f"  MLP CER      : "
        f"{fold_reranked_cer:.4f}%"
    )

    print(
        f"  Oracle CER   : "
        f"{fold_oracle_cer:.4f}%"
    )

    print(
        f"  Baseline WER : "
        f"{fold_baseline_wer:.4f}%"
    )

    print(
        f"  MLP WER      : "
        f"{fold_reranked_wer:.4f}%"
    )

    print(
        f"  Oracle WER   : "
        f"{fold_oracle_wer:.4f}%"
    )

    print(
        f"  CER reduction: "
        f"{fold_cer_gain:+.4f} pp"
    )

    print(
        f"  WER reduction: "
        f"{fold_wer_gain:+.4f} pp"
    )

    print(
        f"  Avg selected rank: "
        f"{fold_rank:.2f}"
    )

    print(
        f"  Selection changed: "
        f"{fold_change_rate:.2f}%"
    )


# ============================================================
# OVERALL OUT-OF-FOLD RESULTS
# ============================================================

overall_baseline_cer = (
    np.mean(all_baseline_cers) * 100
)

overall_reranked_cer = (
    np.mean(all_reranked_cers) * 100
)

overall_oracle_cer = (
    np.mean(all_oracle_cers) * 100
)

overall_baseline_wer = (
    np.mean(all_baseline_wers) * 100
)

overall_reranked_wer = (
    np.mean(all_reranked_wers) * 100
)

overall_oracle_wer = (
    np.mean(all_oracle_wers) * 100
)

overall_cer_reduction = (
    overall_baseline_cer
    - overall_reranked_cer
)

overall_wer_reduction = (
    overall_baseline_wer
    - overall_reranked_wer
)

average_selected_rank = np.mean(
    all_selected_ranks
)

selection_change_rate = (
    changed_selection_count
    / max(1, total_evaluated_samples)
    * 100
)


# ============================================================
# CORRECT GLOBAL CHANGE RATE
# ============================================================

# Recalculate from fold results because each fold stored
# its number of changed selections as a percentage.
#
# Weighted calculation gives the exact overall percentage.

total_changed = 0

for fold_result in fold_results:

    total_changed += round(
        fold_result[
            "selection_change_rate_percent"
        ]
        / 100
        * fold_result[
            "test_utterances"
        ]
    )

selection_change_rate = (
    total_changed
    / total_evaluated_samples
    * 100
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD MLP RERANKING — OUT-OF-FOLD RESULTS")
print("=" * 70)

print("\nCER")
print("-" * 70)

print(
    f"Baseline Top-1 : "
    f"{overall_baseline_cer:.4f}%"
)

print(
    f"MLP Reranked   : "
    f"{overall_reranked_cer:.4f}%"
)

print(
    f"Oracle         : "
    f"{overall_oracle_cer:.4f}%"
)

print(
    f"MLP reduction  : "
    f"{overall_cer_reduction:+.4f} pp"
)

print("\nWER")
print("-" * 70)

print(
    f"Baseline Top-1 : "
    f"{overall_baseline_wer:.4f}%"
)

print(
    f"MLP Reranked   : "
    f"{overall_reranked_wer:.4f}%"
)

print(
    f"Oracle         : "
    f"{overall_oracle_wer:.4f}%"
)

print(
    f"MLP reduction  : "
    f"{overall_wer_reduction:+.4f} pp"
)

print("\nReranker behavior")
print("-" * 70)

print(
    f"Average selected rank : "
    f"{average_selected_rank:.2f}"
)

print(
    f"Selection changed     : "
    f"{selection_change_rate:.2f}%"
)

print(
    f"Samples evaluated     : "
    f"{total_evaluated_samples}"
)

print("\n" + "=" * 70)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "experiment": "5-fold MLP N-best reranking v2",

    "candidate_file": str(
        NBEST_PATH
    ),

    "num_samples": len(
        sample_records
    ),

    "num_candidates": int(
        total_candidates
    ),

    "average_candidates_per_sample": (
        total_candidates
        / len(sample_records)
    ),

    "cross_validation": {
        "method": "KFold",
        "n_splits": N_SPLITS,
        "shuffle": True,
        "random_state": RANDOM_STATE,
        "split_unit": "utterance",
    },

    "model": {
        "type": "sklearn MLPRegressor",
        "hidden_layers": list(
            HIDDEN_LAYERS
        ),
        "activation": "relu",
        "solver": "adam",
        "max_iter": MAX_ITER,
        "random_state": RANDOM_STATE,
        "target": "candidate_CER",
    },

    "features": [
        "logit_score",
        "logit_score_per_character",
        "lm_score",
        "lm_score_per_character",
        "lm_score_per_word",
        "original_rank",
        "character_length",
        "word_count",
        "average_word_length",
    ],

    "metric_definition": {
        "cer": "macro-average of utterance-level CER",
        "wer": "macro-average of utterance-level WER",
    },

    "overall": {
        "baseline_cer_percent": overall_baseline_cer,
        "reranked_cer_percent": overall_reranked_cer,
        "oracle_cer_percent": overall_oracle_cer,
        "cer_reduction_pp": overall_cer_reduction,

        "baseline_wer_percent": overall_baseline_wer,
        "reranked_wer_percent": overall_reranked_wer,
        "oracle_wer_percent": overall_oracle_wer,
        "wer_reduction_pp": overall_wer_reduction,

        "average_selected_rank": average_selected_rank,
        "selection_change_rate_percent": selection_change_rate,
    },

    "fold_results": fold_results,
}


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\nSaved results:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)