import json
import unicodedata
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


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

OUTPUT_PATH = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "pairwise_reranker_results.json"
)

SEED = 42

N_FOLDS = 5

EPOCHS = 15
BATCH_SIZE = 256

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MARGIN = 0.1

# Conservative neural interpolation.
LAMBDA_WEIGHTS = [
    0.0,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
]

# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 70)
print("5-FOLD PAIRWISE NEURAL N-BEST RERANKER")
print("FIXED / RESEARCH-CLEAN VERSION")
print("=" * 70)

print(f"Seed       : {SEED}")
print(f"Folds      : {N_FOLDS}")
print(f"Epochs     : {EPOCHS}")
print(f"Batch size : {BATCH_SIZE}")
print(f"Margin     : {MARGIN}")
print(f"Lambdas    : {LAMBDA_WEIGHTS}")
print()


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text):
    """
    NFC normalization + whitespace normalization.
    """
    return " ".join(
        unicodedata.normalize("NFC", str(text)).split()
    ).strip()


def edit_distance(a, b):
    """
    Standard Levenshtein edit distance.
    Works for lists of characters or words.
    """

    prev = list(range(len(b) + 1))

    for i, x in enumerate(a, 1):

        cur = [i]

        for j, y in enumerate(b, 1):

            cost = 0 if x == y else 1

            cur.append(
                min(
                    cur[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + cost,
                )
            )

        prev = cur

    return prev[-1]


def calc_cer(ref, hyp):

    ref = normalize_text(ref)
    hyp = normalize_text(hyp)

    if not ref:
        return 0.0 if not hyp else 1.0

    return (
        edit_distance(
            list(ref),
            list(hyp)
        )
        / len(ref)
    )


def calc_wer(ref, hyp):

    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return (
        edit_distance(
            ref_words,
            hyp_words
        )
        / len(ref_words)
    )


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(hyp):
    """
    Extract absolute candidate-level features.

    Features:
        0  logit score
        1  logit / character
        2  LM score
        3  LM / character
        4  LM / word
        5  rank
        6  character length
        7  word count
        8  average word length
    """

    text = normalize_text(
        hyp.get("text", "")
    )

    logit = float(
        hyp.get("logit_score", 0.0)
    )

    lm = float(
        hyp.get("lm_score", 0.0)
    )

    rank = float(
        hyp.get("rank", 1.0)
    )

    char_len = len(text)

    word_count = len(text.split())

    avg_word_len = (
        char_len / max(1, word_count)
    )

    return np.array(
        [
            logit,
            logit / max(1, char_len),

            lm,
            lm / max(1, char_len),
            lm / max(1, word_count),

            rank,

            char_len,
            word_count,
            avg_word_len,
        ],
        dtype=np.float32,
    )


# ============================================================
# PAIRWISE NEURAL RANKER
# ============================================================

class PairwiseRanker(nn.Module):

    def __init__(self, in_features):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                in_features,
                64
            ),

            nn.SiLU(),

            nn.Dropout(
                0.10
            ),

            nn.Linear(
                64,
                32
            ),

            nn.SiLU(),

            nn.Linear(
                32,
                1
            ),
        )

    def forward(self, x):

        return self.net(x).squeeze(-1)


# ============================================================
# CHECK FILE
# ============================================================

if not NBEST_PATH.exists():

    raise FileNotFoundError(
        f"\nN-best file not found:\n{NBEST_PATH}"
    )

print("N-best file:")
print(NBEST_PATH)
print()


# ============================================================
# LOAD N-BEST DATA
# ============================================================

print("Loading dataset...")

with open(
    NBEST_PATH,
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


samples = (
    data["samples"]
    if isinstance(data, dict)
    else data
)

print(
    f"Loaded utterances: {len(samples)}"
)


# ============================================================
# BUILD DATASET
# ============================================================

dataset = []

total_candidates = 0

for sample_index, sample in enumerate(samples):

    ref = normalize_text(
        sample.get(
            "reference",
            sample.get("text", "")
        )
    )

    hyps = sample.get(
        "hypotheses",
        sample.get("nbest", [])
    )

    if not hyps:
        continue

    # --------------------------------------------------------
    # Top-1 features
    # --------------------------------------------------------

    top1_feat = extract_features(
        hyps[0]
    )

    candidate_features = []

    candidate_cers = []

    candidate_wers = []

    candidate_texts = []

    candidate_ranks = []

    for h in hyps:

        text = normalize_text(
            h.get("text", "")
        )

        feat = extract_features(h)

        # ----------------------------------------------------
        # Relative-to-Top-1 features
        # ----------------------------------------------------

        delta_feat = (
            feat - top1_feat
        )

        combined_feat = np.concatenate(
            [
                feat,
                delta_feat,
            ]
        )

        candidate_features.append(
            combined_feat
        )

        candidate_cers.append(
            calc_cer(
                ref,
                text
            )
        )

        candidate_wers.append(
            calc_wer(
                ref,
                text
            )
        )

        candidate_texts.append(text)

        candidate_ranks.append(
            int(
                h.get(
                    "rank",
                    len(candidate_texts)
                )
            )
        )

    dataset.append(
        {
            "sample_index": sample_index,

            "ref": ref,

            "feats": np.asarray(
                candidate_features,
                dtype=np.float32
            ),

            "cers": np.asarray(
                candidate_cers,
                dtype=np.float32
            ),

            "wers": np.asarray(
                candidate_wers,
                dtype=np.float32
            ),

            "texts": candidate_texts,

            "ranks": candidate_ranks,
        }
    )

    total_candidates += len(hyps)


print(
    f"Total candidates: {total_candidates:,}"
)

print(
    f"Average candidates/utterance: "
    f"{total_candidates / len(dataset):.2f}"
)

print()


# ============================================================
# FOLD SETUP
# ============================================================

kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED,
)

sample_ids = np.arange(
    len(dataset)
)


# ============================================================
# RESULT STORAGE
# ============================================================

fold_results = {
    lam: {
        "cer_errors": 0,
        "wer_errors": 0,
        "selected_ranks": [],
        "selection_changed": 0,
        "num_samples": 0,
    }
    for lam in LAMBDA_WEIGHTS
}


# Exact baseline totals
baseline_cer_errors = 0
baseline_wer_errors = 0

total_chars = 0
total_words = 0


# Per-sample OOF predictions
oof_predictions = []


# ============================================================
# START CROSS-VALIDATION
# ============================================================

print("=" * 70)
print("STARTING 5-FOLD OUT-OF-FOLD PAIRWISE RERANKING")
print("=" * 70)
print()


for fold, (train_idx, test_idx) in enumerate(
    kf.split(sample_ids),
    start=1
):

    print("-" * 70)

    print(
        f"FOLD {fold}/{N_FOLDS}"
    )

    print(
        f"Training utterances: {len(train_idx)}"
    )

    print(
        f"Test utterances:     {len(test_idx)}"
    )

    # ========================================================
    # FIT FEATURE SCALER ONLY ON TRAINING UTTERANCES
    # ========================================================

    train_feats = np.vstack(
        [
            dataset[i]["feats"]
            for i in train_idx
        ]
    )

    scaler = StandardScaler()

    scaler.fit(
        train_feats
    )

    # ========================================================
    # GENERATE PAIRWISE TRAINING DATA
    # ========================================================

    pairs_x1 = []
    pairs_x2 = []

    pair_count = 0

    for i in train_idx:

        sample = dataset[i]

        feats_scaled = scaler.transform(
            sample["feats"]
        )

        cers = sample["cers"]

        n_candidates = len(cers)

        for a in range(
            n_candidates
        ):

            for b in range(
                a + 1,
                n_candidates
            ):

                # Candidate a is better
                if cers[a] < cers[b]:

                    pairs_x1.append(
                        feats_scaled[a]
                    )

                    pairs_x2.append(
                        feats_scaled[b]
                    )

                    pair_count += 1

                # Candidate b is better
                elif cers[b] < cers[a]:

                    pairs_x1.append(
                        feats_scaled[b]
                    )

                    pairs_x2.append(
                        feats_scaled[a]
                    )

                    pair_count += 1

                # Equal CER:
                # skip because there is no ranking preference.

    print(
        f"Training pairs: {pair_count:,}"
    )

    if pair_count == 0:

        raise RuntimeError(
            "No pairwise training pairs were generated."
        )

    # ========================================================
    # CONVERT TO TORCH
    # ========================================================

    x1_tensor = torch.tensor(
        np.asarray(
            pairs_x1,
            dtype=np.float32
        )
    )

    x2_tensor = torch.tensor(
        np.asarray(
            pairs_x2,
            dtype=np.float32
        )
    )

    # Free Python lists
    del pairs_x1
    del pairs_x2

    in_dim = train_feats.shape[1]

    # ========================================================
    # CREATE MODEL
    # ========================================================

    model = PairwiseRanker(
        in_dim
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = nn.MarginRankingLoss(
        margin=MARGIN
    )

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    y_target = torch.ones(
        BATCH_SIZE,
        dtype=torch.float32
    )

    last_loss = None

    for epoch in range(
        EPOCHS
    ):

        permutation = torch.randperm(
            len(x1_tensor)
        )

        epoch_loss = 0.0

        batches = 0

        for start in range(
            0,
            len(x1_tensor),
            BATCH_SIZE
        ):

            idx = permutation[
                start:
                start + BATCH_SIZE
            ]

            # Skip incomplete final batch
            if len(idx) < BATCH_SIZE:
                continue

            batch_x1 = x1_tensor[
                idx
            ]

            batch_x2 = x2_tensor[
                idx
            ]

            s1 = model(
                batch_x1
            )

            s2 = model(
                batch_x2
            )

            loss = loss_fn(
                s1,
                s2,
                y_target
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

            batches += 1

        if batches > 0:

            last_loss = (
                epoch_loss / batches
            )

    print(
        f"Final training loss: "
        f"{last_loss:.6f}"
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    model.eval()

    with torch.no_grad():

        for i in test_idx:

            sample = dataset[i]

            ref = sample["ref"]

            top1_text = sample["texts"][0]

            # ------------------------------------------------
            # CORRECT DENOMINATORS
            # ------------------------------------------------

            ref_char_count = len(ref)

            ref_word_count = len(
                ref.split()
            )

            total_chars += (
                ref_char_count
            )

            total_words += (
                ref_word_count
            )

            # ------------------------------------------------
            # EXACT ORIGINAL TOP-1 BASELINE
            # ------------------------------------------------

            baseline_char_errors = edit_distance(
                list(ref),
                list(top1_text)
            )

            baseline_word_errors = edit_distance(
                ref.split(),
                top1_text.split()
            )

            baseline_cer_errors += (
                baseline_char_errors
            )

            baseline_wer_errors += (
                baseline_word_errors
            )

            # ------------------------------------------------
            # SCALE TEST FEATURES
            # ------------------------------------------------

            scaled_features = scaler.transform(
                sample["feats"]
            )

            feature_tensor = torch.tensor(
                scaled_features,
                dtype=torch.float32
            )

            neural_scores = (
                model(
                    feature_tensor
                )
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # RANK PRIOR
            #
            # rank 1 -> 0
            # rank 2 -> -log(2)
            # rank 3 -> -log(3)
            # ...
            #
            # This gives the original Top-1 a strong prior.
            # ------------------------------------------------

            n_candidates = len(
                sample["texts"]
            )

            candidate_ranks = np.arange(
                1,
                n_candidates + 1,
                dtype=np.float32
            )

            rank_prior = -np.log(
                candidate_ranks
            )

            # ------------------------------------------------
            # STORE OOF SAMPLE
            # ------------------------------------------------

            sample_output = {
                "sample_index": int(
                    sample["sample_index"]
                ),

                "fold": int(fold),

                "reference": ref,

                "baseline_text": top1_text,

                "baseline_cer": float(
                    calc_cer(
                        ref,
                        top1_text
                    )
                ),

                "baseline_wer": float(
                    calc_wer(
                        ref,
                        top1_text
                    )
                ),

                "lambda_results": {},
            }

            # =================================================
            # LAMBDA EVALUATION
            # =================================================

            for lam in LAMBDA_WEIGHTS:

                # ------------------------------------------------
                # IMPORTANT:
                #
                # lambda = 0 is forced to EXACT Top-1.
                #
                # This guarantees our baseline is exactly the
                # original decoder output.
                # ------------------------------------------------

                if lam == 0.0:

                    best_idx = 0

                else:

                    final_scores = (
                        rank_prior
                        + lam * neural_scores
                    )

                    best_idx = int(
                        np.argmax(
                            final_scores
                        )
                    )

                selected_text = sample[
                    "texts"
                ][best_idx]

                selected_rank = (
                    best_idx + 1
                )

                selected_cer_errors = edit_distance(
                    list(ref),
                    list(selected_text)
                )

                selected_wer_errors = edit_distance(
                    ref.split(),
                    selected_text.split()
                )

                # ------------------------------------------------
                # ACCUMULATE ERRORS
                # ------------------------------------------------

                fold_results[lam][
                    "cer_errors"
                ] += selected_cer_errors

                fold_results[lam][
                    "wer_errors"
                ] += selected_wer_errors

                fold_results[lam][
                    "selected_ranks"
                ].append(
                    selected_rank
                )

                fold_results[lam][
                    "num_samples"
                ] += 1

                if selected_rank != 1:

                    fold_results[lam][
                        "selection_changed"
                    ] += 1

                # ------------------------------------------------
                # SAVE SAMPLE RESULT
                # ------------------------------------------------

                sample_output[
                    "lambda_results"
                ][str(lam)] = {

                    "selected_index": int(
                        best_idx
                    ),

                    "selected_rank": int(
                        selected_rank
                    ),

                    "selected_text": selected_text,

                    "cer": float(
                        calc_cer(
                            ref,
                            selected_text
                        )
                    ),

                    "wer": float(
                        calc_wer(
                            ref,
                            selected_text
                        )
                    ),

                    "neural_score": float(
                        neural_scores[
                            best_idx
                        ]
                    ),

                    "rank_prior": float(
                        rank_prior[
                            best_idx
                        ]
                    ),

                    "final_score": float(
                        (
                            rank_prior[
                                best_idx
                            ]
                            +
                            lam
                            *
                            neural_scores[
                                best_idx
                            ]
                        )
                    ),
                }

            oof_predictions.append(
                sample_output
            )

    print(
        f"Fold {fold}/{N_FOLDS} finished."
    )

    print()


# ============================================================
# FINAL METRICS
# ============================================================

print()
print("=" * 70)
print("ANCHORED PAIRWISE RERANKING RESULTS")
print("=" * 70)

# ------------------------------------------------------------
# Exact baseline
# ------------------------------------------------------------

baseline_cer = (
    100.0
    * baseline_cer_errors
    / total_chars
)

baseline_wer = (
    100.0
    * baseline_wer_errors
    / total_words
)

print()

print("BASELINE")
print("-" * 70)

print(
    f"Original Top-1 CER : "
    f"{baseline_cer:.4f}%"
)

print(
    f"Original Top-1 WER : "
    f"{baseline_wer:.4f}%"
)

print(
    f"Total reference characters : "
    f"{total_chars:,}"
)

print(
    f"Total reference words      : "
    f"{total_words:,}"
)

print()

# ------------------------------------------------------------
# Lambda results
# ------------------------------------------------------------

print("RERANKER RESULTS")
print("-" * 70)

final_results = {}

for lam in LAMBDA_WEIGHTS:

    result = fold_results[lam]

    cer = (
        100.0
        * result["cer_errors"]
        / total_chars
    )

    wer = (
        100.0
        * result["wer_errors"]
        / total_words
    )

    avg_rank = float(
        np.mean(
            result["selected_ranks"]
        )
    )

    median_rank = float(
        np.median(
            result["selected_ranks"]
        )
    )

    selection_change_rate = (
        100.0
        * result["selection_changed"]
        / result["num_samples"]
    )

    cer_delta = (
        cer
        - baseline_cer
    )

    wer_delta = (
        wer
        - baseline_wer
    )

    print(
        f"Lambda {lam:>4.2f} -> "
        f"CER: {cer:.4f}% | "
        f"WER: {wer:.4f}% | "
        f"Avg Rank: {avg_rank:.2f} | "
        f"Median Rank: {median_rank:.1f} | "
        f"Changed: {selection_change_rate:.2f}%"
    )

    final_results[str(lam)] = {

        "cer": float(cer),

        "wer": float(wer),

        "cer_delta_pp": float(
            cer_delta
        ),

        "wer_delta_pp": float(
            wer_delta
        ),

        "average_selected_rank": float(
            avg_rank
        ),

        "median_selected_rank": float(
            median_rank
        ),

        "selection_change_rate_percent": float(
            selection_change_rate
        ),

        "num_samples": int(
            result["num_samples"]
        ),
    }


# ============================================================
# FIND BEST RESULTS
# ============================================================

best_cer_lambda = min(
    final_results.keys(),
    key=lambda x:
        final_results[x]["cer"]
)

best_wer_lambda = min(
    final_results.keys(),
    key=lambda x:
        final_results[x]["wer"]
)

print()
print("=" * 70)
print("BEST RESULTS")
print("=" * 70)

print(
    f"Best CER lambda : "
    f"{best_cer_lambda}"
)

print(
    f"Best CER        : "
    f"{final_results[best_cer_lambda]['cer']:.4f}%"
)

print(
    f"CER improvement : "
    f"{-final_results[best_cer_lambda]['cer_delta_pp']:.4f} pp"
)

print()

print(
    f"Best WER lambda : "
    f"{best_wer_lambda}"
)

print(
    f"Best WER        : "
    f"{final_results[best_wer_lambda]['wer']:.4f}%"
)

print(
    f"WER improvement : "
    f"{-final_results[best_wer_lambda]['wer_delta_pp']:.4f} pp"
)

print()


# ============================================================
# SANITY CHECKS
# ============================================================

print("=" * 70)
print("SANITY CHECKS")
print("=" * 70)

# Check 1:
# Lambda 0 MUST be exact baseline.

lambda0 = final_results["0.0"]

baseline_match_cer = np.isclose(
    lambda0["cer"],
    baseline_cer,
    atol=1e-9
)

baseline_match_wer = np.isclose(
    lambda0["wer"],
    baseline_wer,
    atol=1e-9
)

lambda0_rank_ok = np.isclose(
    lambda0["average_selected_rank"],
    1.0,
    atol=1e-9
)

print(
    f"[CHECK] Lambda 0 CER matches baseline: "
    f"{'PASS' if baseline_match_cer else 'FAIL'}"
)

print(
    f"[CHECK] Lambda 0 WER matches baseline: "
    f"{'PASS' if baseline_match_wer else 'FAIL'}"
)

print(
    f"[CHECK] Lambda 0 average rank = 1: "
    f"{'PASS' if lambda0_rank_ok else 'FAIL'}"
)

if not (
    baseline_match_cer
    and baseline_match_wer
    and lambda0_rank_ok
):

    print()
    print(
        "WARNING: Lambda 0 sanity check failed."
    )

else:

    print()
    print(
        "All baseline sanity checks passed."
    )


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "experiment": {
        "name": (
            "5-fold pairwise neural "
            "N-best reranker"
        ),

        "version": "fixed_v1",

        "seed": SEED,

        "n_folds": N_FOLDS,

        "epochs": EPOCHS,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "margin": MARGIN,

        "lambda_weights": LAMBDA_WEIGHTS,

        "candidate_file": str(
            NBEST_PATH
        ),
    },

    "dataset": {
        "utterances": len(dataset),

        "total_candidates": total_candidates,

        "average_candidates_per_utterance":
            total_candidates
            / len(dataset),

        "total_reference_characters":
            total_chars,

        "total_reference_words":
            total_words,
    },

    "baseline": {

        "cer": float(
            baseline_cer
        ),

        "wer": float(
            baseline_wer
        ),

        "description":
            "Exact original N-best Top-1 candidate",
    },

    "lambda_results": final_results,

    "best": {

        "cer_lambda": best_cer_lambda,

        "cer":
            final_results[
                best_cer_lambda
            ]["cer"],

        "cer_delta_pp":
            final_results[
                best_cer_lambda
            ]["cer_delta_pp"],

        "wer_lambda": best_wer_lambda,

        "wer":
            final_results[
                best_wer_lambda
            ]["wer"],

        "wer_delta_pp":
            final_results[
                best_wer_lambda
            ]["wer_delta_pp"],
    },

    "sanity_checks": {

        "lambda_0_cer_matches_baseline":
            bool(
                baseline_match_cer
            ),

        "lambda_0_wer_matches_baseline":
            bool(
                baseline_match_wer
            ),

        "lambda_0_rank_equals_one":
            bool(
                lambda0_rank_ok
            ),
    },

    "oof_predictions": oof_predictions,
}


OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

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
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print(
    OUTPUT_PATH
)

print()
print("=" * 70)
print("SUCCESS")
print("=" * 70)