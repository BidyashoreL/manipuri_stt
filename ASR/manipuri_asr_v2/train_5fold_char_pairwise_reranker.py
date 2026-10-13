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
# CONFIG
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
    / "char_pairwise_reranker_results.json"
)

TOP_K_CANDS = 20
MAX_TEXT_LEN = 128

SEED = 42
N_FOLDS = 5

EPOCHS = 10
BATCH_SIZE = 128

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

MARGIN = 0.1

# Weighting strength for CER differences.
#
# Weight = 1 + PAIR_WEIGHT_ALPHA * abs(CER_A - CER_B)
#
# This keeps every pair represented while giving more influence
# to pairs where the quality difference is larger.
PAIR_WEIGHT_ALPHA = 2.0

LAMBDA_WEIGHTS = [
    0.0,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
]

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("=" * 70)
print("5-FOLD CHARACTER-AWARE PAIRWISE N-BEST RERANKER")
print("SCALED + CER-DIFFERENCE-WEIGHTED VERSION")
print("=" * 70)

print(f"Device             : {DEVICE}")
print(f"Top-K candidates   : {TOP_K_CANDS}")
print(f"Max text length    : {MAX_TEXT_LEN}")
print(f"Folds              : {N_FOLDS}")
print(f"Epochs             : {EPOCHS}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Margin             : {MARGIN}")
print(f"Pair weight alpha  : {PAIR_WEIGHT_ALPHA}")
print(f"Lambdas            : {LAMBDA_WEIGHTS}")
print()


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text):
    return " ".join(
        unicodedata.normalize(
            "NFC",
            str(text)
        ).split()
    ).strip()


def edit_distance(a, b):

    prev = list(
        range(len(b) + 1)
    )

    for i, x in enumerate(a, 1):

        cur = [i]

        for j, y in enumerate(b, 1):

            cost = (
                0
                if x == y
                else 1
            )

            cur.append(
                min(
                    cur[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + cost
                )
            )

        prev = cur

    return prev[-1]


def calc_cer(ref, hyp):

    ref = normalize_text(ref)
    hyp = normalize_text(hyp)

    if not ref:
        return (
            0.0
            if not hyp
            else 1.0
        )

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
        return (
            0.0
            if not hyp_words
            else 1.0
        )

    return (
        edit_distance(
            ref_words,
            hyp_words
        )
        / len(ref_words)
    )


# ============================================================
# LOAD N-BEST
# ============================================================

if not NBEST_PATH.exists():

    raise FileNotFoundError(
        f"N-best file not found:\n{NBEST_PATH}"
    )

print("Loading N-best dataset...")

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
    f"Loaded {len(samples)} samples."
)

print()


# ============================================================
# CHARACTER VOCABULARY
# ============================================================

print("Building character vocabulary...")

vocab = {
    "<pad>": 0,
    "<unk>": 1,
}

for sample in samples:

    hyps = sample.get(
        "hypotheses",
        sample.get(
            "nbest",
            []
        )
    )[:TOP_K_CANDS]

    for h in hyps:

        text = normalize_text(
            h.get(
                "text",
                ""
            )
        )

        for ch in text:

            if ch not in vocab:

                vocab[ch] = len(vocab)


print(
    f"Character vocabulary size: "
    f"{len(vocab)}"
)

print()


# ============================================================
# CHARACTER ENCODING
# ============================================================

def text_to_ids(
    text,
    max_len=MAX_TEXT_LEN
):

    text = normalize_text(text)

    chars = list(
        text[:max_len]
    )

    ids = [
        vocab.get(
            ch,
            1
        )
        for ch in chars
    ]

    if len(ids) < max_len:

        ids.extend(
            [0]
            * (
                max_len
                - len(ids)
            )
        )

    return ids


# ============================================================
# MODEL
# ============================================================

class CharPairwiseRanker(nn.Module):

    def __init__(
        self,
        vocab_size,
        emb_dim=64,
        hidden_dim=128
    ):

        super().__init__()

        self.char_emb = nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=0
        )

        self.encoder = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.mlp = nn.Sequential(

            nn.Linear(
                hidden_dim * 2 + 3,
                128
            ),

            nn.SiLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                64
            ),

            nn.SiLU(),

            nn.Linear(
                64,
                1
            )
        )

    def forward(
        self,
        char_ids,
        scalar_feats
    ):

        emb = self.char_emb(
            char_ids
        )

        _, h_n = self.encoder(
            emb
        )

        forward_state = h_n[-2]
        backward_state = h_n[-1]

        text_repr = torch.cat(
            [
                forward_state,
                backward_state
            ],
            dim=-1
        )

        combined = torch.cat(
            [
                text_repr,
                scalar_feats
            ],
            dim=-1
        )

        return self.mlp(
            combined
        ).squeeze(-1)


# ============================================================
# BUILD DATASET
# ============================================================

dataset = []

total_candidates = 0

for sample_index, sample in enumerate(samples):

    ref = normalize_text(
        sample.get(
            "reference",
            sample.get(
                "text",
                ""
            )
        )
    )

    hyps = sample.get(
        "hypotheses",
        sample.get(
            "nbest",
            []
        )
    )[:TOP_K_CANDS]

    if not hyps:
        continue

    char_ids = []
    scalar_features = []

    cers = []
    wers = []
    texts = []
    ranks = []

    for rank_idx, h in enumerate(
        hyps,
        start=1
    ):

        htext = normalize_text(
            h.get(
                "text",
                ""
            )
        )

        logit = float(
            h.get(
                "logit_score",
                0.0
            )
        )

        lm = float(
            h.get(
                "lm_score",
                0.0
            )
        )

        char_ids.append(
            text_to_ids(
                htext
            )
        )

        scalar_features.append(
            [
                logit,
                lm,
                float(rank_idx)
            ]
        )

        cers.append(
            calc_cer(
                ref,
                htext
            )
        )

        wers.append(
            calc_wer(
                ref,
                htext
            )
        )

        texts.append(
            htext
        )

        ranks.append(
            rank_idx
        )

    dataset.append(
        {
            "sample_index":
                sample_index,

            "ref":
                ref,

            "char_ids":
                np.asarray(
                    char_ids,
                    dtype=np.int64
                ),

            "scalars":
                np.asarray(
                    scalar_features,
                    dtype=np.float32
                ),

            "cers":
                np.asarray(
                    cers,
                    dtype=np.float32
                ),

            "wers":
                np.asarray(
                    wers,
                    dtype=np.float32
                ),

            "texts":
                texts,

            "ranks":
                ranks,
        }
    )

    total_candidates += len(hyps)


print(
    f"Top-{TOP_K_CANDS} candidate instances: "
    f"{total_candidates:,}"
)

print(
    f"Average candidates/utterance: "
    f"{total_candidates / len(dataset):.2f}"
)

print()


# ============================================================
# CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

sample_ids = np.arange(
    len(dataset)
)


# ============================================================
# RESULTS
# ============================================================

fold_evals = {

    lam: {

        "cer_errors": 0,

        "wer_errors": 0,

        "selected_ranks": [],

        "selection_changed": 0,

        "num_samples": 0,

    }

    for lam in LAMBDA_WEIGHTS
}


baseline_cer_errors = 0
baseline_wer_errors = 0

total_chars = 0
total_words = 0

oof_predictions = []


# ============================================================
# CROSS-VALIDATED TRAINING
# ============================================================

print("=" * 70)
print(
    "STARTING 5-FOLD CHARACTER-AWARE "
    "PAIRWISE RERANKING"
)
print("=" * 70)
print()


for fold, (
    train_idx,
    test_idx
) in enumerate(
    kf.split(sample_ids),
    start=1
):

    print("-" * 70)

    print(
        f"FOLD {fold}/{N_FOLDS}"
    )

    print(
        f"Training utterances: "
        f"{len(train_idx)}"
    )

    print(
        f"Test utterances:     "
        f"{len(test_idx)}"
    )

    # ========================================================
    # SCALER
    #
    # Fit ONLY on training candidates.
    # ========================================================

    training_scalars = np.vstack(
        [
            dataset[i]["scalars"]
            for i in train_idx
        ]
    )

    scaler = StandardScaler()

    scaler.fit(
        training_scalars
    )

    # ========================================================
    # BUILD PAIRS
    # ========================================================

    pairs_c1 = []
    pairs_c2 = []

    pairs_s1 = []
    pairs_s2 = []

    pair_weights = []

    pair_count = 0

    for i in train_idx:

        sample = dataset[i]

        c_ids = sample[
            "char_ids"
        ]

        scalars = sample[
            "scalars"
        ]

        cers = sample[
            "cers"
        ]

        # ----------------------------------------------------
        # Scale candidate scalar features using TRAIN scaler
        # ----------------------------------------------------

        scaled_scalars = scaler.transform(
            scalars
        )

        n_candidates = len(
            cers
        )

        for a in range(
            n_candidates
        ):

            for b in range(
                a + 1,
                n_candidates
            ):

                if cers[a] < cers[b]:

                    better = a
                    worse = b

                elif cers[b] < cers[a]:

                    better = b
                    worse = a

                else:

                    # Equal CER: no ranking signal.
                    continue

                pairs_c1.append(
                    c_ids[better]
                )

                pairs_c2.append(
                    c_ids[worse]
                )

                pairs_s1.append(
                    scaled_scalars[
                        better
                    ]
                )

                pairs_s2.append(
                    scaled_scalars[
                        worse
                    ]
                )

                # ------------------------------------------------
                # CER-DIFFERENCE WEIGHT
                #
                # Large quality difference:
                #   slightly higher weight
                #
                # Tiny quality difference:
                #   close to base weight 1.0
                # ------------------------------------------------

                cer_gap = abs(
                    float(
                        cers[better]
                        - cers[worse]
                    )
                )

                weight = (
                    1.0
                    + PAIR_WEIGHT_ALPHA
                    * cer_gap
                )

                pair_weights.append(
                    weight
                )

                pair_count += 1

    print(
        f"Training pairs: "
        f"{pair_count:,}"
    )

    if pair_count == 0:

        raise RuntimeError(
            "No pairwise training pairs."
        )

    # ========================================================
    # TORCH TENSORS
    # ========================================================

    c1_tensor = torch.tensor(
        np.asarray(
            pairs_c1,
            dtype=np.int64
        ),
        dtype=torch.long
    )

    c2_tensor = torch.tensor(
        np.asarray(
            pairs_c2,
            dtype=np.int64
        ),
        dtype=torch.long
    )

    s1_tensor = torch.tensor(
        np.asarray(
            pairs_s1,
            dtype=np.float32
        ),
        dtype=torch.float32
    )

    s2_tensor = torch.tensor(
        np.asarray(
            pairs_s2,
            dtype=np.float32
        ),
        dtype=torch.float32
    )

    weight_tensor = torch.tensor(
        np.asarray(
            pair_weights,
            dtype=np.float32
        ),
        dtype=torch.float32
    )

    del pairs_c1
    del pairs_c2
    del pairs_s1
    del pairs_s2
    del pair_weights

    # ========================================================
    # MODEL
    # ========================================================

    model = CharPairwiseRanker(
        vocab_size=len(vocab)
    ).to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # Reduction NONE is required so that
    # each pair can have its own CER-difference weight.
    loss_fn = nn.MarginRankingLoss(
        margin=MARGIN,
        reduction="none"
    )

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    last_loss = None

    for epoch in range(
        EPOCHS
    ):

        permutation = torch.randperm(
            len(c1_tensor)
        )

        epoch_loss = 0.0
        batch_count = 0

        for start in range(
            0,
            len(c1_tensor),
            BATCH_SIZE
        ):

            idx = permutation[
                start:
                start + BATCH_SIZE
            ]

            if len(idx) < BATCH_SIZE:
                continue

            c1_batch = (
                c1_tensor[idx]
                .to(DEVICE)
            )

            c2_batch = (
                c2_tensor[idx]
                .to(DEVICE)
            )

            s1_batch = (
                s1_tensor[idx]
                .to(DEVICE)
            )

            s2_batch = (
                s2_tensor[idx]
                .to(DEVICE)
            )

            weights_batch = (
                weight_tensor[idx]
                .to(DEVICE)
            )

            score1 = model(
                c1_batch,
                s1_batch
            )

            score2 = model(
                c2_batch,
                s2_batch
            )

            target = torch.ones_like(
                score1
            )

            pair_losses = loss_fn(
                score1,
                score2,
                target
            )

            loss = (
                pair_losses
                * weights_batch
            ).mean()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += (
                loss.item()
            )

            batch_count += 1

        if batch_count > 0:

            last_loss = (
                epoch_loss
                / batch_count
            )

    print(
        f"Final training loss: "
        f"{last_loss:.6f}"
    )

    # ========================================================
    # OUT-OF-FOLD EVALUATION
    # ========================================================

    model.eval()

    with torch.no_grad():

        for i in test_idx:

            sample = dataset[i]

            ref = sample["ref"]

            top1_text = sample[
                "texts"
            ][0]

            # ------------------------------------------------
            # CORRECT DENOMINATORS
            # ------------------------------------------------

            total_chars += len(ref)

            total_words += len(
                ref.split()
            )

            # ------------------------------------------------
            # EXACT TOP-1 BASELINE
            # ------------------------------------------------

            baseline_cer_errors += edit_distance(
                list(ref),
                list(top1_text)
            )

            baseline_wer_errors += edit_distance(
                ref.split(),
                top1_text.split()
            )

            # ------------------------------------------------
            # SCALE TEST SCALARS
            # ------------------------------------------------

            scaled_scalars = scaler.transform(
                sample["scalars"]
            )

            char_tensor = torch.tensor(
                sample["char_ids"],
                dtype=torch.long
            ).to(DEVICE)

            scalar_tensor = torch.tensor(
                scaled_scalars,
                dtype=torch.float32
            ).to(DEVICE)

            # ------------------------------------------------
            # NEURAL SCORES
            # ------------------------------------------------

            neural_scores = (
                model(
                    char_tensor,
                    scalar_tensor
                )
                .cpu()
                .numpy()
            )

            # ------------------------------------------------
            # RANK PRIOR
            # ------------------------------------------------

            ranks = np.arange(
                1,
                len(
                    sample["texts"]
                ) + 1,
                dtype=np.float32
            )

            rank_prior = -np.log(
                ranks
            )

            # ------------------------------------------------
            # OOF RECORD
            # ------------------------------------------------

            sample_output = {

                "sample_index":
                    int(
                        sample[
                            "sample_index"
                        ]
                    ),

                "fold":
                    int(fold),

                "reference":
                    ref,

                "baseline_text":
                    top1_text,

                "baseline_cer":
                    float(
                        calc_cer(
                            ref,
                            top1_text
                        )
                    ),

                "baseline_wer":
                    float(
                        calc_wer(
                            ref,
                            top1_text
                        )
                    ),

                "lambda_results":
                    {},
            }

            # ------------------------------------------------
            # LAMBDA SWEEP
            # ------------------------------------------------

            for lam in LAMBDA_WEIGHTS:

                # Exact original Top-1.
                if lam == 0.0:

                    best_idx = 0

                    final_scores = None

                else:

                    final_scores = (
                        rank_prior
                        + lam
                        * neural_scores
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

                selected_cer = calc_cer(
                    ref,
                    selected_text
                )

                selected_wer = calc_wer(
                    ref,
                    selected_text
                )

                selected_cer_errors = edit_distance(
                    list(ref),
                    list(selected_text)
                )

                selected_wer_errors = edit_distance(
                    ref.split(),
                    selected_text.split()
                )

                fold_evals[lam][
                    "cer_errors"
                ] += selected_cer_errors

                fold_evals[lam][
                    "wer_errors"
                ] += selected_wer_errors

                fold_evals[lam][
                    "selected_ranks"
                ].append(
                    selected_rank
                )

                fold_evals[lam][
                    "num_samples"
                ] += 1

                if selected_rank != 1:

                    fold_evals[lam][
                        "selection_changed"
                    ] += 1

                sample_output[
                    "lambda_results"
                ][str(lam)] = {

                    "selected_index":
                        int(best_idx),

                    "selected_rank":
                        int(selected_rank),

                    "selected_text":
                        selected_text,

                    "cer":
                        float(selected_cer),

                    "wer":
                        float(selected_wer),

                    "neural_score":
                        float(
                            neural_scores[
                                best_idx
                            ]
                        ),

                    "rank_prior":
                        float(
                            rank_prior[
                                best_idx
                            ]
                        ),

                    "final_score":
                        (
                            None
                            if final_scores is None
                            else float(
                                final_scores[
                                    best_idx
                                ]
                            )
                        ),
                }

            oof_predictions.append(
                sample_output
            )

    print(
        f"Fold {fold}/{N_FOLDS} completed."
    )

    print()


# ============================================================
# BASELINE
# ============================================================

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


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print(
    "CHARACTER-AWARE PAIRWISE "
    "RERANKER RESULTS"
)
print("=" * 70)

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
    f"Reference chars    : "
    f"{total_chars:,}"
)

print(
    f"Reference words    : "
    f"{total_words:,}"
)

print()

print("RERANKER")
print("-" * 70)


final_results = {}


for lam in LAMBDA_WEIGHTS:

    result = fold_evals[
        lam
    ]

    cer = (
        100.0
        * result[
            "cer_errors"
        ]
        / total_chars
    )

    wer = (
        100.0
        * result[
            "wer_errors"
        ]
        / total_words
    )

    avg_rank = float(
        np.mean(
            result[
                "selected_ranks"
            ]
        )
    )

    median_rank = float(
        np.median(
            result[
                "selected_ranks"
            ]
        )
    )

    changed_pct = (
        100.0
        * result[
            "selection_changed"
        ]
        / result[
            "num_samples"
        ]
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
        f"Changed: {changed_pct:.2f}%"
    )

    final_results[
        str(lam)
    ] = {

        "cer":
            float(cer),

        "wer":
            float(wer),

        "cer_delta_pp":
            float(cer_delta),

        "wer_delta_pp":
            float(wer_delta),

        "average_selected_rank":
            float(avg_rank),

        "median_selected_rank":
            float(median_rank),

        "selection_change_rate_percent":
            float(changed_pct),

        "num_samples":
            int(
                result[
                    "num_samples"
                ]
            ),
    }


# ============================================================
# BEST
# ============================================================

best_cer_lambda = min(
    final_results.keys(),
    key=lambda x:
        final_results[x][
            "cer"
        ]
)

best_wer_lambda = min(
    final_results.keys(),
    key=lambda x:
        final_results[x][
            "wer"
        ]
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
    f"CER delta       : "
    f"{final_results[best_cer_lambda]['cer_delta_pp']:+.4f} pp"
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
    f"WER delta       : "
    f"{final_results[best_wer_lambda]['wer_delta_pp']:+.4f} pp"
)


# ============================================================
# SANITY CHECKS
# ============================================================

print()
print("=" * 70)
print("SANITY CHECKS")
print("=" * 70)

lambda0 = final_results["0.0"]

cer_match = np.isclose(
    lambda0["cer"],
    baseline_cer,
    atol=1e-9
)

wer_match = np.isclose(
    lambda0["wer"],
    baseline_wer,
    atol=1e-9
)

rank_match = np.isclose(
    lambda0[
        "average_selected_rank"
    ],
    1.0,
    atol=1e-9
)

print(
    "[CHECK] Lambda 0 CER matches "
    f"baseline: "
    f"{'PASS' if cer_match else 'FAIL'}"
)

print(
    "[CHECK] Lambda 0 WER matches "
    f"baseline: "
    f"{'PASS' if wer_match else 'FAIL'}"
)

print(
    "[CHECK] Lambda 0 average rank = 1: "
    f"{'PASS' if rank_match else 'FAIL'}"
)

if (
    cer_match
    and wer_match
    and rank_match
):

    print(
        "\nAll baseline sanity checks passed."
    )

else:

    print(
        "\nWARNING: Baseline sanity check failed."
    )


# ============================================================
# SAVE
# ============================================================

output = {

    "experiment": {

        "name":
            "5-fold character-aware "
            "pairwise N-best reranker",

        "version":
            "scaled_weighted_v1",

        "seed":
            SEED,

        "n_folds":
            N_FOLDS,

        "top_k_candidates":
            TOP_K_CANDS,

        "max_text_length":
            MAX_TEXT_LEN,

        "epochs":
            EPOCHS,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "margin":
            MARGIN,

        "pair_weight_alpha":
            PAIR_WEIGHT_ALPHA,

        "lambda_weights":
            LAMBDA_WEIGHTS,

        "device":
            str(DEVICE),

        "candidate_file":
            str(NBEST_PATH),
    },

    "vocabulary": {

        "size":
            len(vocab),

        "pad_id":
            0,

        "unk_id":
            1,
    },

    "dataset": {

        "utterances":
            len(dataset),

        "total_candidates":
            total_candidates,

        "average_candidates":
            total_candidates
            / len(dataset),

        "reference_characters":
            total_chars,

        "reference_words":
            total_words,
    },

    "baseline": {

        "cer":
            float(baseline_cer),

        "wer":
            float(baseline_wer),
    },

    "lambda_results":
        final_results,

    "best": {

        "cer_lambda":
            best_cer_lambda,

        "cer":
            final_results[
                best_cer_lambda
            ]["cer"],

        "cer_delta_pp":
            final_results[
                best_cer_lambda
            ]["cer_delta_pp"],

        "wer_lambda":
            best_wer_lambda,

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

        "lambda_0_cer_matches":
            bool(cer_match),

        "lambda_0_wer_matches":
            bool(wer_match),

        "lambda_0_rank_equals_one":
            bool(rank_match),
    },

    "oof_predictions":
        oof_predictions,
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
# FINISHED
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