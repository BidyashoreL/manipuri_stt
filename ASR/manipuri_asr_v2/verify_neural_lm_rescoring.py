from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# FINAL RESEARCH-CLEAN VERSION
# B3 + 4-GRAM POOLED N-BEST
#          +
# CHARACTER-LEVEL NEURAL LM
#
# IMPORTANT:
# - Validation set only: 864 utterances
# - Official 410-sample test set is NOT used
# - Existing N-best candidates are reused
# - Existing trained Neural LM is reused
# - No ASR retraining
# ============================================================


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent


# ------------------------------------------------------------
# CHARACTER NEURAL LM
# ------------------------------------------------------------

LM_DIR = (
    SCRIPT_DIR
    / "checkpoints"
    / "lm"
)

LM_CHECKPOINT = (
    LM_DIR
    / "char_lstm_lm_best.pth"
)

VOCAB_PATH = (
    LM_DIR
    / "char_vocab.json"
)

METADATA_PATH = (
    LM_DIR
    / "char_lm_metadata.json"
)


# ------------------------------------------------------------
# POOLED N-BEST
# ------------------------------------------------------------

NBEST_PATH = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "b3_4gram_pooled100_val.json"
)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "neural_lm"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_JSON = (
    OUTPUT_DIR
    / "neural_lm_rescoring_verification.json"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# We intentionally evaluate only the first 20 candidates.
TOP_K = 20

# Batch size for Neural LM scoring.
# 128 is safe for your RTX 5050 and this small LSTM.
LM_BATCH_SIZE = 128

# The 4-gram decoder used alpha=0.5.
#
# IMPORTANT:
# This is used ONLY to reconstruct an approximate acoustic +
# LM score from the stored pyctcdecode components.
#
# It is NOT treated as the exact pyctcdecode ranking.
FOURGRAM_ALPHA = 0.5


# Neural LM fusion weights.
#
# beta = 0:
# reconstructed 4-gram score only.
#
# beta > 0:
# reconstructed 4-gram score + neural LM.
BETAS = [
    0.0,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    1.00,
    2.00,
    3.00,
]


# Expected validation size.
EXPECTED_VALIDATION_SAMPLES = 864


# Expected Neural LM vocabulary size.
EXPECTED_VOCAB_SIZE = 62


# Expected trained parameter count.
EXPECTED_PARAMETER_COUNT = 875_966


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    text = unicodedata.normalize(
        "NFC",
        str(text)
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_distance(a, b):
    """
    Standard Levenshtein distance.

    Works for:
        strings
    or:
        lists of words
    """

    a = list(a)
    b = list(b)

    if len(a) < len(b):
        a, b = b, a

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

            insert_cost = (
                current[j - 1]
                + 1
            )

            delete_cost = (
                previous[j]
                + 1
            )

            substitute_cost = (
                previous[j - 1]
                + (ca != cb)
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    substitute_cost,
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# CHARACTER LSTM
# ============================================================

class CharLSTM(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim,
        hidden_dim,
        num_layers,
        dropout,
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=(
                dropout
                if num_layers > 1
                else 0.0
            ),
        )

        # IMPORTANT:
        # The original training checkpoint uses
        # "fc.weight" and "fc.bias".
        self.fc = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(self, x):

        x = self.embedding(x)

        x, _ = self.lstm(x)

        logits = self.fc(x)

        return logits


# ============================================================
# Z-SCORE
# ============================================================

def zscore(values):

    values = np.asarray(
        values,
        dtype=np.float64,
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
# METRICS
# ============================================================

def compute_metrics(
    references,
    hypotheses,
):

    total_char_errors = 0
    total_ref_chars = 0

    total_word_errors = 0
    total_ref_words = 0

    for ref, hyp in zip(
        references,
        hypotheses,
    ):

        ref = normalize_text(ref)
        hyp = normalize_text(hyp)

        # Character-level CER.
        total_char_errors += (
            edit_distance(
                ref,
                hyp,
            )
        )

        total_ref_chars += len(
            ref
        )

        # Word-level WER.
        ref_words = ref.split()
        hyp_words = hyp.split()

        total_word_errors += (
            edit_distance(
                ref_words,
                hyp_words,
            )
        )

        total_ref_words += len(
            ref_words
        )

    cer = (
        100.0
        * total_char_errors
        / max(
            total_ref_chars,
            1,
        )
    )

    wer = (
        100.0
        * total_word_errors
        / max(
            total_ref_words,
            1,
        )
    )

    return {
        "cer": cer,
        "wer": wer,
        "char_errors": total_char_errors,
        "ref_chars": total_ref_chars,
        "word_errors": total_word_errors,
        "ref_words": total_ref_words,
    }


# ============================================================
# START
# ============================================================

print("=" * 78)
print(
    "FINAL NEURAL LM N-BEST RESCORING EXPERIMENT"
)
print("=" * 78)

print(
    f"\nDevice : {DEVICE}"
)

print(
    "\nIMPORTANT:"
)

print(
    "  Validation set only : 864 utterances"
)

print(
    "  Official 410 test   : NOT USED"
)

print(
    "  ASR retraining      : NONE"
)


# ============================================================
# FILE CHECKS
# ============================================================

required_files = [
    LM_CHECKPOINT,
    VOCAB_PATH,
    NBEST_PATH,
]

print(
    "\n" + "=" * 78
)

print(
    "CHECKING REQUIRED FILES"
)

print(
    "=" * 78
)

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file does not exist:\n{path}"
        )

    print(
        f"[OK] {path}"
    )


# ============================================================
# LOAD VOCABULARY
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "LOADING CHARACTER VOCABULARY"
)

print(
    "=" * 78
)

with open(
    VOCAB_PATH,
    "r",
    encoding="utf-8",
) as f:

    vocab_data = json.load(f)


# ------------------------------------------------------------
# IMPORTANT:
#
# char_vocab.json structure:
#
# {
#     "itos": [...],
#     "stoi": {...},
#     "pad_token": "...",
#     "unk_token": "...",
#     "bos_token": "...",
#     "eos_token": "...",
#     "pad_id": 0,
#     "unk_id": 1,
#     "bos_id": 2,
#     "eos_id": 3
# }
# ------------------------------------------------------------

if "stoi" not in vocab_data:

    raise RuntimeError(
        "char_vocab.json does not contain 'stoi'."
    )

vocab = vocab_data["stoi"]

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

VOCAB_SIZE = len(
    vocab
)

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


# ------------------------------------------------------------
# Vocabulary sanity checks
# ------------------------------------------------------------

if VOCAB_SIZE != EXPECTED_VOCAB_SIZE:

    raise RuntimeError(
        f"Unexpected vocabulary size: "
        f"{VOCAB_SIZE}. "
        f"Expected {EXPECTED_VOCAB_SIZE}."
    )

required_tokens = [
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
]

for token in required_tokens:

    if token not in vocab:

        raise RuntimeError(
            f"Missing required token: {token}"
        )


print(
    "[PASS] Vocabulary sanity checks."
)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = {}

if METADATA_PATH.exists():

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)


print(
    "\n[MODEL] Metadata:"
)

if metadata:

    for key in [
        "embedding_dim",
        "hidden_dim",
        "num_layers",
        "dropout",
        "vocab_size",
    ]:

        if key in metadata:

            print(
                f"  {key}: "
                f"{metadata[key]}"
            )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "LOADING TRAINED CHARACTER LSTM"
)

print(
    "=" * 78
)

checkpoint = torch.load(
    LM_CHECKPOINT,
    map_location="cpu",
)


# ------------------------------------------------------------
# Extract state dictionary.
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict
):

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        "state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ------------------------------------------------------------
# Infer architecture DIRECTLY from checkpoint.
#
# This avoids depending on incomplete metadata.
# ------------------------------------------------------------

embedding_weight = (
    state_dict[
        "embedding.weight"
    ]
)

embed_dim = (
    embedding_weight.shape[1]
)

vocab_from_checkpoint = (
    embedding_weight.shape[0]
)


fc_weight = (
    state_dict[
        "fc.weight"
    ]
)

hidden_dim = (
    fc_weight.shape[1]
)

fc_vocab_size = (
    fc_weight.shape[0]
)


# Determine number of LSTM layers.
lstm_layer_ids = []

for key in state_dict:

    match = re.match(
        r"lstm\.weight_ih_l(\d+)$",
        key,
    )

    if match:

        lstm_layer_ids.append(
            int(match.group(1))
        )

if not lstm_layer_ids:

    raise RuntimeError(
        "Could not determine LSTM layer count "
        "from checkpoint."
    )

num_layers = (
    max(lstm_layer_ids)
    + 1
)


# Dropout does not affect parameter shapes.
# The training output confirmed dropout=0.2.
dropout = float(
    metadata.get(
        "dropout",
        0.2,
    )
)


print(
    "\n[MODEL] Architecture inferred "
    "from checkpoint:"
)

print(
    f"  Vocabulary : "
    f"{vocab_from_checkpoint}"
)

print(
    f"  Embedding  : "
    f"{embed_dim}"
)

print(
    f"  Hidden     : "
    f"{hidden_dim}"
)

print(
    f"  LSTM layers: "
    f"{num_layers}"
)

print(
    f"  Dropout    : "
    f"{dropout}"
)


# ------------------------------------------------------------
# Architecture consistency checks.
# ------------------------------------------------------------

if vocab_from_checkpoint != VOCAB_SIZE:

    raise RuntimeError(
        "Checkpoint vocabulary size does not "
        "match char_vocab.json."
    )

if fc_vocab_size != VOCAB_SIZE:

    raise RuntimeError(
        "Checkpoint FC output size does not "
        "match vocabulary size."
    )


# ------------------------------------------------------------
# Build exact architecture.
# ------------------------------------------------------------

model = CharLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=embed_dim,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
)


# ------------------------------------------------------------
# Load weights.
# ------------------------------------------------------------

missing_keys, unexpected_keys = (
    model.load_state_dict(
        state_dict,
        strict=False,
    )
)


if missing_keys:

    raise RuntimeError(
        "Missing checkpoint keys:\n"
        + "\n".join(
            missing_keys
        )
    )

if unexpected_keys:

    raise RuntimeError(
        "Unexpected checkpoint keys:\n"
        + "\n".join(
            unexpected_keys
        )
    )


model.to(
    DEVICE
)

model.eval()


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print(
    f"\n[MODEL] Parameters: "
    f"{parameter_count:,}"
)

if (
    parameter_count
    != EXPECTED_PARAMETER_COUNT
):

    raise RuntimeError(
        f"Unexpected parameter count: "
        f"{parameter_count:,}. "
        f"Expected "
        f"{EXPECTED_PARAMETER_COUNT:,}."
    )


print(
    "[PASS] Neural LM checkpoint loaded exactly."
)


# ============================================================
# LOAD POOLED N-BEST
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "LOADING POOLED N-BEST"
)

print(
    "=" * 78
)

with open(
    NBEST_PATH,
    "r",
    encoding="utf-8",
) as f:

    nbest_data = json.load(f)


if "samples" not in nbest_data:

    raise RuntimeError(
        "N-best JSON does not contain "
        "'samples'."
    )

samples = (
    nbest_data["samples"]
)

print(
    f"Samples: "
    f"{len(samples):,}"
)

if (
    len(samples)
    != EXPECTED_VALIDATION_SAMPLES
):

    raise RuntimeError(
        f"Expected "
        f"{EXPECTED_VALIDATION_SAMPLES} "
        f"validation samples, "
        f"found {len(samples)}."
    )


print(
    "[PASS] Exactly 864 validation "
    "utterances loaded."
)


# ============================================================
# ENCODE TEXT
# ============================================================

def encode_text(text):

    text = normalize_text(
        text
    )

    ids = [
        BOS_ID
    ]

    for char in text:

        ids.append(
            vocab.get(
                char,
                UNK_ID,
            )
        )

    ids.append(
        EOS_ID
    )

    return ids


# ============================================================
# BATCHED NEURAL LM SCORING
# ============================================================

@torch.no_grad()
def score_candidates_batch(
    texts,
):
    """
    Score a batch of candidate strings.

    For each candidate:

        BOS -> character 1
        character 1 -> character 2
        ...
        character N -> EOS

    Returns:

        total_log_prob
        average_log_prob
        token_count
        character_count
    """

    encoded = [
        encode_text(text)
        for text in texts
    ]

    input_sequences = [
        ids[:-1]
        for ids in encoded
    ]

    target_sequences = [
        ids[1:]
        for ids in encoded
    ]

    lengths = [
        len(ids)
        for ids in input_sequences
    ]

    max_len = max(
        lengths
    )

    batch_size = len(
        texts
    )

    input_tensor = torch.full(
        (
            batch_size,
            max_len,
        ),
        PAD_ID,
        dtype=torch.long,
        device=DEVICE,
    )

    target_tensor = torch.full(
        (
            batch_size,
            max_len,
        ),
        PAD_ID,
        dtype=torch.long,
        device=DEVICE,
    )

    valid_mask = torch.zeros(
        (
            batch_size,
            max_len,
        ),
        dtype=torch.bool,
        device=DEVICE,
    )

    for i, (
        input_ids,
        target_ids,
    ) in enumerate(
        zip(
            input_sequences,
            target_sequences,
        )
    ):

        length = len(
            input_ids
        )

        input_tensor[
            i,
            :length
        ] = torch.tensor(
            input_ids,
            dtype=torch.long,
            device=DEVICE,
        )

        target_tensor[
            i,
            :length
        ] = torch.tensor(
            target_ids,
            dtype=torch.long,
            device=DEVICE,
        )

        valid_mask[
            i,
            :length
        ] = True

    logits = model(
        input_tensor
    )

    log_probs = torch.log_softmax(
        logits,
        dim=-1,
    )

    # Gather probability of the actual next token.
    token_log_probs = (
        log_probs
        .gather(
            2,
            target_tensor.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    # Remove padding positions.
    token_log_probs = (
        token_log_probs
        * valid_mask
    )

    total_log_probs = (
        token_log_probs
        .sum(dim=1)
    )

    token_counts = (
        valid_mask
        .sum(dim=1)
    )

    avg_log_probs = (
        total_log_probs
        / token_counts.clamp(
            min=1
        )
    )

    character_counts = torch.tensor(
        [
            len(
                normalize_text(text)
            )
            for text in texts
        ],
        dtype=torch.long,
    )

    return [
        {
            "neural_lm_log_prob": float(
                total_log_probs[i]
                .item()
            ),
            "neural_lm_avg_log_prob": float(
                avg_log_probs[i]
                .item()
            ),
            "neural_lm_token_count": int(
                token_counts[i]
                .item()
            ),
            "neural_lm_char_count": int(
                character_counts[i]
                .item()
            ),
        }
        for i in range(
            batch_size
        )
    ]


# ============================================================
# PREPARE TOP-20 CANDIDATES
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "PREPARING TOP-20 CANDIDATES"
)

print(
    "=" * 78
)

references = []
candidate_sets = []

total_candidates = 0

for sample_idx, sample in enumerate(
    samples
):

    reference = normalize_text(
        sample["reference"]
    )

    hypotheses = (
        sample["hypotheses"][
            :TOP_K
        ]
    )

    if not hypotheses:

        raise RuntimeError(
            f"Sample {sample_idx} "
            "contains no hypotheses."
        )

    # --------------------------------------------------------
    # Deduplicate normalized candidate text.
    # --------------------------------------------------------

    unique = []
    seen = set()

    for hyp in hypotheses:

        text = normalize_text(
            hyp["text"]
        )

        if text in seen:

            continue

        seen.add(
            text
        )

        # Copy so we don't mutate
        # the original loaded object.
        copied = dict(
            hyp
        )

        copied["text"] = (
            text
        )

        unique.append(
            copied
        )

    references.append(
        reference
    )

    candidate_sets.append(
        unique
    )

    total_candidates += len(
        unique
    )


print(
    f"Validation utterances : "
    f"{len(candidate_sets):,}"
)

print(
    f"Top-K requested        : "
    f"{TOP_K}"
)

print(
    f"Candidate instances    : "
    f"{total_candidates:,}"
)

print(
    f"Average candidates     : "
    f"{total_candidates / len(candidate_sets):.2f}"
)


# ============================================================
# SCORE CANDIDATES IN BATCHES
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "SCORING CANDIDATES WITH CHARACTER LSTM"
)

print(
    "=" * 78
)

all_candidate_refs = []

for sample_idx, candidates in enumerate(
    candidate_sets
):

    for candidate_idx, candidate in enumerate(
        candidates
    ):

        all_candidate_refs.append(
            (
                sample_idx,
                candidate_idx,
            )
        )


all_texts = [
    candidate_sets[
        sample_idx
    ][
        candidate_idx
    ]["text"]
    for (
        sample_idx,
        candidate_idx
    )
    in all_candidate_refs
]


for start in range(
    0,
    len(all_texts),
    LM_BATCH_SIZE,
):

    end = min(
        start + LM_BATCH_SIZE,
        len(all_texts),
    )

    batch_texts = all_texts[
        start:end
    ]

    batch_scores = (
        score_candidates_batch(
            batch_texts
        )
    )

    for local_idx, scores in enumerate(
        batch_scores
    ):

        global_idx = (
            start
            + local_idx
        )

        sample_idx, candidate_idx = (
            all_candidate_refs[
                global_idx
            ]
        )

        candidate = candidate_sets[
            sample_idx
        ][
            candidate_idx
        ]

        candidate.update(
            scores
        )

        # ----------------------------------------------------
        # Approximate reconstructed score.
        #
        # IMPORTANT:
        # This is NOT claimed to exactly reproduce
        # pyctcdecode's internal beam ranking.
        # ----------------------------------------------------

        candidate[
            "reconstructed_4gram_score"
        ] = (
            float(
                candidate[
                    "logit_score"
                ]
            )
            + FOURGRAM_ALPHA
            * float(
                candidate[
                    "lm_score"
                ]
            )
        )

    processed = end

    if (
        processed % 1000 == 0
        or processed == len(all_texts)
    ):

        print(
            f"Scored "
            f"{processed:,}/"
            f"{len(all_texts):,} candidates"
        )


print(
    "[PASS] All Top-20 candidates scored."
)


# ============================================================
# BASELINE 1:
# ORIGINAL N-BEST TOP-1
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "PRIMARY BASELINE: ORIGINAL N-BEST TOP-1"
)

print(
    "=" * 78
)

original_top1 = []

for candidates in candidate_sets:

    # The first candidate is the
    # original N-best rank-1 hypothesis.
    original_top1.append(
        candidates[0]["text"]
    )


original_metrics = compute_metrics(
    references,
    original_top1,
)

print(
    f"CER : "
    f"{original_metrics['cer']:.4f}%"
)

print(
    f"WER : "
    f"{original_metrics['wer']:.4f}%"
)

print(
    f"Reference characters : "
    f"{original_metrics['ref_chars']:,}"
)

print(
    f"Reference words      : "
    f"{original_metrics['ref_words']:,}"
)


# ============================================================
# BASELINE 2:
# RECONSTRUCTED 4-GRAM SCORE
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "DIAGNOSTIC: RECONSTRUCTED 4-GRAM SCORE"
)

print(
    "=" * 78
)

reconstructed_top1 = []
reconstructed_ranks = []

for candidates in candidate_sets:

    base_scores = np.array(
        [
            c[
                "reconstructed_4gram_score"
            ]
            for c in candidates
        ],
        dtype=np.float64,
    )

    best_idx = int(
        np.argmax(
            base_scores
        )
    )

    reconstructed_top1.append(
        candidates[
            best_idx
        ]["text"]
    )

    reconstructed_ranks.append(
        best_idx + 1
    )


reconstructed_metrics = compute_metrics(
    references,
    reconstructed_top1,
)

changed_from_original = sum(
    a != b
    for a, b in zip(
        original_top1,
        reconstructed_top1,
    )
)

print(
    f"CER : "
    f"{reconstructed_metrics['cer']:.4f}%"
)

print(
    f"WER : "
    f"{reconstructed_metrics['wer']:.4f}%"
)

print(
    f"Mean selected rank : "
    f"{np.mean(reconstructed_ranks):.2f}"
)

print(
    f"Median selected rank : "
    f"{np.median(reconstructed_ranks):.0f}"
)

print(
    f"Changed from original Top-1 : "
    f"{100.0 * changed_from_original / len(samples):.2f}%"
)

print(
    "\nNOTE:"
)

print(
    "This reconstructed score is diagnostic only."
)

print(
    "The primary baseline remains the original "
    "N-best rank-1 hypothesis."
)


# ============================================================
# BASELINE 3:
# NEURAL LM ONLY
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "DIAGNOSTIC: NEURAL-LM-ONLY SELECTION"
)

print(
    "=" * 78
)

nlm_only = []
nlm_ranks = []

for candidates in candidate_sets:

    neural_scores = np.array(
        [
            c[
                "neural_lm_avg_log_prob"
            ]
            for c in candidates
        ],
        dtype=np.float64,
    )

    best_idx = int(
        np.argmax(
            neural_scores
        )
    )

    nlm_only.append(
        candidates[
            best_idx
        ]["text"]
    )

    nlm_ranks.append(
        best_idx + 1
    )


nlm_metrics = compute_metrics(
    references,
    nlm_only,
)

print(
    f"CER : "
    f"{nlm_metrics['cer']:.4f}%"
)

print(
    f"WER : "
    f"{nlm_metrics['wer']:.4f}%"
)

print(
    f"Mean selected rank : "
    f"{np.mean(nlm_ranks):.2f}"
)


# ============================================================
# NEURAL LM FUSION
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "4-GRAM SCORE + CHARACTER NEURAL LM"
)

print(
    "=" * 78
)

print(
    "\nFusion formula:"
)

print(
    "  normalized 4-gram score"
)

print(
    "  + beta Ã— normalized Neural-LM score"
)


fusion_results = []


for beta in BETAS:

    selected_hypotheses = []

    selected_ranks = []

    changed_from_original = 0

    for candidates, original in zip(
        candidate_sets,
        original_top1,
    ):

        # ----------------------------------------------------
        # Base score.
        # ----------------------------------------------------

        base_scores = np.array(
            [
                c[
                    "reconstructed_4gram_score"
                ]
                for c in candidates
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Neural LM score.
        #
        # Average log probability avoids
        # simply preferring shorter strings.
        # ----------------------------------------------------

        neural_scores = np.array(
            [
                c[
                    "neural_lm_avg_log_prob"
                ]
                for c in candidates
            ],
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Per-utterance normalization.
        # ----------------------------------------------------

        base_z = zscore(
            base_scores
        )

        neural_z = zscore(
            neural_scores
        )

        combined_score = (
            base_z
            + beta * neural_z
        )

        best_idx = int(
            np.argmax(
                combined_score
            )
        )

        selected = candidates[
            best_idx
        ]

        selected_hypotheses.append(
            selected["text"]
        )

        selected_ranks.append(
            best_idx + 1
        )

        if (
            selected["text"]
            != original
        ):

            changed_from_original += 1

    metrics = compute_metrics(
        references,
        selected_hypotheses,
    )

    result = {
        "beta": beta,

        "cer": metrics["cer"],
        "wer": metrics["wer"],

        "char_errors": metrics[
            "char_errors"
        ],

        "word_errors": metrics[
            "word_errors"
        ],

        "ref_chars": metrics[
            "ref_chars"
        ],

        "ref_words": metrics[
            "ref_words"
        ],

        "average_selected_rank": float(
            np.mean(
                selected_ranks
            )
        ),

        "median_selected_rank": float(
            np.median(
                selected_ranks
            )
        ),

        "changed_from_original_pct": (
            100.0
            * changed_from_original
            / len(samples)
        ),

        "selected_ranks": (
            selected_ranks
        ),

        "selected_hypotheses": (
            selected_hypotheses
        ),
    }

    fusion_results.append(
        result
    )

    print(
        f"\nBeta = {beta:.2f}"
    )

    print(
        f"  CER        : "
        f"{metrics['cer']:.4f}%"
    )

    print(
        f"  WER        : "
        f"{metrics['wer']:.4f}%"
    )

    print(
        f"  Avg rank   : "
        f"{np.mean(selected_ranks):.2f}"
    )

    print(
        f"  Median rank: "
        f"{np.median(selected_ranks):.0f}"
    )

    print(
        f"  Changed    : "
        f"{100.0 * changed_from_original / len(samples):.2f}%"
    )


# ============================================================
# BEST CONFIGURATIONS
# ============================================================

best_cer = min(
    fusion_results,
    key=lambda x: x["cer"]
)

best_wer = min(
    fusion_results,
    key=lambda x: x["wer"]
)


print(
    "\n" + "=" * 78
)

print(
    "BEST NEURAL-LM FUSION RESULTS"
)

print(
    "=" * 78
)

print(
    "\nPRIMARY BASELINE:"
)

print(
    f"  CER = "
    f"{original_metrics['cer']:.4f}%"
)

print(
    f"  WER = "
    f"{original_metrics['wer']:.4f}%"
)


print(
    "\nBEST CER:"
)

print(
    f"  Beta = "
    f"{best_cer['beta']:.2f}"
)

print(
    f"  CER  = "
    f"{best_cer['cer']:.4f}%"
)

print(
    f"  WER  = "
    f"{best_cer['wer']:.4f}%"
)

print(
    f"  CER improvement vs Top-1 = "
    f"{original_metrics['cer'] - best_cer['cer']:.4f} pp"
)


print(
    "\nBEST WER:"
)

print(
    f"  Beta = "
    f"{best_wer['beta']:.2f}"
)

print(
    f"  CER  = "
    f"{best_wer['cer']:.4f}%"
)

print(
    f"  WER  = "
    f"{best_wer['wer']:.4f}%"
)

print(
    f"  WER improvement vs Top-1 = "
    f"{original_metrics['wer'] - best_wer['wer']:.4f} pp"
)


# ============================================================
# SANITY CHECKS
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "FINAL SANITY CHECKS"
)

print(
    "=" * 78
)


# ------------------------------------------------------------
# Check 1: 864 samples.
# ------------------------------------------------------------

assert (
    len(samples)
    == EXPECTED_VALIDATION_SAMPLES
)

print(
    "[PASS] 864 validation samples."
)


# ------------------------------------------------------------
# Check 2: all candidates have required fields.
# ------------------------------------------------------------

required_candidate_fields = [
    "text",
    "lm_score",
    "logit_score",
    "original_rank",
]

for candidates in candidate_sets:

    for candidate in candidates:

        for field in required_candidate_fields:

            if field not in candidate:

                raise RuntimeError(
                    f"Missing candidate field: "
                    f"{field}"
                )


print(
    "[PASS] Candidate fields are complete."
)


# ------------------------------------------------------------
# Check 3: Neural LM scores exist.
# ------------------------------------------------------------

for candidates in candidate_sets:

    for candidate in candidates:

        if (
            "neural_lm_avg_log_prob"
            not in candidate
        ):

            raise RuntimeError(
                "Missing Neural-LM score."
            )


print(
    "[PASS] Neural-LM scores exist "
    "for every candidate."
)


# ------------------------------------------------------------
# Check 4: Primary baseline is unchanged.
#
# We explicitly select candidate 0.
# ------------------------------------------------------------

baseline_again = compute_metrics(
    references,
    [
        candidates[0]["text"]
        for candidates in candidate_sets
    ],
)

if (
    abs(
        baseline_again["cer"]
        - original_metrics["cer"]
    )
    > 1e-12
):

    raise RuntimeError(
        "Primary baseline changed unexpectedly."
    )

if (
    abs(
        baseline_again["wer"]
        - original_metrics["wer"]
    )
    > 1e-12
):

    raise RuntimeError(
        "Primary baseline WER changed unexpectedly."
    )


print(
    "[PASS] Primary Top-1 baseline is stable."
)


# ------------------------------------------------------------
# Check 5: metric denominators.
# ------------------------------------------------------------

if (
    original_metrics["ref_chars"]
    != 81162
):

    raise RuntimeError(
        "Unexpected reference-character "
        "denominator."
    )

if (
    original_metrics["ref_words"]
    != 13478
):

    raise RuntimeError(
        "Unexpected reference-word "
        "denominator."
    )


print(
    "[PASS] Reference denominators match "
    "the established 864-sample evaluation."
)


# ------------------------------------------------------------
# Check 6: no official test data.
# ------------------------------------------------------------

if (
    "test" in str(
        NBEST_PATH
    ).lower()
):

    raise RuntimeError(
        "N-best path appears to reference "
        "a test set."
    )


print(
    "[PASS] Official 410-sample test "
    "was not used."
)


# ============================================================
# SAVE RESULTS
# ============================================================

output = {

    "experiment": (
        "B3-NARROW + 4-gram pooled "
        "Top-20 + character Neural LM"
    ),

    "experiment_status": (
        "validation_only"
    ),

    "device": str(
        DEVICE
    ),

    "num_validation_samples": (
        len(samples)
    ),

    "top_k": TOP_K,

    "lm_batch_size": (
        LM_BATCH_SIZE
    ),

    "neural_lm_checkpoint": str(
        LM_CHECKPOINT
    ),

    "vocabulary_path": str(
        VOCAB_PATH
    ),

    "n_best_source": str(
        NBEST_PATH
    ),

    "neural_lm": {
        "vocab_size": VOCAB_SIZE,
        "embedding_dim": embed_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "parameters": parameter_count,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "pad_id": PAD_ID,
        "unk_id": UNK_ID,
    },

    # --------------------------------------------------------
    # Primary baseline.
    # --------------------------------------------------------

    "primary_original_top1": {
        "cer": original_metrics[
            "cer"
        ],
        "wer": original_metrics[
            "wer"
        ],
        "char_errors": original_metrics[
            "char_errors"
        ],
        "word_errors": original_metrics[
            "word_errors"
        ],
        "ref_chars": original_metrics[
            "ref_chars"
        ],
        "ref_words": original_metrics[
            "ref_words"
        ],
    },

    # --------------------------------------------------------
    # Diagnostic reconstructed score.
    # --------------------------------------------------------

    "diagnostic_reconstructed_4gram": {
        "alpha": FOURGRAM_ALPHA,
        "cer": reconstructed_metrics[
            "cer"
        ],
        "wer": reconstructed_metrics[
            "wer"
        ],
        "average_rank": float(
            np.mean(
                reconstructed_ranks
            )
        ),
        "median_rank": float(
            np.median(
                reconstructed_ranks
            )
        ),
        "changed_from_original_pct": (
            100.0
            * changed_from_original
            / len(samples)
        ),
    },

    # --------------------------------------------------------
    # Neural LM only.
    # --------------------------------------------------------

    "diagnostic_neural_lm_only": {
        "cer": nlm_metrics[
            "cer"
        ],
        "wer": nlm_metrics[
            "wer"
        ],
        "average_rank": float(
            np.mean(
                nlm_ranks
            )
        ),
        "median_rank": float(
            np.median(
                nlm_ranks
            )
        ),
    },

    # --------------------------------------------------------
    # Fusion.
    # --------------------------------------------------------

    "fusion": {

        "fourgram_alpha": (
            FOURGRAM_ALPHA
        ),

        "betas": BETAS,

        "results": fusion_results,

    },

    # --------------------------------------------------------
    # Best configurations.
    # --------------------------------------------------------

    "best_cer": {
        "beta": best_cer[
            "beta"
        ],
        "cer": best_cer[
            "cer"
        ],
        "wer": best_cer[
            "wer"
        ],
        "improvement_vs_primary_cer_pp": (
            original_metrics[
                "cer"
            ]
            - best_cer[
                "cer"
            ]
        ),
    },

    "best_wer": {
        "beta": best_wer[
            "beta"
        ],
        "cer": best_wer[
            "cer"
        ],
        "wer": best_wer[
            "wer"
        ],
        "improvement_vs_primary_wer_pp": (
            original_metrics[
                "wer"
            ]
            - best_wer[
                "wer"
            ]
        ),
    },

}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 78
)

print(
    "SUCCESS â€” FINAL NEURAL LM EXPERIMENT COMPLETE"
)

print(
    "=" * 78
)

print(
    f"\nResults saved to:"
)

print(
    OUTPUT_JSON
)

print(
    "\nPrimary baseline:"
)

print(
    f"  CER = "
    f"{original_metrics['cer']:.4f}%"
)

print(
    f"  WER = "
    f"{original_metrics['wer']:.4f}%"
)

print(
    "\nBest Neural-LM CER:"
)

print(
    f"  Beta = "
    f"{best_cer['beta']:.2f}"
)

print(
    f"  CER = "
    f"{best_cer['cer']:.4f}%"
)

print(
    "\nBest Neural-LM WER:"
)

print(
    f"  Beta = "
    f"{best_wer['beta']:.2f}"
)

print(
    f"  WER = "
    f"{best_wer['wer']:.4f}%"
)

print(
    "\nOfficial 410-sample test:"
)

print(
    "  UNTOUCHED"
)

print(
    "\nNo further training was performed."
)
