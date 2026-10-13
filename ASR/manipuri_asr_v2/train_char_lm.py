"""
======================================================================
CHARACTER-LEVEL NEURAL LANGUAGE MODEL
FOR MANIPURI / MEITEILON ASR N-BEST RESCORING
======================================================================

Purpose
-------
Train an external character-level LSTM language model using the clean
Manipuri corpus.

The LM is trained ONLY on the text corpus.

It is NOT trained on:
    - ASR N-best hypotheses
    - ASR validation hypotheses
    - official ASR test data

Later, this LM will be used to rescore the existing B3 + 4-gram
Top-20 ASR hypotheses.

Architecture
------------
Character IDs
      |
Embedding: 64
      |
2-layer LSTM: hidden 256
      |
Linear
      |
Character probabilities

Dataset
-------
7,768 cleaned Manipuri utterances

Split
-----
90% train
5% validation
5% held-out LM test

Important
---------
The 864-sample ASR validation set is NOT used to train this LM.

The official 410-sample ASR test set remains untouched.
======================================================================
"""

from pathlib import Path
import json
import random
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader


# ======================================================================
# 1. PATHS
# ======================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

CORPUS_PATH = (
    SCRIPT_DIR
    / "kenlm_experiment"
    / "manipuri_corpus_clean.txt"
)

SAVE_DIR = (
    SCRIPT_DIR
    / "checkpoints"
    / "lm"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_PATH = (
    SAVE_DIR
    / "char_lstm_lm_best.pth"
)

VOCAB_PATH = (
    SAVE_DIR
    / "char_vocab.json"
)

METADATA_PATH = (
    SAVE_DIR
    / "char_lm_metadata.json"
)

HISTORY_PATH = (
    SAVE_DIR
    / "training_history.json"
)


# ======================================================================
# 2. CONFIGURATION
# ======================================================================

SEED = 42

# Corpus split
TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

# Model
EMBED_DIM = 64
HIDDEN_DIM = 256
LAYERS = 2
DROPOUT = 0.20

# Training
SEQ_LEN = 64
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

NUM_WORKERS = 0

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ======================================================================
# 3. REPRODUCIBILITY
# ======================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ======================================================================
# 4. TEXT NORMALIZATION
# ======================================================================

def normalize_text(text):
    """
    NFC normalization followed by whitespace normalization.
    """

    text = unicodedata.normalize(
        "NFC",
        str(text),
    )

    return " ".join(
        text.split()
    ).strip()


# ======================================================================
# 5. LOAD CORPUS
# ======================================================================

print("=" * 78)
print("CHARACTER-LEVEL MANIPURI / MEITEILON NEURAL LANGUAGE MODEL")
print("=" * 78)

print()
print(f"Device        : {DEVICE}")
print(f"Corpus        : {CORPUS_PATH}")
print(f"Save directory: {SAVE_DIR}")

if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"\nCorpus not found:\n{CORPUS_PATH}"
    )

print()
print("Loading corpus...")

with open(
    CORPUS_PATH,
    "r",
    encoding="utf-8",
) as f:

    raw_lines = [
        line.rstrip("\n")
        for line in f
    ]


# ======================================================================
# 6. NORMALIZE CORPUS
# ======================================================================

text_data = []

for line in raw_lines:

    normalized = normalize_text(line)

    if normalized:
        text_data.append(normalized)


print(
    f"Raw lines       : {len(raw_lines):,}"
)

print(
    f"Valid lines     : {len(text_data):,}"
)


if len(text_data) < 100:
    raise RuntimeError(
        "Corpus is unexpectedly small."
    )


# ======================================================================
# 7. TRAIN / VALIDATION / TEST SPLIT
# ======================================================================

indices = list(
    range(len(text_data))
)

rng = random.Random(SEED)

rng.shuffle(indices)

n_total = len(indices)

n_train = int(
    TRAIN_RATIO * n_total
)

n_val = int(
    VAL_RATIO * n_total
)

train_indices = indices[
    :n_train
]

val_indices = indices[
    n_train:n_train + n_val
]

test_indices = indices[
    n_train + n_val:
]

train_texts = [
    text_data[i]
    for i in train_indices
]

val_texts = [
    text_data[i]
    for i in val_indices
]

test_texts = [
    text_data[i]
    for i in test_indices
]


print()
print("=" * 78)
print("CORPUS SPLIT")
print("=" * 78)

print(
    f"Total : {n_total:,}"
)

print(
    f"Train : {len(train_texts):,}"
)

print(
    f"Val   : {len(val_texts):,}"
)

print(
    f"Test  : {len(test_texts):,}"
)

print(
    f"Ratios: "
    f"{len(train_texts)/n_total:.3f} / "
    f"{len(val_texts)/n_total:.3f} / "
    f"{len(test_texts)/n_total:.3f}"
)


# ======================================================================
# 8. BUILD CHARACTER VOCABULARY
# ======================================================================
#
# IMPORTANT:
# Vocabulary is constructed ONLY from training text.
#
# This avoids allowing held-out validation/test text to influence
# the LM vocabulary.
# ======================================================================

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

special_tokens = [
    PAD_TOKEN,
    UNK_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN,
]

train_characters = sorted(
    set(
        "".join(train_texts)
    )
)

itos = (
    special_tokens
    + train_characters
)

stoi = {
    token: idx
    for idx, token in enumerate(itos)
}

PAD_ID = stoi[PAD_TOKEN]
UNK_ID = stoi[UNK_TOKEN]
BOS_ID = stoi[BOS_TOKEN]
EOS_ID = stoi[EOS_TOKEN]

VOCAB_SIZE = len(itos)


print()
print("=" * 78)
print("CHARACTER VOCABULARY")
print("=" * 78)

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


# ======================================================================
# 9. ENCODING
# ======================================================================

def encode_text(text):
    """
    Convert a sentence into:

        <bos> characters <eos>
    """

    ids = [
        BOS_ID
    ]

    for char in text:

        ids.append(
            stoi.get(
                char,
                UNK_ID,
            )
        )

    ids.append(
        EOS_ID
    )

    return ids


# ======================================================================
# 10. DATASET
# ======================================================================

class CharLMDataset(Dataset):

    def __init__(
        self,
        texts,
        seq_len,
    ):

        self.samples = []

        for text in texts:

            ids = encode_text(text)

            # Need at least input + target.
            if len(ids) < 2:
                continue

            # ----------------------------------------------------------
            # Chunk each sentence independently.
            #
            # We intentionally DO NOT concatenate all sentences into
            # one giant stream.
            # ----------------------------------------------------------

            start = 0

            while start < len(ids) - 1:

                chunk = ids[
                    start:start + seq_len + 1
                ]

                if len(chunk) >= 2:

                    self.samples.append(
                        chunk
                    )

                start += seq_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        ids = self.samples[index]

        x = ids[:-1]

        y = ids[1:]

        return (
            torch.tensor(
                x,
                dtype=torch.long,
            ),
            torch.tensor(
                y,
                dtype=torch.long,
            ),
        )


# ======================================================================
# 11. COLLATE FUNCTION
# ======================================================================

def collate_batch(batch):

    xs, ys = zip(*batch)

    max_length = max(
        len(x)
        for x in xs
    )

    padded_x = torch.full(
        (
            len(xs),
            max_length,
        ),
        PAD_ID,
        dtype=torch.long,
    )

    padded_y = torch.full(
        (
            len(ys),
            max_length,
        ),
        PAD_ID,
        dtype=torch.long,
    )

    for i, (
        x,
        y,
    ) in enumerate(
        zip(xs, ys)
    ):

        padded_x[
            i,
            :len(x)
        ] = x

        padded_y[
            i,
            :len(y)
        ] = y

    return (
        padded_x,
        padded_y,
    )


# ======================================================================
# 12. DATASETS
# ======================================================================

train_dataset = CharLMDataset(
    train_texts,
    SEQ_LEN,
)

val_dataset = CharLMDataset(
    val_texts,
    SEQ_LEN,
)

test_dataset = CharLMDataset(
    test_texts,
    SEQ_LEN,
)


print()
print("=" * 78)
print("SEQUENCE DATASET")
print("=" * 78)

print(
    f"Train sequences : "
    f"{len(train_dataset):,}"
)

print(
    f"Val sequences   : "
    f"{len(val_dataset):,}"
)

print(
    f"Test sequences  : "
    f"{len(test_dataset):,}"
)


# ======================================================================
# 13. DATA LOADERS
# ======================================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_batch,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_batch,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
    collate_fn=collate_batch,
    pin_memory=torch.cuda.is_available(),
)


# ======================================================================
# 14. MODEL
# ======================================================================

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
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_ID,
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

        self.dropout = nn.Dropout(
            dropout
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size,
        )

    def forward(self, x):

        embedded = self.embedding(x)

        output, _ = self.lstm(
            embedded
        )

        output = self.dropout(
            output
        )

        logits = self.fc(
            output
        )

        return logits


model = CharLSTM(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=LAYERS,
    dropout=DROPOUT,
).to(DEVICE)


# ======================================================================
# 15. LOSS / OPTIMIZER
# ======================================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# ======================================================================
# 16. PARAMETER COUNT
# ======================================================================

total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print()
print("=" * 78)
print("MODEL CONFIGURATION")
print("=" * 78)

print(
    f"Embedding dimension : {EMBED_DIM}"
)

print(
    f"Hidden dimension    : {HIDDEN_DIM}"
)

print(
    f"LSTM layers         : {LAYERS}"
)

print(
    f"Dropout             : {DROPOUT}"
)

print(
    f"Sequence length     : {SEQ_LEN}"
)

print(
    f"Batch size          : {BATCH_SIZE}"
)

print(
    f"Learning rate       : {LR}"
)

print(
    f"Weight decay        : {WEIGHT_DECAY}"
)

print(
    f"Gradient clip       : {GRAD_CLIP}"
)

print(
    f"Total parameters    : "
    f"{total_params:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_params:,}"
)


# ======================================================================
# 17. TRAIN / EVALUATION FUNCTION
# ======================================================================

def run_epoch(
    loader,
    training,
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    if training:
        context = torch.enable_grad()
    else:
        context = torch.no_grad()

    with context:

        for x, y in loader:

            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            if training:

                optimizer.zero_grad(
                    set_to_none=True
                )

            logits = model(x)

            loss = criterion(
                logits.reshape(
                    -1,
                    VOCAB_SIZE,
                ),
                y.reshape(-1),
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP,
                )

                optimizer.step()

            valid_tokens = (
                y != PAD_ID
            ).sum().item()

            total_loss += (
                loss.item()
                * valid_tokens
            )

            total_tokens += (
                valid_tokens
            )

    return (
        total_loss
        / max(
            total_tokens,
            1,
        )
    )


# ======================================================================
# 18. PERPLEXITY
# ======================================================================

def calculate_perplexity(loss):

    # Avoid numerical overflow.
    return float(
        np.exp(
            min(
                loss,
                20.0,
            )
        )
    )


# ======================================================================
# 19. TRAINING
# ======================================================================

print()
print("=" * 78)
print("STARTING LSTM LANGUAGE MODEL TRAINING")
print("=" * 78)

best_val_loss = float("inf")
best_epoch = 0

history = []

for epoch in range(
    1,
    EPOCHS + 1,
):

    train_loss = run_epoch(
        train_loader,
        training=True,
    )

    val_loss = run_epoch(
        val_loader,
        training=False,
    )

    train_ppl = calculate_perplexity(
        train_loss
    )

    val_ppl = calculate_perplexity(
        val_loss
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_perplexity": train_ppl,
            "val_perplexity": val_ppl,
        }
    )

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.5f} | "
        f"Val Loss: {val_loss:.5f} | "
        f"Train PPL: {train_ppl:.3f} | "
        f"Val PPL: {val_ppl:.3f}"
    )

    # --------------------------------------------------------------
    # Save best validation checkpoint
    # --------------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_epoch = epoch

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "vocab_size":
                    VOCAB_SIZE,

                "embed_dim":
                    EMBED_DIM,

                "hidden_dim":
                    HIDDEN_DIM,

                "layers":
                    LAYERS,

                "dropout":
                    DROPOUT,

                "seq_len":
                    SEQ_LEN,

                "pad_id":
                    PAD_ID,

                "unk_id":
                    UNK_ID,

                "bos_id":
                    BOS_ID,

                "eos_id":
                    EOS_ID,

                "best_epoch":
                    best_epoch,

                "best_val_loss":
                    best_val_loss,

                "best_val_perplexity":
                    calculate_perplexity(
                        best_val_loss
                    ),

                "seed":
                    SEED,
            },
            MODEL_PATH,
        )

        print(
            f"  [BEST] "
            f"Validation checkpoint saved "
            f"(epoch {epoch})"
        )


# ======================================================================
# 20. LOAD BEST CHECKPOINT
# ======================================================================

print()
print("=" * 78)
print("LOADING BEST CHECKPOINT")
print("=" * 78)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)

print(
    f"Best epoch : {best_epoch}"
)

print(
    f"Best val loss : "
    f"{best_val_loss:.5f}"
)

print(
    f"Best val PPL : "
    f"{calculate_perplexity(best_val_loss):.3f}"
)


# ======================================================================
# 21. HELD-OUT TEST EVALUATION
# ======================================================================

print()
print("=" * 78)
print("HELD-OUT LANGUAGE MODEL TEST")
print("=" * 78)

test_loss = run_epoch(
    test_loader,
    training=False,
)

test_ppl = calculate_perplexity(
    test_loss
)

print(
    f"Test loss : {test_loss:.5f}"
)

print(
    f"Test PPL  : {test_ppl:.3f}"
)


# ======================================================================
# 22. SAVE VOCABULARY
# ======================================================================

vocab_data = {
    "itos": itos,
    "stoi": stoi,
    "pad_token": PAD_TOKEN,
    "unk_token": UNK_TOKEN,
    "bos_token": BOS_TOKEN,
    "eos_token": EOS_TOKEN,
    "pad_id": PAD_ID,
    "unk_id": UNK_ID,
    "bos_id": BOS_ID,
    "eos_id": EOS_ID,
    "vocab_size": VOCAB_SIZE,
}

with open(
    VOCAB_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        vocab_data,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ======================================================================
# 23. SAVE TRAINING HISTORY
# ======================================================================

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        history,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ======================================================================
# 24. SAVE METADATA
# ======================================================================

metadata = {

    "experiment":
        "External Character-Level Neural LM",

    "corpus":
        str(CORPUS_PATH),

    "total_corpus_lines":
        len(text_data),

    "train_lines":
        len(train_texts),

    "validation_lines":
        len(val_texts),

    "test_lines":
        len(test_texts),

    "split_ratios": {
        "train":
            TRAIN_RATIO,
        "validation":
            VAL_RATIO,
        "test":
            TEST_RATIO,
    },

    "vocabulary_size":
        VOCAB_SIZE,

    "embedding_dimension":
        EMBED_DIM,

    "hidden_dimension":
        HIDDEN_DIM,

    "lstm_layers":
        LAYERS,

    "dropout":
        DROPOUT,

    "sequence_length":
        SEQ_LEN,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LR,

    "weight_decay":
        WEIGHT_DECAY,

    "gradient_clip":
        GRAD_CLIP,

    "optimizer":
        "Adam",

    "seed":
        SEED,

    "device":
        str(DEVICE),

    "best_epoch":
        best_epoch,

    "best_validation_loss":
        best_val_loss,

    "best_validation_perplexity":
        calculate_perplexity(
            best_val_loss
        ),

    "heldout_test_loss":
        test_loss,

    "heldout_test_perplexity":
        test_ppl,

    "model_path":
        str(MODEL_PATH),

    "vocabulary_path":
        str(VOCAB_PATH),

    "history_path":
        str(HISTORY_PATH),
}

with open(
    METADATA_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2,
    )


# ======================================================================
# 25. FINAL REPORT
# ======================================================================

print()
print("=" * 78)
print("FINAL CHARACTER LM RESULTS")
print("=" * 78)

print()
print(
    f"Corpus lines       : "
    f"{len(text_data):,}"
)

print(
    f"Training lines     : "
    f"{len(train_texts):,}"
)

print(
    f"Validation lines   : "
    f"{len(val_texts):,}"
)

print(
    f"Held-out test lines: "
    f"{len(test_texts):,}"
)

print()
print(
    f"Vocabulary size    : "
    f"{VOCAB_SIZE}"
)

print(
    f"Model parameters   : "
    f"{total_params:,}"
)

print()
print(
    f"Best epoch         : "
    f"{best_epoch}"
)

print(
    f"Best Val Loss      : "
    f"{best_val_loss:.5f}"
)

print(
    f"Best Val PPL       : "
    f"{calculate_perplexity(best_val_loss):.3f}"
)

print(
    f"Held-out Test Loss : "
    f"{test_loss:.5f}"
)

print(
    f"Held-out Test PPL  : "
    f"{test_ppl:.3f}"
)

print()
print(
    f"Model saved        : "
    f"{MODEL_PATH}"
)

print(
    f"Vocabulary saved   : "
    f"{VOCAB_PATH}"
)

print(
    f"Metadata saved     : "
    f"{METADATA_PATH}"
)

print(
    f"History saved      : "
    f"{HISTORY_PATH}"
)

print()
print("=" * 78)
print("SUCCESS")
print("=" * 78)