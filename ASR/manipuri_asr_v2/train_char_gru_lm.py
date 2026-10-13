from pathlib import Path
import json
import random
import unicodedata

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CHARACTER-LEVEL MANIPURI / MEITEILON GRU LANGUAGE MODEL
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------
# INPUT CORPUS
# ------------------------------------------------------------

CORPUS_PATH = (
    SCRIPT_DIR
    / "kenlm_experiment"
    / "manipuri_corpus_clean.txt"
)

# ------------------------------------------------------------
# OUTPUT DIRECTORY
# ------------------------------------------------------------

SAVE_DIR = (
    SCRIPT_DIR
    / "checkpoints"
    / "lm_gru"
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = SAVE_DIR / "char_gru_lm_best.pth"
VOCAB_PATH = SAVE_DIR / "char_gru_vocab.json"
META_PATH = SAVE_DIR / "char_gru_metadata.json"
HISTORY_PATH = SAVE_DIR / "gru_training_history.json"


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

EMBED_DIM = 64
HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.2

SEQ_LEN = 64
BATCH_SIZE = 128

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0

EPOCHS = 20

TRAIN_RATIO = 0.90
VAL_RATIO = 0.05
TEST_RATIO = 0.05

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


# ============================================================
# HEADER
# ============================================================

print("=" * 72)
print("CHARACTER-LEVEL MANIPURI / MEITEILON GRU LANGUAGE MODEL")
print("=" * 72)

print(f"Device : {DEVICE}")
print(f"Corpus : {CORPUS_PATH}")
print(f"Output : {SAVE_DIR}")


# ============================================================
# CHECK CORPUS
# ============================================================

if not CORPUS_PATH.exists():
    raise FileNotFoundError(
        f"\nCorpus not found:\n{CORPUS_PATH}\n"
    )


# ============================================================
# LOAD + NORMALIZE CORPUS
# ============================================================

with open(
    CORPUS_PATH,
    "r",
    encoding="utf-8"
) as f:

    raw_lines = [
        line.strip()
        for line in f
    ]


lines = []

for line in raw_lines:

    line = unicodedata.normalize(
        "NFC",
        line
    )

    line = " ".join(
        line.split()
    )

    if line:
        lines.append(line)


print()
print("CORPUS")
print("-" * 72)

print(
    f"Raw lines   : {len(raw_lines):,}"
)

print(
    f"Valid lines : {len(lines):,}"
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

rng = random.Random(SEED)

indices = list(
    range(len(lines))
)

rng.shuffle(indices)

n_total = len(indices)

n_train = int(
    n_total * TRAIN_RATIO
)

n_val = int(
    n_total * VAL_RATIO
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


train_lines = [
    lines[i]
    for i in train_indices
]

val_lines = [
    lines[i]
    for i in val_indices
]

test_lines = [
    lines[i]
    for i in test_indices
]


print()
print("CORPUS SPLIT")
print("-" * 72)

print(
    f"Total : {len(lines):,}"
)

print(
    f"Train : {len(train_lines):,}"
)

print(
    f"Val   : {len(val_lines):,}"
)

print(
    f"Test  : {len(test_lines):,}"
)

print(
    f"Ratios: "
    f"{len(train_lines) / n_total:.3f} / "
    f"{len(val_lines) / n_total:.3f} / "
    f"{len(test_lines) / n_total:.3f}"
)


# ============================================================
# CHARACTER VOCABULARY
# IMPORTANT:
# Vocabulary is built ONLY from training text.
# ============================================================

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<eos>"

SPECIAL_TOKENS = [
    PAD,
    UNK,
    BOS,
    EOS
]


train_characters = sorted(
    set(
        "".join(train_lines)
    )
)


itos = (
    SPECIAL_TOKENS
    + train_characters
)


stoi = {
    token: idx
    for idx, token in enumerate(itos)
}


PAD_ID = stoi[PAD]
UNK_ID = stoi[UNK]
BOS_ID = stoi[BOS]
EOS_ID = stoi[EOS]

VOCAB_SIZE = len(itos)


print()
print("CHARACTER VOCABULARY")
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
# SAVE VOCABULARY
# ============================================================

vocab_data = {
    "itos": itos,
    "stoi": stoi,

    "pad_token": PAD,
    "unk_token": UNK,
    "bos_token": BOS,
    "eos_token": EOS,

    "pad_id": PAD_ID,
    "unk_id": UNK_ID,
    "bos_id": BOS_ID,
    "eos_id": EOS_ID,
}


with open(
    VOCAB_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        vocab_data,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# TEXT → IDS
# ============================================================

def encode_text(text):

    ids = [
        stoi.get(
            ch,
            UNK_ID
        )
        for ch in text
    ]

    return (
        [BOS_ID]
        + ids
        + [EOS_ID]
    )


# ============================================================
# CREATE SEQUENCES
# ============================================================

def make_sequences(text_lines):

    sequences = []

    for text in text_lines:

        ids = encode_text(text)

        if len(ids) < 2:
            continue

        for start in range(
            0,
            len(ids) - 1,
            SEQ_LEN
        ):

            chunk = ids[
                start:start + SEQ_LEN + 1
            ]

            if len(chunk) >= 2:
                sequences.append(chunk)

    return sequences


train_sequences = make_sequences(
    train_lines
)

val_sequences = make_sequences(
    val_lines
)

test_sequences = make_sequences(
    test_lines
)


print()
print("SEQUENCE DATASET")
print("-" * 72)

print(
    f"Train sequences : "
    f"{len(train_sequences):,}"
)

print(
    f"Val sequences   : "
    f"{len(val_sequences):,}"
)

print(
    f"Test sequences  : "
    f"{len(test_sequences):,}"
)


# ============================================================
# DATASET
# ============================================================

class CharSequenceDataset(Dataset):

    def __init__(
        self,
        sequences
    ):

        self.sequences = sequences

    def __len__(self):

        return len(
            self.sequences
        )

    def __getitem__(
        self,
        idx
    ):

        seq = self.sequences[idx]

        x = torch.tensor(
            seq[:-1],
            dtype=torch.long
        )

        y = torch.tensor(
            seq[1:],
            dtype=torch.long
        )

        return x, y


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    xs, ys = zip(*batch)

    max_len = max(
        max(len(x) for x in xs),
        max(len(y) for y in ys)
    )

    batch_x = torch.full(
        (
            len(xs),
            max_len
        ),
        PAD_ID,
        dtype=torch.long
    )

    batch_y = torch.full(
        (
            len(ys),
            max_len
        ),
        PAD_ID,
        dtype=torch.long
    )

    for i, (x, y) in enumerate(
        zip(xs, ys)
    ):

        batch_x[
            i,
            :len(x)
        ] = x

        batch_y[
            i,
            :len(y)
        ] = y

    return batch_x, batch_y


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = CharSequenceDataset(
    train_sequences
)

val_dataset = CharSequenceDataset(
    val_sequences
)

test_dataset = CharSequenceDataset(
    test_sequences
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available()
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
        dropout
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=PAD_ID
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


model = CharGRU(
    vocab_size=VOCAB_SIZE,
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(DEVICE)


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
print("MODEL")
print("-" * 72)

print(
    f"Embedding       : {EMBED_DIM}"
)

print(
    f"Hidden          : {HIDDEN_DIM}"
)

print(
    f"GRU layers      : {NUM_LAYERS}"
)

print(
    f"Dropout         : {DROPOUT}"
)

print(
    f"Sequence length : {SEQ_LEN}"
)

print(
    f"Batch size      : {BATCH_SIZE}"
)

print(
    f"Learning rate   : {LEARNING_RATE}"
)

print(
    f"Weight decay    : {WEIGHT_DECAY}"
)

print(
    f"Gradient clip   : {GRAD_CLIP}"
)

print(
    f"Epochs          : {EPOCHS}"
)

print(
    f"Total params    : {total_params:,}"
)

print(
    f"Trainable       : {trainable_params:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_ID
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# RUN EPOCH
# ============================================================

def run_epoch(
    loader,
    training=True
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            logits = model(x)

            loss = criterion(
                logits.reshape(
                    -1,
                    VOCAB_SIZE
                ),
                y.reshape(-1)
            )

        if training:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            optimizer.step()

        valid_tokens = (
            y != PAD_ID
        ).sum().item()

        total_loss += (
            loss.item()
            * valid_tokens
        )

        total_tokens += valid_tokens

    return (
        total_loss
        / max(total_tokens, 1)
    )


# ============================================================
# PERPLEXITY
# ============================================================

def calculate_ppl(loss):

    return float(
        np.exp(
            min(loss, 20.0)
        )
    )


# ============================================================
# TRAIN
# ============================================================

history = []

best_val_loss = float("inf")
best_epoch = 0


print()
print("=" * 72)
print("TRAINING")
print("=" * 72)


for epoch in range(
    1,
    EPOCHS + 1
):

    train_loss = run_epoch(
        train_loader,
        training=True
    )

    val_loss = run_epoch(
        val_loader,
        training=False
    )

    train_ppl = calculate_ppl(
        train_loss
    )

    val_ppl = calculate_ppl(
        val_loss
    )

    history_row = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_ppl": train_ppl,
        "val_ppl": val_ppl
    }

    history.append(
        history_row
    )

    print(
        f"Epoch {epoch:02d} "
        f"Train Loss {train_loss:.5f} "
        f"Val {val_loss:.5f} "
        f"Train PPL {train_ppl:.3f} "
        f"Val PPL {val_ppl:.3f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            f"  -> BEST MODEL SAVED "
            f"(epoch={epoch}, "
            f"val_ppl={val_ppl:.3f})"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

with open(
    HISTORY_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        history,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

best_state = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    best_state
)


# ============================================================
# HELD-OUT TEXT TEST
# ============================================================

test_loss = run_epoch(
    test_loader,
    training=False
)

test_ppl = calculate_ppl(
    test_loss
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "model_type":
        "character_level_gru",

    "language":
        "Manipuri / Meiteilon",

    "corpus":
        str(CORPUS_PATH),

    "seed":
        SEED,

    "corpus_lines":
        len(lines),

    "train_lines":
        len(train_lines),

    "val_lines":
        len(val_lines),

    "test_lines":
        len(test_lines),

    "train_sequences":
        len(train_sequences),

    "val_sequences":
        len(val_sequences),

    "test_sequences":
        len(test_sequences),

    "vocab_size":
        VOCAB_SIZE,

    "embedding_dim":
        EMBED_DIM,

    "hidden_dim":
        HIDDEN_DIM,

    "num_layers":
        NUM_LAYERS,

    "dropout":
        DROPOUT,

    "sequence_length":
        SEQ_LEN,

    "batch_size":
        BATCH_SIZE,

    "learning_rate":
        LEARNING_RATE,

    "weight_decay":
        WEIGHT_DECAY,

    "gradient_clip":
        GRAD_CLIP,

    "epochs":
        EPOCHS,

    "best_epoch":
        best_epoch,

    "best_val_loss":
        best_val_loss,

    "best_val_ppl":
        calculate_ppl(
            best_val_loss
        ),

    "test_loss":
        test_loss,

    "test_ppl":
        test_ppl,

    "model_path":
        str(MODEL_PATH),

    "vocab_path":
        str(VOCAB_PATH)
}


with open(
    META_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 72)
print("FINAL GRU LANGUAGE MODEL RESULTS")
print("=" * 72)

print(
    f"Best epoch       : {best_epoch}"
)

print(
    f"Best validation PPL : "
    f"{calculate_ppl(best_val_loss):.3f}"
)

print(
    f"Held-out test PPL : "
    f"{test_ppl:.3f}"
)

print()
print("FILES SAVED")
print("-" * 72)

print(
    f"Model    : {MODEL_PATH}"
)

print(
    f"Vocab    : {VOCAB_PATH}"
)

print(
    f"Metadata : {META_PATH}"
)

print(
    f"History  : {HISTORY_PATH}"
)

print()
print("=" * 72)
print("GRU TRAINING COMPLETE")
print("=" * 72)