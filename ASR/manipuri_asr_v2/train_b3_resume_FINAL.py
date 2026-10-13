from pathlib import Path
import os
import random
import json
import time
import io
import wave
import gc
import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from pyctcdecode import build_ctcdecoder


# ============================================================
# B3-NARROW RESUME
#
# Resume:
#     Epoch 8 -> Epoch 9 -> Epoch 10 -> Epoch 11 -> Epoch 12
#
# Official test set is NOT touched.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()
ASR = ROOT.parent

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"

ENCODER_MODEL = Path(
    r"E:\hf_models\w2v-bert-2.0"
)

B2_CHECKPOINT = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "regularized"
    / "FINAL_B2_MODEL.pt"
)

B3_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
)

START_CHECKPOINT = B3_DIR / "epoch_08.pt"

VOCAB_PATH = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

UNIGRAM_PATH = (
    ROOT
    / "kenlm_experiment"
    / "manipuri_unigrams.txt"
)

B3_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TRAINING CONFIG
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAMPLE_RATE = 16000
FEATURE_DIM = 1024
BOTTLENECK = 256

SEED = 42

BATCH_SIZE = 1
GRAD_ACCUM = 16

LR_ENCODER = 5e-6
LR_HEAD = 2e-4

WEIGHT_DECAY = 0.02
MAX_GRAD_NORM = 1.0

# ============================================================
# IMPORTANT:
# We are RESUMING from Epoch 8.
# ============================================================

START_EPOCH = 9
END_EPOCH = 12

NUM_WORKERS = 0

AMP = True

TRAINABLE_LAYERS = [22, 23]

BEAM_WIDTH = 500

PARQUET_CACHE_SIZE = 4

SPECIAL_BLANK = "<blank>"
SPECIAL_UNK = "<unk>"


USE_BF16 = (
    DEVICE.type == "cuda"
    and torch.cuda.is_bf16_supported()
)

AMP_DTYPE = (
    torch.bfloat16
    if USE_BF16
    else torch.float32
)


# ============================================================
# SEED
# ============================================================

def seed_everything(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# TEXT
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = unicodedata.normalize(
        "NFC",
        str(text)
    )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# AUDIO
# ============================================================

def decode_pcm(
    raw,
    sample_width
):

    if sample_width == 1:

        audio = np.frombuffer(
            raw,
            dtype=np.uint8
        ).astype(np.float32)

        return (
            audio - 128.0
        ) / 128.0


    if sample_width == 2:

        return (
            np.frombuffer(
                raw,
                dtype=np.int16
            ).astype(np.float32)
            / 32768.0
        )


    if sample_width == 3:

        b = np.frombuffer(
            raw,
            dtype=np.uint8
        )

        if len(b) % 3:
            raise ValueError(
                "Invalid 24-bit PCM byte length"
            )

        b = b.reshape(
            -1,
            3
        )

        audio = (
            b[:, 0].astype(np.int32)
            |
            (b[:, 1].astype(np.int32) << 8)
            |
            (b[:, 2].astype(np.int32) << 16)
        )

        mask = audio & 0x800000

        audio = audio - (
            mask << 1
        )

        return (
            audio.astype(np.float32)
            / 8388608.0
        )


    if sample_width == 4:

        return (
            np.frombuffer(
                raw,
                dtype=np.int32
            ).astype(np.float32)
            / 2147483648.0
        )


    raise ValueError(
        f"Unsupported PCM sample width: "
        f"{sample_width}"
    )


@lru_cache(
    maxsize=PARQUET_CACHE_SIZE
)
def _read_parquet_cached(
    parquet_file
):

    return pd.read_parquet(
        str(parquet_file),
        columns=["audio"]
    )


def load_parquet_audio(
    parquet_file,
    row_index
):

    parquet_file = str(
        parquet_file
    )

    if not os.path.exists(
        parquet_file
    ):
        raise FileNotFoundError(
            parquet_file
        )

    df = _read_parquet_cached(
        parquet_file
    )

    obj = df.iloc[
        int(row_index)
    ]["audio"]

    if not isinstance(
        obj,
        dict
    ):
        raise ValueError(
            f"Unexpected audio object: "
            f"{type(obj)}"
        )

    raw_bytes = obj.get(
        "bytes"
    )

    if raw_bytes is None:
        raise ValueError(
            "IndicVoices audio object "
            "has no bytes"
        )

    with wave.open(
        io.BytesIO(raw_bytes),
        "rb"
    ) as wf:

        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sr = wf.getframerate()
        frames = wf.getnframes()

        raw = wf.readframes(
            frames
        )

    audio = decode_pcm(
        raw,
        width
    )

    if channels > 1:

        audio = audio.reshape(
            -1,
            channels
        ).mean(
            axis=1
        )

    return audio, sr


def normalize_audio(
    audio,
    sample_rate
):

    audio = torch.as_tensor(
        audio,
        dtype=torch.float32
    )

    if audio.ndim == 2:

        audio = audio.mean(
            dim=(
                0
                if audio.shape[0] <= 8
                else 1
            )
        )

    elif audio.ndim > 2:

        audio = audio.squeeze()

    if sample_rate != SAMPLE_RATE:

        audio = (
            torchaudio.functional.resample(
                audio,
                sample_rate,
                SAMPLE_RATE
            )
        )

    return audio.contiguous()


def resolve_audio(row):

    source = str(
        row.get(
            "source",
            ""
        )
    ).strip().lower()

    if source == "indicvoices":

        audio, sr = load_parquet_audio(
            row["parquet_file"],
            int(row["parquet_row"])
        )

        return normalize_audio(
            audio,
            sr
        )

    path = row.get(
        "audio_path",
        ""
    )

    if (
        pd.isna(path)
        or not str(path).strip()
    ):

        raise RuntimeError(
            f"Invalid audio_path "
            f"for source={source}"
        )

    path = str(
        path
    ).strip()

    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )

    array, sr = sf.read(
        path,
        dtype="float32",
        always_2d=False
    )

    return normalize_audio(
        torch.from_numpy(
            np.asarray(array)
        ),
        sr
    )


# ============================================================
# VOCAB
# ============================================================

def load_vocab():

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        vocab = json.load(f)

    if (
        len(vocab) != 60
        or vocab.get(
            SPECIAL_BLANK
        ) != 0
    ):

        raise RuntimeError(
            f"Unexpected vocab: "
            f"size={len(vocab)}, "
            f"blank={vocab.get(SPECIAL_BLANK)}"
        )

    return vocab


def encode_text(
    text,
    vocab
):

    return torch.tensor(
        [
            vocab.get(
                ch,
                vocab[SPECIAL_UNK]
            )
            for ch in normalize_text(text)
        ],
        dtype=torch.long
    )


# ============================================================
# DATASET
# ============================================================

class AudioCTCDataset(
    Dataset
):

    def __init__(
        self,
        df,
        vocab
    ):

        self.df = (
            df.reset_index(
                drop=True
            )
        )

        self.vocab = vocab


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[
            idx
        ]

        return {
            "audio":
                resolve_audio(row),

            "target":
                encode_text(
                    row["text"],
                    self.vocab
                ),

            "text":
                normalize_text(
                    row["text"]
                )
        }


def collate_fn(batch):

    max_audio = max(
        x["audio"].shape[0]
        for x in batch
    )

    audios = torch.zeros(
        len(batch),
        max_audio,
        dtype=torch.float32
    )

    lengths = []
    targets = []
    target_lengths = []
    texts = []

    for i, x in enumerate(batch):

        n = x["audio"].shape[0]

        audios[
            i,
            :n
        ] = x["audio"]

        lengths.append(n)

        targets.append(
            x["target"]
        )

        target_lengths.append(
            x["target"].shape[0]
        )

        texts.append(
            x["text"]
        )

    return {
        "audio":
            audios,

        "audio_lengths":
            torch.tensor(
                lengths
            ),

        "targets":
            torch.cat(
                targets
            ),

        "target_lengths":
            torch.tensor(
                target_lengths
            ),

        "texts":
            texts
    }


# ============================================================
# ADAPTER
# ============================================================

class ResidualBottleneckAdapter(
    nn.Module
):

    def __init__(
        self,
        dim=1024,
        bottleneck=256
    ):

        super().__init__()

        self.norm_in = nn.LayerNorm(
            dim
        )

        self.down = nn.Linear(
            dim,
            bottleneck
        )

        self.act = nn.GELU()

        self.up = nn.Linear(
            bottleneck,
            dim
        )

        self.norm_out = nn.LayerNorm(
            dim
        )

        nn.init.normal_(
            self.up.weight,
            mean=0.0,
            std=1e-3
        )

        nn.init.zeros_(
            self.up.bias
        )


    def forward(self, x):

        z = self.norm_in(
            x
        )

        z = self.down(
            z
        )

        z = self.act(
            z
        )

        z = self.up(
            z
        )

        return self.norm_out(
            x + z
        )


# ============================================================
# MODEL
# ============================================================

class B2Model(
    nn.Module
):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()

        self.processor = (
            AutoFeatureExtractor
            .from_pretrained(
                ENCODER_MODEL,
                local_files_only=True
            )
        )

        self.encoder = (
            Wav2Vec2BertModel
            .from_pretrained(
                ENCODER_MODEL,
                dtype=torch.float32,
                local_files_only=True
            )
        )

        self.encoder.config.layerdrop = 0.0

        for p in self.encoder.parameters():

            p.requires_grad = False

        self.adapter = (
            ResidualBottleneckAdapter(
                FEATURE_DIM,
                BOTTLENECK
            )
        )

        self.ctc_head = nn.Linear(
            FEATURE_DIM,
            vocab_size
        )

        self.encoder.to(
            DEVICE
        )

        self.adapter.to(
            DEVICE
        )

        self.ctc_head.to(
            DEVICE
        )

        try:

            self.encoder.gradient_checkpointing_disable()

        except Exception:

            pass


    def forward(
        self,
        input_features
    ):

        hidden = (
            self.encoder(
                input_features=
                input_features
            ).last_hidden_state
        )

        return self.ctc_head(
            self.adapter(
                hidden
            )
        )


# ============================================================
# B3 CONFIGURATION
# ============================================================

def configure_b3(
    model
):

    # Freeze EVERYTHING first.

    for p in model.parameters():

        p.requires_grad = False


    # Only layers 22 and 23.

    for layer_idx in TRAINABLE_LAYERS:

        for p in (
            model
            .encoder
            .encoder
            .layers[
                layer_idx
            ]
            .parameters()
        ):

            p.requires_grad = True


    # Adapter.

    for p in model.adapter.parameters():

        p.requires_grad = True


    # CTC head.

    for p in model.ctc_head.parameters():

        p.requires_grad = True


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_resume_checkpoint(
    model
):

    print()
    print(
        "[RESUME] Loading:"
    )
    print(
        START_CHECKPOINT
    )

    ckpt = torch.load(
        START_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    checkpoint_epoch = int(
        ckpt.get(
            "epoch",
            -1
        )
    )

    if checkpoint_epoch != 8:

        raise RuntimeError(
            "Safety failure: "
            f"expected Epoch 8, "
            f"found Epoch "
            f"{checkpoint_epoch}"
        )

    state = None

    if (
        isinstance(
            ckpt,
            dict
        )
    ):

        if (
            "model_state_dict"
            in ckpt
        ):

            state = ckpt[
                "model_state_dict"
            ]

        elif (
            "state_dict"
            in ckpt
        ):

            state = ckpt[
                "state_dict"
            ]

        elif (
            "model"
            in ckpt
        ):

            state = ckpt[
                "model"
            ]

    if state is None:

        raise RuntimeError(
            "No model state dict found "
            "in Epoch 8 checkpoint."
        )

    model.load_state_dict(
        state,
        strict=True
    )

    print(
        "[RESUME] Epoch:",
        checkpoint_epoch
    )

    print(
        "[RESUME] Train loss:",
        ckpt.get(
            "train_loss"
        )
    )

    print(
        "[RESUME] Val loss:",
        ckpt.get(
            "val_loss"
        )
    )

    print(
        "[RESUME] Beam-500 CER:",
        f"{100 * ckpt.get('val_beam500_cer', 0):.4f}%"
    )

    print(
        "[RESUME] Beam-500 WER:",
        f"{100 * ckpt.get('val_beam500_wer', 0):.4f}%"
    )

    print(
        "[RESUME] Epoch 8 checkpoint: PASS"
    )

    return ckpt


# ============================================================
# OPTIMIZER
# ============================================================

def make_optimizer(
    model
):

    encoder_params = []

    for layer_idx in TRAINABLE_LAYERS:

        encoder_params.extend(
            list(
                model
                .encoder
                .encoder
                .layers[
                    layer_idx
                ]
                .parameters()
            )
        )

    head_params = (
        list(
            model.adapter.parameters()
        )
        +
        list(
            model.ctc_head.parameters()
        )
    )

    optimizer = torch.optim.AdamW(

        [
            {
                "params":
                    encoder_params,

                "lr":
                    LR_ENCODER
            },

            {
                "params":
                    head_params,

                "lr":
                    LR_HEAD
            }
        ],

        weight_decay=
            WEIGHT_DECAY,

        betas=
            (0.9, 0.98)
    )

    return optimizer


# ============================================================
# MODEL REPORT
# ============================================================

def print_report(
    model
):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "[MODEL] Total parameters:",
        f"{total:,}"
    )

    print(
        "[MODEL] Trainable:",
        f"{trainable:,}"
    )

    print(
        "[MODEL] Trainable ratio:",
        f"{100 * trainable / total:.4f}%"
    )

    print()

    for i in range(24):

        n = sum(
            p.numel()
            for p in (
                model
                .encoder
                .encoder
                .layers[i]
                .parameters()
            )
            if p.requires_grad
        )

        print(
            f"[LAYER {i:02d}] "
            f"{'TRAINABLE' if n else 'FROZEN':10s} "
            f"{n:,}"
        )

    print()

    print(
        "[ADAPTER] Trainable:",
        sum(
            p.numel()
            for p in model.adapter.parameters()
            if p.requires_grad
        )
    )

    print(
        "[CTC HEAD] Trainable:",
        sum(
            p.numel()
            for p in model.ctc_head.parameters()
            if p.requires_grad
        )
    )


# ============================================================
# FEATURES
# ============================================================

def make_features(
    processor,
    audios
):

    audio_list = [
        w.cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False
        )
        for w in audios
    ]

    output = processor(
        audio_list,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    features = (
        output.input_features
    )

    del audio_list
    del output

    return features


# ============================================================
# DECODING
# ============================================================

def decode_ids(
    ids,
    vocab
):

    inverse_vocab = {
        int(v): k
        for k, v in vocab.items()
    }

    blank = vocab[
        SPECIAL_BLANK
    ]

    previous = None

    output = []

    for token in ids:

        token = int(
            token
        )

        if token == blank:

            previous = None

            continue

        if token == previous:

            continue

        output.append(
            inverse_vocab.get(
                token,
                SPECIAL_UNK
            )
        )

        previous = token

    return "".join(
        output
    )


# ============================================================
# METRICS
# ============================================================

def levenshtein(
    a,
    b
):

    if len(a) < len(b):

        a, b = b, a

    previous = list(
        range(
            len(b) + 1
        )
    )

    for i, ca in enumerate(
        a,
        1
    ):

        current = [i]

        for j, cb in enumerate(
            b,
            1
        ):

            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1]
                    + (ca != cb)
                )
            )

        previous = current

    return previous[-1]


def metrics(
    refs,
    hyps
):

    char_errors = 0
    char_total = 0

    word_errors = 0
    word_total = 0

    for ref, hyp in zip(
        refs,
        hyps
    ):

        ref = normalize_text(
            ref
        )

        hyp = normalize_text(
            hyp
        )

        ref_chars = list(
            ref
        )

        hyp_chars = list(
            hyp
        )

        ref_words = ref.split()
        hyp_words = hyp.split()

        char_errors += levenshtein(
            ref_chars,
            hyp_chars
        )

        char_total += max(
            len(ref_chars),
            1
        )

        word_errors += levenshtein(
            ref_words,
            hyp_words
        )

        word_total += max(
            len(ref_words),
            1
        )

    return (
        char_errors /
        max(char_total, 1),

        word_errors /
        max(word_total, 1)
    )


# ============================================================
# DECODER
# ============================================================

def build_decoder(
    vocab
):

    labels = [
        key
        for key, value
        in sorted(
            vocab.items(),
            key=lambda x: x[1]
        )
    ]

    labels = [
        ""
        if x == SPECIAL_BLANK
        else x
        for x in labels
    ]

    with open(
        UNIGRAM_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        unigrams = [
            line.strip()
            for line in f
            if line.strip()
        ]

    print(
        "[DECODER] labels:",
        len(labels)
    )

    print(
        "[DECODER] unigrams:",
        f"{len(unigrams):,}"
    )

    print(
        "[DECODER] beam:",
        BEAM_WIDTH
    )

    return build_ctcdecoder(
        labels=labels,
        unigrams=unigrams
    )


# ============================================================
# TRAIN MODE
# ============================================================

def set_train_modes(
    model
):

    model.eval()

    for layer_idx in TRAINABLE_LAYERS:

        model.encoder.encoder.layers[
            layer_idx
        ].train()

    model.adapter.train()
    model.ctc_head.train()


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    ctc,
    epoch
):

    set_train_modes(
        model
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0

    samples_seen = 0

    optimizer_steps = 0

    start_time = time.time()

    for step, batch in enumerate(
        loader,
        1
    ):

        features = make_features(
            model.processor,
            batch["audio"]
        ).to(
            DEVICE
        )

        targets = batch[
            "targets"
        ].to(
            DEVICE
        )

        target_lengths = batch[
            "target_lengths"
        ].to(
            DEVICE
        )

        amp_enabled = (
            AMP
            and DEVICE.type == "cuda"
            and USE_BF16
        )

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=amp_enabled
        ):

            logits = model(
                features
            )

            input_lengths = torch.full(
                (
                    logits.shape[0],
                ),
                logits.shape[1],
                dtype=torch.long,
                device=DEVICE
            )

            log_probs = (
                logits
                .float()
                .log_softmax(
                    -1
                )
                .transpose(
                    0,
                    1
                )
            )

            loss = ctc(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )

        if not torch.isfinite(
            loss
        ):

            raise FloatingPointError(
                f"Non-finite loss at "
                f"step {step}: "
                f"{loss.item()}"
            )

        # ----------------------------------------------------
        # Gradient accumulation
        # ----------------------------------------------------

        accumulation_scale = (
            1.0 / GRAD_ACCUM
        )

        # Last partial accumulation:
        # 7768 / 16 leaves 8 samples.
        #
        # Correct the final partial group so its gradient
        # is not artificially divided by 16.
        is_last_step = (
            step == len(loader)
        )

        remainder = (
            len(loader) % GRAD_ACCUM
        )

        if (
            is_last_step
            and remainder != 0
        ):

            accumulation_scale = (
                1.0 / remainder
            )

        scaled_loss = (
            loss
            * accumulation_scale
        )

        scaled_loss.backward()

        running_loss += (
            loss.item()
            * len(batch["texts"])
        )

        samples_seen += len(
            batch["texts"]
        )

        should_step = (
            step % GRAD_ACCUM == 0
            or step == len(loader)
        )

        if should_step:

            grad_norm = (
                torch.nn.utils
                .clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM
                )
            )

            if not torch.isfinite(
                torch.as_tensor(
                    grad_norm
                )
            ):

                raise FloatingPointError(
                    "Non-finite gradient norm"
                )

            optimizer.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            optimizer_steps += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if step % 100 == 0:

            if DEVICE.type == "cuda":

                memory = (
                    torch.cuda
                    .memory_allocated(
                        DEVICE
                    )
                    / 1024**3
                )

            else:

                memory = 0.0

            print(
                f"[TRAIN] "
                f"epoch={epoch} "
                f"step={step}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"opt={optimizer_steps} "
                f"VRAM={memory:.2f}GB",
                flush=True
            )

        del features
        del logits
        del log_probs
        del loss

        if step % 100 == 0:

            gc.collect()

            if DEVICE.type == "cuda":

                torch.cuda.empty_cache()

    elapsed = (
        time.time()
        - start_time
    ) / 60.0

    average_loss = (
        running_loss
        / max(
            samples_seen,
            1
        )
    )

    return (
        average_loss,
        elapsed,
        optimizer_steps
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    vocab,
    decoder
):

    model.eval()

    refs = []
    greedy_predictions = []
    beam_predictions = []

    total_loss = 0.0
    items = 0

    ctc = nn.CTCLoss(
        blank=vocab[
            SPECIAL_BLANK
        ],
        zero_infinity=True
    )

    for batch_index, batch in enumerate(
        loader,
        1
    ):

        features = make_features(
            model.processor,
            batch["audio"]
        ).to(
            DEVICE
        )

        targets = batch[
            "targets"
        ].to(
            DEVICE
        )

        target_lengths = batch[
            "target_lengths"
        ].to(
            DEVICE
        )

        amp_enabled = (
            AMP
            and DEVICE.type == "cuda"
            and USE_BF16
        )

        with torch.autocast(
            device_type="cuda",
            dtype=AMP_DTYPE,
            enabled=amp_enabled
        ):

            logits = model(
                features
            )

            input_lengths = torch.full(
                (
                    logits.shape[0],
                ),
                logits.shape[1],
                dtype=torch.long,
                device=DEVICE
            )

            log_probs = (
                logits
                .float()
                .log_softmax(
                    -1
                )
                .transpose(
                    0,
                    1
                )
            )

            loss = ctc(
                log_probs,
                targets,
                input_lengths,
                target_lengths
            )

        if not torch.isfinite(
            loss
        ):

            raise FloatingPointError(
                f"Non-finite validation "
                f"loss: {loss.item()}"
            )

        total_loss += (
            loss.item()
            * len(batch["texts"])
        )

        items += len(
            batch["texts"]
        )

        greedy_ids = (
            logits
            .argmax(
                -1
            )
            .cpu()
            .tolist()
        )

        beam_log_probs = (
            log_probs
            .transpose(
                0,
                1
            )
            .cpu()
            .numpy()
        )

        for i, reference in enumerate(
            batch["texts"]
        ):

            refs.append(
                normalize_text(
                    reference
                )
            )

            greedy_predictions.append(
                normalize_text(
                    decode_ids(
                        greedy_ids[i],
                        vocab
                    )
                )
            )

            beam_predictions.append(
                normalize_text(
                    decoder.decode(
                        beam_log_probs[i],
                        beam_width=
                            BEAM_WIDTH
                    )
                )
            )

        if batch_index % 100 == 0:

            print(
                f"[VAL] "
                f"{batch_index}/{len(loader)}",
                flush=True
            )

        del features
        del logits
        del log_probs
        del loss

    gc.collect()

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Do NOT name this variable "gc".
    #
    # That was the bug that killed the Epoch 8 validation
    # after training had already completed.
    # --------------------------------------------------------

    greedy_metrics = metrics(
        refs,
        greedy_predictions
    )

    beam_metrics = metrics(
        refs,
        beam_predictions
    )

    return {
        "loss":
            total_loss
            / max(items, 1),

        "greedy_cer":
            greedy_metrics[0],

        "greedy_wer":
            greedy_metrics[1],

        "beam_cer":
            beam_metrics[0],

        "beam_wer":
            beam_metrics[1]
    }


# ============================================================
# SAFE CHECKPOINT
# ============================================================

def save_verified_checkpoint(
    model,
    epoch,
    metrics_dict,
    train_loss
):

    payload = {

        "epoch":
            int(epoch),

        "model_state_dict":
            model.state_dict(),

        "train_loss":
            float(train_loss),

        "val_loss":
            float(
                metrics_dict["loss"]
            ),

        "val_greedy_cer":
            float(
                metrics_dict[
                    "greedy_cer"
                ]
            ),

        "val_greedy_wer":
            float(
                metrics_dict[
                    "greedy_wer"
                ]
            ),

        "val_beam500_cer":
            float(
                metrics_dict[
                    "beam_cer"
                ]
            ),

        "val_beam500_wer":
            float(
                metrics_dict[
                    "beam_wer"
                ]
            ),

        "trainable_layers":
            TRAINABLE_LAYERS,

        "beam_width":
            BEAM_WIDTH,

        "config": {

            "batch_size":
                BATCH_SIZE,

            "grad_accum":
                GRAD_ACCUM,

            "lr_encoder":
                LR_ENCODER,

            "lr_head":
                LR_HEAD,

            "weight_decay":
                WEIGHT_DECAY,

            "max_grad_norm":
                MAX_GRAD_NORM,

            "bf16":
                USE_BF16
        }
    }

    epoch_path = (
        B3_DIR
        / f"epoch_{epoch:02d}.pt"
    )

    temporary_path = (
        B3_DIR
        / f"epoch_{epoch:02d}.pt.tmp"
    )

    # --------------------------------------------------------
    # Write NEW epoch first.
    # Do NOT touch Epoch 8 / previous epoch.
    # --------------------------------------------------------

    if temporary_path.exists():

        temporary_path.unlink()

    print(
        f"[CHECKPOINT] Writing "
        f"{epoch_path.name}..."
    )

    torch.save(
        payload,
        temporary_path
    )

    # --------------------------------------------------------
    # Verify the temporary checkpoint.
    # --------------------------------------------------------

    print(
        "[CHECKPOINT] Verifying..."
    )

    verified = torch.load(
        temporary_path,
        map_location="cpu",
        weights_only=True
    )

    if int(
        verified.get(
            "epoch",
            -1
        )
    ) != epoch:

        raise RuntimeError(
            "Checkpoint verification "
            "failed: wrong epoch."
        )

    if (
        "model_state_dict"
        not in verified
    ):

        raise RuntimeError(
            "Checkpoint verification "
            "failed: model state missing."
        )

    del verified

    # --------------------------------------------------------
    # Atomic replacement.
    # --------------------------------------------------------

    if epoch_path.exists():

        epoch_path.unlink()

    temporary_path.replace(
        epoch_path
    )

    print(
        "[CHECKPOINT] VERIFIED:",
        epoch_path
    )

    return epoch_path


# ============================================================
# SAFE HARDLINK ALIAS
# ============================================================

def update_alias(
    alias_path,
    target_path
):

    alias_path = Path(
        alias_path
    )

    target_path = Path(
        target_path
    )

    temporary_alias = (
        alias_path.with_suffix(
            alias_path.suffix
            + ".tmp"
        )
    )

    if temporary_alias.exists():

        temporary_alias.unlink()

    # --------------------------------------------------------
    # Prefer hardlink.
    # This avoids another 2.3 GB physical copy.
    # --------------------------------------------------------

    try:

        os.link(
            target_path,
            temporary_alias
        )

        temporary_alias.replace(
            alias_path
        )

        print(
            f"[ALIAS] {alias_path.name} "
            f"-> {target_path.name}"
        )

    except Exception as e:

        print(
            "[WARNING] Hardlink failed:",
            e
        )

        # Fallback: physical copy.

        import shutil

        shutil.copy2(
            target_path,
            temporary_alias
        )

        temporary_alias.replace(
            alias_path
        )

        print(
            f"[ALIAS] Copied "
            f"{alias_path.name}"
        )


# ============================================================
# HISTORY
# ============================================================

def load_history():

    history_path = (
        B3_DIR
        / "training_history.json"
    )

    if not history_path.exists():

        return []

    try:

        history = json.loads(
            history_path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(
            history,
            list
        ):

            return []

        return history

    except Exception as e:

        print(
            "[WARNING] Could not read "
            "history:",
            e
        )

        return []


def save_history(
    history
):

    history_path = (
        B3_DIR
        / "training_history.json"
    )

    temporary = (
        B3_DIR
        / "training_history.json.tmp"
    )

    temporary.write_text(
        json.dumps(
            history,
            indent=2
        ),
        encoding="utf-8"
    )

    temporary.replace(
        history_path
    )


def update_history(
    history,
    epoch,
    train_loss,
    metrics_dict
):

    record = {

        "epoch":
            int(epoch),

        "train_loss":
            float(train_loss),

        "loss":
            float(
                metrics_dict["loss"]
            ),

        "greedy_cer":
            float(
                metrics_dict[
                    "greedy_cer"
                ]
            ),

        "greedy_wer":
            float(
                metrics_dict[
                    "greedy_wer"
                ]
            ),

        "beam_cer":
            float(
                metrics_dict[
                    "beam_cer"
                ]
            ),

        "beam_wer":
            float(
                metrics_dict[
                    "beam_wer"
                ]
            )
    }

    history = [
        item
        for item in history
        if int(
            item.get(
                "epoch",
                -1
            )
        ) != epoch
    ]

    history.append(
        record
    )

    history.sort(
        key=lambda x:
            int(
                x["epoch"]
            )
    )

    save_history(
        history
    )

    return history


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything()

    print()
    print("=" * 72)
    print(
        "MANIPURI ASR V2 — "
        "B3-NARROW RESUME"
    )
    print("=" * 72)

    print(
        "[DEVICE]",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "[GPU]",
            torch.cuda.get_device_name(
                0
            )
        )

        print(
            "[VRAM]",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print(
        "[AMP] BF16:",
        USE_BF16
    )

    print(
        "[AMP] dtype:",
        AMP_DTYPE
    )

    print()
    print(
        "[RESUME] Epoch 8 → "
        f"{START_EPOCH} → "
        f"{END_EPOCH}"
    )

    # ========================================================
    # PATH CHECK
    # ========================================================

    required_paths = [

        TRAIN_CSV,

        VAL_CSV,

        ENCODER_MODEL,

        B2_CHECKPOINT,

        START_CHECKPOINT,

        VOCAB_PATH,

        UNIGRAM_PATH
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Required path missing:\n"
                f"{path}"
            )

        print(
            "[PATH] PASS:",
            path
        )


    # ========================================================
    # DATA
    # ========================================================

    train_df = pd.read_csv(
        TRAIN_CSV,
        encoding="utf-8-sig"
    )

    val_df = pd.read_csv(
        VAL_CSV,
        encoding="utf-8-sig"
    )

    if len(train_df) != 7768:

        raise RuntimeError(
            f"Expected 7768 train "
            f"samples, got "
            f"{len(train_df)}"
        )

    if len(val_df) != 864:

        raise RuntimeError(
            f"Expected 864 val "
            f"samples, got "
            f"{len(val_df)}"
        )

    vocab = load_vocab()

    print()
    print(
        "[DATA]",
        f"train={len(train_df):,}",
        f"val={len(val_df):,}",
        f"vocab={len(vocab)}"
    )


    # ========================================================
    # DATALOADERS
    # ========================================================

    train_loader = DataLoader(

        AudioCTCDataset(
            train_df,
            vocab
        ),

        batch_size=
            BATCH_SIZE,

        shuffle=True,

        num_workers=
            NUM_WORKERS,

        collate_fn=
            collate_fn,

        pin_memory=
            DEVICE.type == "cuda"
    )


    val_loader = DataLoader(

        AudioCTCDataset(
            val_df,
            vocab
        ),

        batch_size=
            BATCH_SIZE,

        shuffle=False,

        num_workers=
            NUM_WORKERS,

        collate_fn=
            collate_fn,

        pin_memory=
            DEVICE.type == "cuda"
    )


    # ========================================================
    # MODEL
    # ========================================================

    print()
    print(
        "[MODEL] Building B3 model..."
    )

    model = B2Model(
        len(vocab)
    )


    # ========================================================
    # IMPORTANT:
    #
    # First load the B2 architecture, then apply B3 freezing,
    # then overwrite it with the actual Epoch 8 B3 checkpoint.
    # ========================================================

    print(
        "[MODEL] Loading B2 base..."
    )

    b2_ckpt = torch.load(
        B2_CHECKPOINT,
        map_location="cpu",
        weights_only=True
    )

    if (
        "state_dict"
        in b2_ckpt
    ):

        b2_state = b2_ckpt[
            "state_dict"
        ]

    elif (
        "model_state_dict"
        in b2_ckpt
    ):

        b2_state = b2_ckpt[
            "model_state_dict"
        ]

    elif (
        "model"
        in b2_ckpt
    ):

        b2_state = b2_ckpt[
            "model"
        ]

    else:

        b2_state = b2_ckpt

    model.load_state_dict(
        b2_state,
        strict=False
    )

    del b2_ckpt
    del b2_state


    # ========================================================
    # B3 FREEZING
    # ========================================================

    configure_b3(
        model
    )

    print_report(
        model
    )


    # ========================================================
    # LOAD EPOCH 8
    # ========================================================

    epoch8_ckpt = (
        load_resume_checkpoint(
            model
        )
    )


    # ========================================================
    # OPTIMIZER
    #
    # Epoch 8 checkpoint intentionally contains the model
    # weights required for continuation. We create a fresh
    # optimizer for the continuation.
    # ========================================================

    optimizer = make_optimizer(
        model
    )

    ctc = nn.CTCLoss(
        blank=vocab[
            SPECIAL_BLANK
        ],
        zero_infinity=True
    )

    decoder = build_decoder(
        vocab
    )


    # ========================================================
    # HISTORY
    # ========================================================

    history = load_history()

    # Ensure Epoch 8 is present and correct.

    epoch8_record = {

        "epoch":
            8,

        "train_loss":
            float(
                epoch8_ckpt[
                    "train_loss"
                ]
            ),

        "loss":
            float(
                epoch8_ckpt[
                    "val_loss"
                ]
            ),

        "greedy_cer":
            float(
                epoch8_ckpt[
                    "val_greedy_cer"
                ]
            ),

        "greedy_wer":
            float(
                epoch8_ckpt[
                    "val_greedy_wer"
                ]
            ),

        "beam_cer":
            float(
                epoch8_ckpt[
                    "val_beam500_cer"
                ]
            ),

        "beam_wer":
            float(
                epoch8_ckpt[
                    "val_beam500_wer"
                ]
            )
    }

    history = [
        item
        for item in history
        if int(
            item.get(
                "epoch",
                -1
            )
        ) != 8
    ]

    history.append(
        epoch8_record
    )

    history.sort(
        key=lambda x:
            int(x["epoch"])
    )

    save_history(
        history
    )

    print()
    print(
        "[HISTORY] Epochs:",
        [
            x["epoch"]
            for x in history
        ]
    )


    # ========================================================
    # BEST VALIDATION CER
    #
    # Epoch 8 beam CER = 16.2847%
    # ========================================================

    best_beam_cer = min(
        float(
            item["beam_cer"]
        )
        for item in history
        if "beam_cer" in item
    )

    print()
    print(
        "[BEST] Existing best "
        "Beam-500 CER:",
        f"{best_beam_cer * 100:.4f}%"
    )


    # ========================================================
    # EPOCH 9 → 12
    # ========================================================

    for epoch in range(
        START_EPOCH,
        END_EPOCH + 1
    ):

        print()
        print("=" * 72)
        print(
            f"B3-NARROW EPOCH "
            f"{epoch}/{END_EPOCH}"
        )
        print("=" * 72)


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_loss, minutes, steps = (
            train_epoch(
                model,
                train_loader,
                optimizer,
                ctc,
                epoch
            )
        )

        print()
        print(
            f"[TRAIN] loss="
            f"{train_loss:.6f}"
        )

        print(
            f"[TRAIN] time="
            f"{minutes:.2f} min"
        )

        print(
            f"[TRAIN] optimizer_steps="
            f"{steps}"
        )


        gc.collect()

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print()
        print(
            f"[VAL] Epoch {epoch}"
        )

        metrics_dict = validate(
            model,
            val_loader,
            vocab,
            decoder
        )


        # ----------------------------------------------------
        # RESULTS
        # ----------------------------------------------------

        print()
        print("=" * 72)

        print(
            f"EPOCH {epoch} RESULTS"
        )

        print("=" * 72)

        print(
            f"Train loss       : "
            f"{train_loss:.6f}"
        )

        print(
            f"Val loss         : "
            f"{metrics_dict['loss']:.6f}"
        )

        print(
            f"Greedy CER       : "
            f"{metrics_dict['greedy_cer'] * 100:.4f}%"
        )

        print(
            f"Greedy WER       : "
            f"{metrics_dict['greedy_wer'] * 100:.4f}%"
        )

        print(
            f"Beam-500 CER     : "
            f"{metrics_dict['beam_cer'] * 100:.4f}%"
        )

        print(
            f"Beam-500 WER     : "
            f"{metrics_dict['beam_wer'] * 100:.4f}%"
        )

        print(
            "B2 Beam-500 CER  : 20.1736%"
        )

        print(
            "B2 Beam-500 WER  : 71.7230%"
        )


        # ----------------------------------------------------
        # SAVE NEW PHYSICAL CHECKPOINT FIRST
        # ----------------------------------------------------

        epoch_path = (
            save_verified_checkpoint(
                model,
                epoch,
                metrics_dict,
                train_loss
            )
        )


        # ----------------------------------------------------
        # ONLY NOW UPDATE LAST
        # ----------------------------------------------------

        update_alias(
            B3_DIR / "last.pt",
            epoch_path
        )


        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if (
            metrics_dict["beam_cer"]
            < best_beam_cer
        ):

            best_beam_cer = (
                metrics_dict[
                    "beam_cer"
                ]
            )

            update_alias(
                B3_DIR / "best.pt",
                epoch_path
            )

            print()
            print(
                "🔥 NEW BEST!"
            )

            print(
                "Beam-500 CER:",
                f"{best_beam_cer * 100:.4f}%"
            )

        else:

            print()
            print(
                "[BEST] No improvement."
            )

            print(
                "Current best:",
                f"{best_beam_cer * 100:.4f}%"
            )


        # ----------------------------------------------------
        # HISTORY
        # ----------------------------------------------------

        history = update_history(
            history,
            epoch,
            train_loss,
            metrics_dict
        )

        print(
            "[HISTORY] Epochs:",
            [
                x["epoch"]
                for x in history
            ]
        )


        # ----------------------------------------------------
        # CLEANUP
        # ----------------------------------------------------

        gc.collect()

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()

            allocated = (
                torch.cuda
                .memory_allocated()
                / 1024**3
            )

            reserved = (
                torch.cuda
                .memory_reserved()
                / 1024**3
            )

            print(
                f"[GPU] allocated="
                f"{allocated:.2f} GB "
                f"reserved="
                f"{reserved:.2f} GB"
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 72)
    print(
        "B3-NARROW EPOCH 9 → 12 COMPLETE"
    )
    print("=" * 72)

    print(
        "Best Beam-500 CER:",
        f"{best_beam_cer * 100:.4f}%"
    )

    print(
        "Best checkpoint:",
        B3_DIR / "best.pt"
    )

    print(
        "Last checkpoint:",
        B3_DIR / "last.pt"
    )

    print()
    print(
        "Official 410-sample test:"
    )

    print(
        "NOT TOUCHED"
    )

    print()
    print(
        "Training continuation finished safely."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()