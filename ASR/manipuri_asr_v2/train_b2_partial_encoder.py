from pathlib import Path
import os

import sys
import math
import random
import json
import time
import io
import wave
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf

from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2BertModel,
)
# ============================================================
# AUDIO
# ============================================================



def decode_pcm(raw, sample_width):
    if sample_width == 1:
        audio = np.frombuffer(
            raw,
            dtype=np.uint8
        ).astype(np.float32)

        # Convert unsigned 8-bit PCM to signed float
        audio = (audio - 128.0) / 128.0

    elif sample_width == 2:
        audio = np.frombuffer(
            raw,
            dtype=np.int16
        ).astype(np.float32) / 32768.0

    elif sample_width == 3:
        raw_bytes = np.frombuffer(
            raw,
            dtype=np.uint8
        )

        if len(raw_bytes) % 3 != 0:
            raise ValueError(
                "Invalid 24-bit PCM byte length"
            )

        raw_bytes = raw_bytes.reshape(-1, 3)

        audio = (
            raw_bytes[:, 0].astype(np.int32)
            | (raw_bytes[:, 1].astype(np.int32) << 8)
            | (raw_bytes[:, 2].astype(np.int32) << 16)
        )

        # Sign extension for 24-bit PCM
        mask = audio & 0x800000
        audio = audio - (mask << 1)

        audio = audio.astype(np.float32) / 8388608.0

    elif sample_width == 4:
        audio = np.frombuffer(
            raw,
            dtype=np.int32
        ).astype(np.float32) / 2147483648.0

    else:
        raise ValueError(
            f"Unsupported PCM sample width: {sample_width}"
        )

    return audio

def load_parquet_audio(parquet_file, row_index):
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(
            f"Parquet does not exist: {parquet_file}"
        )

    df = pd.read_parquet(
        parquet_file,
        columns=["audio"],
    )

    audio_obj = df.iloc[int(row_index)]["audio"]

    if not isinstance(audio_obj, dict):
        raise ValueError(
            f"Unexpected audio object type: {type(audio_obj)}"
        )

    audio_bytes = audio_obj.get("bytes")

    if audio_bytes is None:
        raise ValueError(
            "IndicVoices audio object has no bytes"
        )

    with wave.open(
        io.BytesIO(audio_bytes),
        "rb",
    ) as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    audio = decode_pcm(
        raw,
        sample_width,
    )

    if channels > 1:
        audio = audio.reshape(
            -1,
            channels,
        )
        audio = audio.mean(axis=1)

    return audio, sample_rate

def normalize_audio(
    audio,
    sample_rate,
):

    audio = torch.as_tensor(
        audio,
        dtype=torch.float32,
    )

    if audio.ndim == 2:

        if audio.shape[0] <= 8:

            audio = audio.mean(
                dim=0
            )

        else:

            audio = audio.mean(
                dim=1
            )

    elif audio.ndim > 2:

        audio = audio.squeeze()

    if sample_rate != SAMPLE_RATE:

        audio = torchaudio.functional.resample(
            audio,
            sample_rate,
            SAMPLE_RATE,
        )

    return audio.contiguous()


def resolve_audio(row):

    source = str(
        row.get(
            "source",
            "",
        )
    ).strip().lower()

    # --------------------------------------------------------
    # IndicVoices
    # --------------------------------------------------------

    if source == "indicvoices":

        parquet_file = str(
            row["parquet_file"]
        )

        parquet_row = int(
            row["parquet_row"]
        )

        audio, sr = load_parquet_audio(
            parquet_file,
            parquet_row,
        )

        return normalize_audio(
            audio,
            sr,
        )

    # --------------------------------------------------------
    # 1K / benchmark / ordinary audio files
    # --------------------------------------------------------

    audio_path = row.get(
        "audio_path",
        "",
    )

    if pd.isna(audio_path):

        raise RuntimeError(
            f"audio_path is NaN for "
            f"source={source}"
        )

    audio_path = str(
        audio_path
    ).strip()

    if not audio_path:

        raise RuntimeError(
            f"audio_path is empty for "
            f"source={source}"
        )

    if not os.path.exists(
        audio_path
    ):

        raise FileNotFoundError(
            f"Audio file does not exist: "
            f"{audio_path}"
        )

    array, sr = sf.read(
    	audio_path,
    	dtype="float32",
    	always_2d=False,
    )

    wav = torch.from_numpy(
        np.asarray(array)
    )

    return normalize_audio(
        wav,
        sr,
    )

# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

ASR = ROOT / "ASR"

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"

ENCODER_MODEL = Path(r"E:\hf_models\w2v-bert-2.0")

B1_DIR = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b1_10epoch"
    / "best"
)

RUN_ROOT = (
    ASR
    / "checkpoints"
    / "v2_ctc"
)

RUN_NAME = "b2_partial_encoder"
RUN_DIR = RUN_ROOT / RUN_NAME

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAMPLE_RATE = 16000
FEATURE_DIM = 1024
BOTTLENECK = 256

SEED = 42

# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 1
GRAD_ACCUM = 16

LR_ENCODER = 5e-6

LR_HEAD = 2e-4

WEIGHT_DECAY = 0.02
MAX_GRAD_NORM = 1.0

EPOCHS = 1

NUM_WORKERS = 0

AMP = True

TRAINABLE_LAYERS = [20, 21, 22, 23]

# ============================================================
# SEED
# ============================================================

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# TEXT
# ============================================================

SPECIAL_BLANK = "<blank>"
SPECIAL_UNK = "<unk>"


def normalize_text(text):
    import unicodedata

    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()


def build_vocab(df):

    chars = set()

    for text in df["text"].astype(str):

        text = normalize_text(text)

        chars.update(text)

    chars = sorted(
        c for c in chars
        if c != "\x00"
    )

    vocab = {
        SPECIAL_BLANK: 0,
        SPECIAL_UNK: 1,
    }

    for ch in chars:

        if ch not in vocab:

            vocab[ch] = len(vocab)

    return vocab


def encode_text(text, vocab):

    text = normalize_text(text)

    return torch.tensor(
        [
            vocab.get(
                ch,
                vocab[SPECIAL_UNK],
            )
            for ch in text
        ],
        dtype=torch.long,
    )




# ============================================================
# DATASET
# ============================================================

class AudioCTCDataset(Dataset):

    def __init__(
        self,
        df,
        vocab,
    ):

        self.df = df.reset_index(drop=True)

        self.vocab = vocab

    def __len__(self):

        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        wav = resolve_audio(row)

        target = encode_text(
            row["text"],
            self.vocab,
        )

        return {
            "audio": wav,
            "target": target,
            "text": normalize_text(
                row["text"]
            ),
        }


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    max_audio = max(
        x["audio"].shape[0]
        for x in batch
    )

    max_target = max(
        x["target"].shape[0]
        for x in batch
    )

    audios = torch.zeros(
        len(batch),
        max_audio,
        dtype=torch.float32,
    )

    audio_lengths = torch.zeros(
        len(batch),
        dtype=torch.long,
    )

    targets = []

    target_lengths = []

    texts = []

    for i, item in enumerate(batch):

        n = item["audio"].shape[0]

        audios[i, :n] = item["audio"]

        audio_lengths[i] = n

        targets.append(
            item["target"]
        )

        target_lengths.append(
            item["target"].shape[0]
        )

        texts.append(
            item["text"]
        )

    targets = torch.cat(
        targets,
        dim=0,
    )

    target_lengths = torch.tensor(
        target_lengths,
        dtype=torch.long,
    )

    return {
        "audio": audios,
        "audio_lengths": audio_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": texts,
    }


# ============================================================
# ADAPTER
# ============================================================

class ResidualBottleneckAdapter(nn.Module):

    def __init__(
        self,
        dim=1024,
        bottleneck=256,
    ):

        super().__init__()

        self.norm_in = nn.LayerNorm(dim)

        self.down = nn.Linear(
            dim,
            bottleneck,
        )

        self.act = nn.GELU()

        self.up = nn.Linear(
            bottleneck,
            dim,
        )

        self.norm_out = nn.LayerNorm(dim)

        # Same identity-like initialization
        # as B1.

        nn.init.normal_(
            self.up.weight,
            mean=0.0,
            std=1e-3,
        )

        nn.init.zeros_(
            self.up.bias
        )

    def forward(self, x):

        z = self.norm_in(x)

        z = self.down(z)

        z = self.act(z)

        z = self.up(z)

        return self.norm_out(
            x + z
        )


# ============================================================
# MODEL
# ============================================================

class B2Model(nn.Module):

    def __init__(
        self,
        vocab_size,
    ):

        super().__init__()

        print(
            "[MODEL] Loading W2V-BERT 2.0..."
        )

        self.processor = (
            AutoFeatureExtractor.from_pretrained(
                ENCODER_MODEL,
                local_files_only=True,
            )
        )

        self.encoder = (
            Wav2Vec2BertModel.from_pretrained(
                ENCODER_MODEL,
                dtype=torch.float32,
                local_files_only=True,
            )
        )

        # Disable stochastic layer dropping
        # for controlled fine-tuning.

        self.encoder.config.layerdrop = 0.0

        # ----------------------------------------------------
        # Freeze everything
        # ----------------------------------------------------

        for p in self.encoder.parameters():

            p.requires_grad = False

        # ----------------------------------------------------
        # Unfreeze final four Transformer layers
        # ----------------------------------------------------

        for layer_idx in TRAINABLE_LAYERS:

            for p in self.encoder.encoder.layers[
                layer_idx
            ].parameters():

                p.requires_grad = True

        # ----------------------------------------------------
        # Adapter
        # ----------------------------------------------------

        self.adapter = (
            ResidualBottleneckAdapter(
                FEATURE_DIM,
                BOTTLENECK,
            )
        )

        # ----------------------------------------------------
        # CTC
        # ----------------------------------------------------

        self.ctc_head = nn.Linear(
            FEATURE_DIM,
            vocab_size,
        )

        # W2V-BERT stays FP32.
        self.encoder.to(DEVICE)

        # Adapter/head follow device.
        self.adapter.to(DEVICE)
        self.ctc_head.to(DEVICE)

        # ----------------------------------------------------
        # Gradient checkpointing
        # ----------------------------------------------------

        # Disabled for B2 Windows/CUDA stability.
        # Only layers 20-23 are trainable, so the additional
        # checkpointing complexity is not necessary for this run.

        try:
            self.encoder.gradient_checkpointing_disable()
        except Exception:
            pass

        print(
            "[MODEL] Gradient checkpointing: DISABLED"
        )

        # ----------------------------------------------------
        # Print trainable parameters
        # ----------------------------------------------------

        total = sum(
            p.numel()
            for p in self.parameters()
        )

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        encoder_trainable = sum(
            p.numel()
            for p in self.encoder.parameters()
            if p.requires_grad
        )

        adapter_trainable = sum(
            p.numel()
            for p in self.adapter.parameters()
        )

        head_trainable = sum(
            p.numel()
            for p in self.ctc_head.parameters()
        )

        print(
            "[MODEL] Total parameters:",
            f"{total:,}",
        )

        print(
            "[MODEL] Encoder trainable:",
            f"{encoder_trainable:,}",
        )

        print(
            "[MODEL] Adapter trainable:",
            f"{adapter_trainable:,}",
        )

        print(
            "[MODEL] CTC head trainable:",
            f"{head_trainable:,}",
        )

        print(
            "[MODEL] TOTAL TRAINABLE:",
            f"{trainable:,}",
        )

    def forward(
        self,
        input_features,
    ):

        outputs = self.encoder(
            input_features=input_features,
        )

        hidden = outputs.last_hidden_state

        hidden = self.adapter(
            hidden
        )

        logits = self.ctc_head(
            hidden
        )

        return logits


# ============================================================
# CTC DECODER
# ============================================================

def decode_ids(
    ids,
    vocab,
):

    inv = {
        int(v): k
        for k, v in vocab.items()
    }

    blank = vocab[
        SPECIAL_BLANK
    ]

    prev = None

    output = []

    for token in ids:

        token = int(token)

        if token == blank:

            prev = None

            continue

        if token == prev:

            continue

        output.append(
            inv.get(
                token,
                SPECIAL_UNK,
            )
        )

        prev = token

    return "".join(output)


def levenshtein(a, b):

    if len(a) < len(b):

        a, b = b, a

    previous = list(
        range(len(b) + 1)
    )

    for i, ca in enumerate(
        a,
        start=1,
    ):

        current = [
            i
        ]

        for j, cb in enumerate(
            b,
            start=1,
        ):

            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1]
                    + (ca != cb),
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# LOAD B1 WEIGHTS
# ============================================================

def load_b1_weights(
    model,
    vocab,
):

    b1_model_path = (
        B1_DIR / "model.pt"
    )

    if not b1_model_path.exists():

        raise FileNotFoundError(
            f"B1 checkpoint not found: "
            f"{b1_model_path}"
        )

    state = torch.load(
        b1_model_path,
        map_location="cpu",
        weights_only=True,
    )

    # B1 adapter
    adapter_state = {}

    for key, value in state.items():

        if key.startswith(
            "adapter."
        ):

            adapter_state[
                key[len("adapter."):]
            ] = value

    if not adapter_state:

        raise RuntimeError(
            "Could not find adapter weights "
            "in B1 checkpoint."
        )

    model.adapter.load_state_dict(
        adapter_state,
        strict=True,
    )

    # B1 CTC head
    head_state = {}

    for key, value in state.items():

        if key.startswith(
            "ctc_head."
        ):

            head_state[
                key[len("ctc_head."):]
            ] = value

    if not head_state:

        raise RuntimeError(
            "Could not find CTC head weights "
            "in B1 checkpoint."
        )

    model.ctc_head.load_state_dict(
        head_state,
        strict=True,
    )

    print(
        "[INIT] B1 adapter weights: PASS"
    )

    print(
        "[INIT] B1 CTC head weights: PASS"
    )


# ============================================================
# ENCODER INPUT
# ============================================================

def make_input_features(
    processor,
    audios,
):

    # processor expects raw waveform at 16 kHz.
    #
    # We intentionally return CPU tensors here
    # and move them to CUDA after feature extraction.

    processed = processor(
        audios.cpu().numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )

    return processed.input_features



# ============================================================
# B2 RECOVERY / DEBUG HELPERS
# ============================================================

CHECKPOINT_EVERY_OPT_STEPS = 500


def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")

    torch.save(obj, tmp)

    if path.exists():
        path.unlink()

    tmp.replace(path)


def save_recovery_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    batch_step,
    optimizer_step,
    running_loss,
    sample_info,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (
            scaler.state_dict()
            if scaler is not None
            else None
        ),
        "epoch": int(epoch),
        "batch_step": int(batch_step),
        "optimizer_step": int(optimizer_step),
        "running_loss": float(running_loss),
        "sample_info": sample_info,
        "timestamp": time.time(),
    }

    path = RUN_DIR / "last_safe.pt"

    atomic_torch_save(
        payload,
        path,
    )

    print(
        f"[RECOVERY] Saved optimizer_step={optimizer_step} "
        f"batch_step={batch_step} -> {path}",
        flush=True,
    )


def describe_batch(batch):
    try:
        return {
            "text": str(batch["texts"][0]),
        }
    except Exception:
        return {
            "text": "<unavailable>",
        }


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    processor,
    loader,
    optimizer,
    scaler,
    ctc_loss_fn,
    epoch,
    optimizer_step_start=0,
):
    model.train()

    # Frozen portion stays in eval mode.
    model.encoder.feature_projection.eval()

    for idx in range(20):
        model.encoder.encoder.layers[idx].eval()

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    items = 0
    optimizer_steps = optimizer_step_start

    start_time = time.time()

    for step, batch in enumerate(loader, start=1):

        sample_info = describe_batch(batch)

        try:

            if step % 100 == 1 or step % 100 == 0:
                print(
                    f"[DATA] epoch={epoch} "
                    f"batch={step}/{len(loader)} "
                    f"text={sample_info['text'][:80]}",
                    flush=True,
                )

            input_features = make_input_features(
                processor,
                batch["audio"],
            )

            input_features = input_features.to(
                DEVICE,
                non_blocking=True,
            )

            targets = batch["targets"].to(DEVICE)

            target_lengths = batch["target_lengths"].to(DEVICE)

            use_amp = (
                AMP
                and DEVICE.type == "cuda"
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = model(
                    input_features
                )

                # Force CUDA errors to surface here.
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                input_lengths = torch.full(
                    (logits.shape[0],),
                    logits.shape[1],
                    dtype=torch.long,
                    device=DEVICE,
                )

                log_probs = (
                    logits
                    .float()
                    .log_softmax(dim=-1)
                    .transpose(0, 1)
                )

                loss = ctc_loss_fn(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths,
                )

                scaled_loss = loss / GRAD_ACCUM

            if scaler is not None:
                scaler.scale(
                    scaled_loss
                ).backward()
            else:
                scaled_loss.backward()

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            should_step = (
                step % GRAD_ACCUM == 0
                or step == len(loader)
            )

            if should_step:

                if scaler is not None:
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                optimizer_steps += 1

                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()

                if (
                    optimizer_steps
                    % CHECKPOINT_EVERY_OPT_STEPS
                    == 0
                ):

                    save_recovery_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        batch_step=step,
                        optimizer_step=optimizer_steps,
                        running_loss=running_loss,
                        sample_info=sample_info,
                    )

            running_loss += (
                float(loss.item())
                * len(batch["texts"])
            )

            items += len(batch["texts"])

            if step % 100 == 0:

                mem = 0.0

                if DEVICE.type == "cuda":
                    mem = (
                        torch.cuda.memory_allocated(DEVICE)
                        / 1024**3
                    )

                print(
                    f"[TRAIN] epoch={epoch} "
                    f"step={step}/{len(loader)} "
                    f"loss={loss.item():.4f} "
                    f"opt_step={optimizer_steps} "
                    f"VRAM={mem:.2f}GB",
                    flush=True,
                )

        except KeyboardInterrupt:
            raise

        except Exception as exc:

            print(
                "\n" + "=" * 72,
                flush=True,
            )

            print(
                "[BATCH ERROR]",
                flush=True,
            )

            print(
                f"epoch={epoch}",
                flush=True,
            )

            print(
                f"batch_step={step}",
                flush=True,
            )

            print(
                f"optimizer_step={optimizer_steps}",
                flush=True,
            )

            print(
                f"sample={sample_info}",
                flush=True,
            )

            print(
                f"exception={type(exc).__name__}: {exc}",
                flush=True,
            )

            print(
                "=" * 72,
                flush=True,
            )

            # Save whatever state is still valid.
            try:
                save_recovery_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    batch_step=step,
                    optimizer_step=optimizer_steps,
                    running_loss=running_loss,
                    sample_info=sample_info,
                )
            except Exception as save_exc:
                print(
                    "[RECOVERY] Save failed:",
                    repr(save_exc),
                    flush=True,
                )

            raise

    return (
        running_loss / max(items, 1),
        time.time() - start_time,
        optimizer_steps,
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    processor,
    loader,
    vocab,
):

    model.eval()

    refs = []

    hyps = []

    total_loss = 0.0

    total_items = 0

    ctc_loss_fn = nn.CTCLoss(
        blank=vocab[
            SPECIAL_BLANK
        ],
        zero_infinity=True,
    )

    for batch in loader:

        input_features = (
            make_input_features(
                processor,
                batch["audio"],
            )
        )

        input_features = (
            input_features.to(
                DEVICE,
                non_blocking=True,
            )
        )

        targets = batch[
            "targets"
        ].to(DEVICE)

        target_lengths = batch[
            "target_lengths"
        ].to(DEVICE)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(
                AMP
                and DEVICE.type == "cuda"
            ),
        ):

            logits = model(
                input_features
            )

            input_lengths = torch.full(
                (
                    logits.shape[0],
                ),
                logits.shape[1],
                dtype=torch.long,
                device=DEVICE,
            )

            log_probs = (
                logits
                .float()
                .log_softmax(
                    dim=-1
                )
                .transpose(0, 1)
            )

            loss = ctc_loss_fn(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

        total_loss += (
            float(loss.item())
            * len(batch["texts"])
        )

        total_items += len(
            batch["texts"]
        )

        predictions = (
            logits.argmax(
                dim=-1
            )
            .detach()
            .cpu()
            .tolist()
        )

        for ref, pred in zip(
            batch["texts"],
            predictions,
        ):

            refs.append(
                normalize_text(ref)
            )

            hyps.append(
                normalize_text(
                    decode_ids(
                        pred,
                        vocab,
                    )
                )
            )

    char_errors = 0
    char_total = 0

    word_errors = 0
    word_total = 0

    for ref, hyp in zip(
        refs,
        hyps,
    ):

        rchars = list(ref)
        hchars = list(hyp)

        char_errors += (
            levenshtein(
                rchars,
                hchars,
            )
        )

        char_total += max(
            len(rchars),
            1,
        )

        rwords = ref.split()
        hwords = hyp.split()

        word_errors += (
            levenshtein(
                rwords,
                hwords,
            )
        )

        word_total += max(
            len(rwords),
            1,
        )

    cer = (
        char_errors
        / max(char_total, 1)
    )

    wer = (
        word_errors
        / max(word_total, 1)
    )

    val_loss = (
        total_loss
        / max(total_items, 1)
    )

    print(
        f"[VAL] loss={val_loss:.4f} "
        f"CER={cer:.4f} "
        f"WER={wer:.4f}"
    )

    for i in range(
        min(5, len(refs))
    ):

        print(
            f"  REF: {refs[i]}"
        )

        print(
            f"  HYP: {hyps[i]}"
        )

    return {
        "loss": val_loss,
        "cer": cer,
        "wer": wer,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything()

    print("=" * 72)

    print(
        "MANIPURI ASR V2 — B2 PARTIAL ENCODER"
    )

    print("=" * 72)

    print(
        "[DEVICE]",
        DEVICE,
    )

    if DEVICE.type == "cuda":

        print(
            "[GPU]",
            torch.cuda.get_device_name(0),
        )

        print(
            "[VRAM]",
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_df = pd.read_csv(
        TRAIN_CSV,
        encoding="utf-8-sig",
    )

    val_df = pd.read_csv(
        VAL_CSV,
        encoding="utf-8-sig",
    )

    print(
        "[DATA]",
        f"train={len(train_df):,}",
        f"val={len(val_df):,}",
    )

    vocab = build_vocab(
        train_df
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        RUN_DIR / "vocab.json"
    ).write_text(
        json.dumps(
            vocab,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_ds = AudioCTCDataset(
        train_df,
        vocab,
    )

    val_ds = AudioCTCDataset(
        val_df,
        vocab,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = B2Model(
        vocab_size=len(vocab)
    )

    load_b1_weights(
        model,
        vocab,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    encoder_params = []

    for layer_idx in TRAINABLE_LAYERS:

        encoder_params.extend(
            list(
                model.encoder.encoder.layers[
                    layer_idx
                ].parameters()
            )
        )

    head_params = list(
        model.adapter.parameters()
    ) + list(
        model.ctc_head.parameters()
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": LR_ENCODER,
            },
            {
                "params": head_params,
                "lr": LR_HEAD,
            },
        ],
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.98),
    )

    scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )
        if DEVICE.type == "cuda"
        else None
    )

    ctc_loss_fn = nn.CTCLoss(
        blank=vocab[
            SPECIAL_BLANK
        ],
        zero_infinity=True,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    best_cer = float("inf")

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        print()
        print(
            "=" * 72
        )

        print(
            f"EPOCH {epoch}/{EPOCHS}"
        )

        print(
            "=" * 72
        )

        train_loss, elapsed, optimizer_steps = (
            train_one_epoch(
                model,
                model.processor,
                train_loader,
                optimizer,
                scaler,
                ctc_loss_fn,
                epoch,
                optimizer_step_start=0,
            )
        )

        print(
            f"[TRAIN] loss={train_loss:.4f} "
            f"time={elapsed:.1f}s "
            f"optimizer_steps={optimizer_steps}",
            flush=True,
        )

        metrics = evaluate(
            model,
            model.processor,
            val_loader,
            vocab,
        )

        if metrics["cer"] < best_cer:

            best_cer = metrics[
                "cer"
            ]

            torch.save(
                model.state_dict(),
                RUN_DIR / "best.pt",
            )

            print(
                "[CHECKPOINT] New best:",
                f"CER={best_cer:.4f}",
            )

        torch.save(
            model.state_dict(),
            RUN_DIR / "last.pt",
        )

    print()
    print("=" * 72)

    print(
        "B2 TRAINING COMPLETE"
    )

    print(
        "BEST CER:",
        best_cer,
    )

    print(
        "BEST CHECKPOINT:",
        RUN_DIR / "best.pt",
    )

    print("=" * 72)


if __name__ == "__main__":

    main()
