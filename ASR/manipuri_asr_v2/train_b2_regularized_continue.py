from pathlib import Path
import csv
import json
import random
import time
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_b2_partial_encoder import (
    B2Model,
    DEVICE,
    SPECIAL_BLANK,
    normalize_text,
    collate_fn,
)

# ============================================================
# B2 REGULARIZED CONTINUATION
# Continues from Regularized Epoch 1.
# One epoch per process/run.
# ============================================================

LR_ENCODER = 5e-6
LR_HEAD = 2e-4
WEIGHT_DECAY = 0.02
MAX_GRAD_NORM = 1.0
AMP = True

GRAD_ACCUM = 16
BATCH_SIZE = 1
NUM_WORKERS = 0

TRAINABLE_LAYERS = [20, 21, 22, 23]

# Patience is evaluated externally after validation.
# This script performs exactly ONE continuation epoch.
# Therefore it is safe to validate after every run.

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent

RUN_DIR = (
    ASR_DIR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
)

REG_DIR = RUN_DIR / "regularized"

AUDIO_CACHE = ASR_DIR / "audio_cache"
TRAIN_CSV = ASR_DIR / "combined_dataset" / "train.csv"
VOCAB_PATH = RUN_DIR / "vocab.json"

TRAIN_LOG = REG_DIR / "continuation_training_log.csv"
SUMMARY_LOG = REG_DIR / "continuation_summary.csv"

TRAIN_HEADER = [
    "epoch",
    "global_sample",
    "batch",
    "optimizer_step",
    "loss",
    "lr_encoder",
    "lr_head",
    "vram_gb",
    "audio_frames",
    "target_length",
    "elapsed_seconds",
    "status",
    "text",
]

SUMMARY_HEADER = [
    "epoch",
    "start_sample",
    "end_sample",
    "samples_processed",
    "optimizer_steps",
    "avg_loss",
    "elapsed_seconds",
    "status",
]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv(path, header, row):
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow(row)
        f.flush()


class CachedAudioCTCDataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cache_path = AUDIO_CACHE / f"{idx:06d}.pt"

        if not cache_path.exists():
            raise FileNotFoundError(f"Audio cache missing: {cache_path}")

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        audio = payload["audio"]

        if not isinstance(audio, torch.Tensor):
            raise RuntimeError(f"Invalid cached audio: {cache_path}")

        audio = audio.float().contiguous()

        if audio.numel() == 0:
            raise RuntimeError(f"Empty cached audio: {cache_path}")

        text = normalize_text(self.df.iloc[idx]["text"])

        target = torch.tensor(
            [
                self.vocab.get(ch, self.vocab["<unk>"])
                for ch in text
            ],
            dtype=torch.long,
        )

        return {
            "audio": audio,
            "target": target,
            "text": text,
            "global_idx": idx,
        }


def create_model(vocab):
    print("[MODEL] Creating B2 model", flush=True)
    model = B2Model(vocab_size=len(vocab))
    model.to(DEVICE)
    return model

def create_model_from_regularized(vocab):
    print("[MODEL] Creating B2 model", flush=True)

    model = B2Model(vocab_size=len(vocab))
    model.to(DEVICE)

    model_path = REG_DIR / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Regularized checkpoint missing:\n{model_path}"
        )

    print(f"[RESUME] Loading Regularized checkpoint: {model_path}", flush=True)

    state = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state, strict=True)

    print("[RESUME] Regularized model weights: PASS", flush=True)

    return model


def create_optimizer(model):
    encoder_params = []

    for layer_idx in TRAINABLE_LAYERS:
        encoder_params.extend(
            list(model.encoder.encoder.layers[layer_idx].parameters())
        )

    head_params = (
        list(model.adapter.parameters())
        + list(model.ctc_head.parameters())
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

    return optimizer


def load_regularized_state(model, optimizer, scaler):
    model_path = REG_DIR / "model.pt"
    optimizer_path = REG_DIR / "optimizer.pt"
    scaler_path = REG_DIR / "scaler.pt"
    state_path = REG_DIR / "state.json"

    for path in [model_path, optimizer_path, state_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required checkpoint missing:\n{path}")

    print("=" * 72)
    print("[RESUME] Loading B2 Regularized Epoch 1", flush=True)
    print(f"[RESUME] {REG_DIR}", flush=True)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        ),
        strict=True,
    )

    optimizer.load_state_dict(
        torch.load(
            optimizer_path,
            map_location="cpu",
            weights_only=True,
        )
    )

    if scaler is not None and scaler_path.exists():
        scaler.load_state_dict(
            torch.load(
                scaler_path,
                map_location="cpu",
                weights_only=True,
            )
        )
        print("[RESUME] AMP scaler: PASS", flush=True)

    state = json.loads(state_path.read_text(encoding="utf-8"))

    previous_epoch = int(state["epoch"])
    optimizer_steps = int(state["optimizer_steps"])

    print(f"[RESUME] Previous epoch: {previous_epoch}", flush=True)
    print(f"[RESUME] Previous optimizer steps: {optimizer_steps}", flush=True)
    print(f"[REGULARIZED] Encoder LR: {LR_ENCODER}", flush=True)
    print(f"[REGULARIZED] Head LR: {LR_HEAD}", flush=True)
    print(f"[REGULARIZED] Weight decay: {WEIGHT_DECAY}", flush=True)
    print("=" * 72)

    return previous_epoch, optimizer_steps


def train_one_epoch(
    model,
    loader,
    optimizer,
    ctc_loss_fn,
    optimizer_steps,
    epoch,
):
    model.train()

    # Frozen feature projection and early encoder layers remain eval.
    model.encoder.feature_projection.eval()

    for idx in range(20):
        model.encoder.encoder.layers[idx].eval()

    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_items = 0
    start_time = time.time()

    for batch_number, batch in enumerate(loader, start=1):
        global_idx = batch_number - 1
        text = batch["texts"][0]
        sample_start = time.time()

        try:
            audio = batch["audio"].float().contiguous()

            audio_np = (
                audio[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )

            processor_output = None
            input_features = None

            try:
                processor_output = model.processor(
                    audio_np,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )

                input_features = processor_output.input_features.to(
                    DEVICE,
                    non_blocking=True,
                )
            finally:
                del processor_output
                del audio_np

            targets = batch["targets"].to(DEVICE)
            target_lengths = batch["target_lengths"].to(DEVICE)

            use_amp = AMP and DEVICE.type == "cuda"

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(input_features)

                input_lengths = torch.full(
                    (logits.shape[0],),
                    logits.shape[1],
                    dtype=torch.long,
                    device=DEVICE,
                )

                log_probs = (
                    logits.float()
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

            scaled_loss.backward()

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            should_step = batch_number % GRAD_ACCUM == 0

            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

            loss_value = float(loss.item())

            total_loss += loss_value
            total_items += 1

            vram = 0.0
            if DEVICE.type == "cuda":
                vram = (
                    torch.cuda.memory_allocated(DEVICE)
                    / 1024**3
                )

            elapsed = time.time() - sample_start

            append_csv(
                TRAIN_LOG,
                TRAIN_HEADER,
                [
                    epoch,
                    global_idx,
                    batch_number,
                    optimizer_steps,
                    f"{loss_value:.8f}",
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                    f"{vram:.4f}",
                    int(audio.shape[1]),
                    int(batch["target_lengths"][0]),
                    f"{elapsed:.4f}",
                    "ok",
                    text,
                ],
            )

            if batch_number % 100 == 0:
                gc.collect()

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

                print(
                    f"[REGULARIZED E{epoch}] "
                    f"{batch_number}/{len(loader)} "
                    f"global={global_idx} "
                    f"loss={loss_value:.4f} "
                    f"opt={optimizer_steps} "
                    f"VRAM={vram:.2f}GB",
                    flush=True,
                )

        except Exception as exc:
            append_csv(
                TRAIN_LOG,
                TRAIN_HEADER,
                [
                    epoch,
                    global_idx,
                    batch_number,
                    optimizer_steps,
                    "",
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                    "",
                    "",
                    "",
                    "",
                    "ERROR",
                    text,
                ],
            )

            print("=" * 72)
            print("[ERROR]")
            print(f"epoch={epoch}")
            print(f"global_sample={global_idx}")
            print(f"batch={batch_number}")
            print(f"text={text}")
            print(f"{type(exc).__name__}: {exc}")
            print("=" * 72, flush=True)

            raise

    # Finish final partial accumulation.
    if len(loader) % GRAD_ACCUM != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_steps += 1

    avg_loss = total_loss / max(total_items, 1)
    elapsed = time.time() - start_time

    return avg_loss, optimizer_steps, elapsed


def save_checkpoint(
    model,
    optimizer,
    scaler,
    epoch,
    optimizer_steps,
    avg_loss,
    elapsed,
):
    torch.save(
        model.state_dict(),
        REG_DIR / "model.pt",
    )

    torch.save(
        optimizer.state_dict(),
        REG_DIR / "optimizer.pt",
    )

    if scaler is not None:
        torch.save(
            scaler.state_dict(),
            REG_DIR / "scaler.pt",
        )

    old_state = {}
    state_path = REG_DIR / "state.json"

    if state_path.exists():
        try:
            old_state = json.loads(
                state_path.read_text(encoding="utf-8")
            )
        except Exception:
            old_state = {}

    state = {
        "epoch": epoch,
        "start_sample": 0,
        "end_sample": 7767,
        "samples_processed": 7768,
        "optimizer_steps": optimizer_steps,
        "avg_loss": avg_loss,
        "elapsed_seconds": elapsed,
        "status": "complete",
        "source_checkpoint": str(REG_DIR),
        "previous_epoch": old_state.get("epoch"),
    }

    state_path.write_text(
        json.dumps(state, indent=2),
        encoding="utf-8",
    )

    append_csv(
        SUMMARY_LOG,
        SUMMARY_HEADER,
        [
            epoch,
            0,
            7767,
            7768,
            optimizer_steps,
            f"{avg_loss:.8f}",
            f"{elapsed:.3f}",
            "complete",
        ],
    )


def main():
    seed_everything()

    REG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("MANIPURI ASR V2 — B2 REGULARIZED CONTINUATION")
    print("=" * 72)

    print("[DEVICE]", DEVICE, flush=True)

    if DEVICE.type == "cuda":
        print("[GPU]", torch.cuda.get_device_name(0), flush=True)

    cache_count = len(list(AUDIO_CACHE.glob("*.pt")))
    print("[CACHE]", cache_count, "files", flush=True)

    if cache_count != 7768:
        raise RuntimeError(
            f"Expected exactly 7768 training cache files, found {cache_count}"
        )

    train_df = pd.read_csv(
        TRAIN_CSV,
        encoding="utf-8-sig",
    )

    if len(train_df) != 7768:
        raise RuntimeError(
            f"Expected 7768 training rows, found {len(train_df)}"
        )

    vocab = json.loads(
        VOCAB_PATH.read_text(encoding="utf-8")
    )

    print("[VOCAB]", len(vocab), "tokens", flush=True)

    dataset = CachedAudioCTCDataset(
        train_df,
        vocab,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )

    print("[DATA] Samples:", len(dataset), flush=True)

    # Construct model, then load the actual regularized checkpoint.
    model = B2Model(vocab_size=len(vocab))
    model.to(DEVICE)

    optimizer = create_optimizer(model)

    scaler = None
    if DEVICE.type == "cuda":
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

    previous_epoch, optimizer_steps = load_regularized_state(
        model,
        optimizer,
        scaler,
    )

    epoch = previous_epoch + 1

    print(f"[TRAIN] Starting Regularized Epoch {epoch}", flush=True)

    ctc_loss_fn = nn.CTCLoss(
        blank=vocab[SPECIAL_BLANK],
        zero_infinity=True,
    )

    avg_loss, optimizer_steps, elapsed = train_one_epoch(
        model,
        loader,
        optimizer,
        ctc_loss_fn,
        optimizer_steps,
        epoch,
    )

    save_checkpoint(
        model,
        optimizer,
        scaler,
        epoch,
        optimizer_steps,
        avg_loss,
        elapsed,
    )

    print()
    print("=" * 72)
    print(f"B2 REGULARIZED EPOCH {epoch} COMPLETE")
    print(f"Average loss: {avg_loss:.6f}")
    print(f"Optimizer steps: {optimizer_steps}")
    print(f"Checkpoint: {REG_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
