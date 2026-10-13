from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_b2_partial_encoder import (
    B2Model,
    build_vocab,
    DEVICE,
    TRAIN_CSV,
    LR_ENCODER,
    LR_HEAD,
    WEIGHT_DECAY,
    MAX_GRAD_NORM,
    AMP,
    SPECIAL_BLANK,
    normalize_text,
    collate_fn,
)


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR_DIR = SCRIPT_DIR.parent

RUN_DIR = (
    ASR_DIR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
)

EPOCH2_DIR = RUN_DIR / "epoch_02"
EPOCH3_DIR = RUN_DIR / "epoch_03"

AUDIO_CACHE = ASR_DIR / "audio_cache"

B1_DIR = (
    ASR_DIR
    / "checkpoints"
    / "v2_ctc"
    / "b1_10epoch"
    / "best"
)

GRAD_ACCUM = 16
BATCH_SIZE = 1
NUM_WORKERS = 0

TRAINABLE_LAYERS = [20, 21, 22, 23]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# CSV
# ============================================================

TRAIN_LOG = EPOCH3_DIR / "training_log.csv"
SUMMARY_LOG = EPOCH3_DIR / "summary.csv"

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


def append_csv(path, header, row):
    exists = path.exists()

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.writer(f)

        if not exists:
            writer.writerow(header)

        writer.writerow(row)

        f.flush()


# ============================================================
# CACHED AUDIO DATASET
# ============================================================

class CachedAudioCTCDataset(Dataset):

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

        global_idx = idx

        cache_path = (
            AUDIO_CACHE
            / f"{global_idx:06d}.pt"
        )

        if not cache_path.exists():

            raise FileNotFoundError(
                f"Audio cache missing: "
                f"{cache_path}"
            )

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        audio = payload["audio"]

        if not isinstance(
            audio,
            torch.Tensor,
        ):

            raise RuntimeError(
                f"Invalid cached audio: "
                f"{cache_path}"
            )

        audio = audio.float().contiguous()

        if audio.numel() == 0:

            raise RuntimeError(
                f"Empty cached audio: "
                f"{cache_path}"
            )

        text = normalize_text(
            self.df.iloc[idx]["text"]
        )

        target = torch.tensor(
            [
                self.vocab.get(
                    ch,
                    self.vocab["<unk>"],
                )
                for ch in text
            ],
            dtype=torch.long,
        )

        return {
            "audio": audio,
            "target": target,
            "text": text,
            "global_idx": global_idx,
        }


# ============================================================
# MODEL
# ============================================================

def create_model(vocab):

    print(
        "[MODEL] Creating B2 model",
        flush=True,
    )

    model = B2Model(
        vocab_size=len(vocab)
    )

    model.to(DEVICE)

    # --------------------------------------------------------
    # Load B1 adapter + CTC head
    # --------------------------------------------------------

    b1_path = B1_DIR / "model.pt"

    if not b1_path.exists():

        raise FileNotFoundError(
            f"B1 checkpoint not found: "
            f"{b1_path}"
        )

    state = torch.load(
        b1_path,
        map_location="cpu",
        weights_only=True,
    )

    adapter_state = {
        k[len("adapter."):]: v
        for k, v in state.items()
        if k.startswith("adapter.")
    }

    head_state = {
        k[len("ctc_head."):]: v
        for k, v in state.items()
        if k.startswith("ctc_head.")
    }

    model.adapter.load_state_dict(
        adapter_state,
        strict=True,
    )

    model.ctc_head.load_state_dict(
        head_state,
        strict=True,
    )

    print(
        "[INIT] B1 adapter: PASS",
        flush=True,
    )

    print(
        "[INIT] B1 CTC head: PASS",
        flush=True,
    )

    return model


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(model):

    encoder_params = []

    for layer_idx in TRAINABLE_LAYERS:

        encoder_params.extend(
            list(
                model.encoder.encoder.layers[
                    layer_idx
                ].parameters()
            )
        )

    head_params = (
        list(model.adapter.parameters())
        +
        list(model.ctc_head.parameters())
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


# ============================================================
# LOAD EPOCH 2 FINAL CHECKPOINT
# ============================================================

# LOAD EPOCH 2 FINAL CHECKPOINT

def load_epoch2_checkpoint(
    model,
    optimizer,
    scaler,
):

    model_path = EPOCH2_DIR / "model.pt"
    optimizer_path = EPOCH2_DIR / "optimizer.pt"
    scaler_path = EPOCH2_DIR / "scaler.pt"
    state_path = EPOCH2_DIR / "state.json"

    required = [
        model_path,
        optimizer_path,
        state_path,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required Epoch 2 checkpoint missing:\n{path}"
            )

    print("=" * 72)

    print(
        "[RESUME] Loading B2 Epoch 2 final checkpoint",
        flush=True,
    )

    print(
        f"[RESUME] {EPOCH2_DIR}",
        flush=True,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            EPOCH2_DIR / "model.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer.load_state_dict(
        torch.load(
            optimizer_path,
            map_location=DEVICE,
            weights_only=True,
        )
    )

    print(
        "[RESUME] Optimizer: PASS",
        flush=True,
    )

    # --------------------------------------------------------
    # AMP scaler
    # --------------------------------------------------------

    if scaler is not None and scaler_path.exists():

        scaler.load_state_dict(
            torch.load(
                scaler_path,
                map_location="cpu",
                weights_only=True,
            )
        )

        print(
            "[RESUME] AMP scaler: PASS",
            flush=True,
        )

    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    state = json.loads(
        state_path.read_text(
            encoding="utf-8"
        )
    )

    previous_steps = int(
        state["optimizer_steps"]
    )

    print(
        f"[RESUME] Epoch 2 optimizer steps: "
        f"{previous_steps}",
        flush=True,
    )

    print("=" * 72)

    return previous_steps


# ============================================================
# TRAIN EPOCH 2
# ============================================================
def train_epoch3(
    model,
    loader,
    optimizer,
    scaler,
    ctc_loss_fn,
    optimizer_steps,
):

    model.train()

    # Frozen layers stay in evaluation mode.
    model.encoder.feature_projection.eval()

    for idx in range(20):

        model.encoder.encoder.layers[
            idx
        ].eval()

    optimizer.zero_grad(
        set_to_none=True
    )

    total_loss = 0.0
    total_items = 0

    start_time = time.time()

    for batch_number, batch in enumerate(
        loader,
        start=1,
    ):

        global_idx = batch_number - 1

        text = batch["texts"][0]

        sample_start = time.time()

        try:

            audio = batch["audio"]

            input_features = (
                model.processor(
                    audio.numpy(),
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True,
                )
                .input_features
                .to(
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
                    .log_softmax(dim=-1)
                    .transpose(0, 1)
                )

                loss = ctc_loss_fn(
                    log_probs,
                    targets,
                    input_lengths,
                    target_lengths,
                )

                scaled_loss = (
                    loss / GRAD_ACCUM
                )

            scaled_loss.backward()

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            should_step = (
                batch_number % GRAD_ACCUM == 0
            )

            if should_step:

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                optimizer_steps += 1

            loss_value = float(
                loss.item()
            )

            total_loss += loss_value
            total_items += 1

            vram = 0.0

            if DEVICE.type == "cuda":

                vram = (
                    torch.cuda.memory_allocated(
                        DEVICE
                    )
                    / 1024**3
                )

            elapsed = (
                time.time()
                - sample_start
            )

            append_csv(
                TRAIN_LOG,
                TRAIN_HEADER,
                [
                    2,
                    global_idx,
                    batch_number,
                    optimizer_steps,
                    f"{loss_value:.8f}",
                    optimizer.param_groups[0]["lr"],
                    optimizer.param_groups[1]["lr"],
                    f"{vram:.4f}",
                    int(audio.shape[1]),
                    int(
                        batch[
                            "target_lengths"
                        ][0]
                    ),
                    f"{elapsed:.4f}",
                    "ok",
                    text,
                ],
            )

            if batch_number % 100 == 0:

                print(
                    f"[EPOCH 3] "
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
                    3,
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
            print(
                f"global_sample={global_idx}"
            )
            print(
                f"batch={batch_number}"
            )
            print(
                f"text={text}"
            )
            print(
                f"{type(exc).__name__}: {exc}"
            )
            print("=" * 72)

            raise

    # --------------------------------------------------------
    # Finish final partial gradient accumulation
    # --------------------------------------------------------

    if len(loader) % GRAD_ACCUM != 0:

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            MAX_GRAD_NORM,
        )

        optimizer.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        optimizer_steps += 1

    avg_loss = (
        total_loss
        / max(total_items, 1)
    )

    elapsed = (
        time.time()
        - start_time
    )

    return (
        avg_loss,
        optimizer_steps,
        elapsed,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    seed_everything()

    EPOCH3_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "MANIPURI ASR V2 — B2 EPOCH 3"
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

    # --------------------------------------------------------
    # Verify training cache
    # --------------------------------------------------------

    cache_count = len(
        list(
            AUDIO_CACHE.glob("*.pt")
        )
    )

    print(
        "[CACHE]",
        cache_count,
        "files",
    )

    if cache_count < 7768:

        raise RuntimeError(
            f"Expected at least 7768 "
            f"training cache files, "
            f"found {cache_count}"
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_df = pd.read_csv(
        ASR_DIR / "combined_dataset" / "train.csv",
        encoding="utf-8-sig",
    )

    if len(train_df) != 7768:

        raise RuntimeError(
            f"Expected 7768 training rows, "
            f"found {len(train_df)}"
        )

    vocab_path = (
        RUN_DIR / "vocab.json"
    )

    if not vocab_path.exists():

        raise FileNotFoundError(
            f"Vocabulary missing: "
            f"{vocab_path}"
        )

    vocab = json.loads(
        vocab_path.read_text(
            encoding="utf-8"
        )
    )

    print(
        "[VOCAB]",
        len(vocab),
        "tokens",
    )

    # --------------------------------------------------------
    # Full Epoch 2 dataset
    # --------------------------------------------------------

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
        pin_memory=(
            DEVICE.type == "cuda"
        ),
    )

    print(
        "[DATA] Epoch 3 samples:",
        len(dataset),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = create_model(
        vocab
    )

    optimizer = create_optimizer(
        model
    )

    scaler = None

    if DEVICE.type == "cuda":

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

    # --------------------------------------------------------
    # Resume Epoch 1
    # --------------------------------------------------------

    optimizer_steps = load_epoch2_checkpoint(
        model,
        optimizer,
        scaler,
    )

    # --------------------------------------------------------
    # CTC
    # --------------------------------------------------------

    ctc_loss_fn = nn.CTCLoss(
        blank=vocab[
            SPECIAL_BLANK
        ],
        zero_infinity=True,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    avg_loss, optimizer_steps, elapsed = (
        train_epoch3(
            model,
            loader,
            optimizer,
            scaler,
            ctc_loss_fn,
            optimizer_steps,
        )
    )

    # --------------------------------------------------------
    # Save Epoch 2
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        EPOCH3_DIR / "model.pt",
    )

    torch.save(
        optimizer.state_dict(),
        EPOCH3_DIR / "optimizer.pt",
    )

    if scaler is not None:

        torch.save(
            scaler.state_dict(),
            EPOCH3_DIR / "scaler.pt",
        )

    state = {
        "epoch": 3,
        "start_sample": 0,
        "end_sample": 7767,
        "samples_processed": 7768,
        "optimizer_steps": optimizer_steps,
        "avg_loss": avg_loss,
        "elapsed_seconds": elapsed,
        "status": "complete",
        "source_checkpoint": str(
            EPOCH2_DIR
        ),
    }

    (
        EPOCH3_DIR / "state.json"
    ).write_text(
        json.dumps(
            state,
            indent=2,
        ),
        encoding="utf-8",
    )

    append_csv(
        SUMMARY_LOG,
        SUMMARY_HEADER,
        [
            3,
            0,
            7767,
            7768,
            optimizer_steps,
            f"{avg_loss:.8f}",
            f"{elapsed:.3f}",
            "complete",
        ],
    )

    print()
    print("=" * 72)

    print(
        "B2 EPOCH 3 COMPLETE"
    )

    print(
        f"Average loss: {avg_loss:.6f}"
    )

    print(
        f"Optimizer steps: {optimizer_steps}"
    )

    print(
        f"Checkpoint: {EPOCH3_DIR}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
