from pathlib import Path
import json
import gc
import os
import shutil
import tempfile
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SAVE_DIR = BASE_DIR / ".." / "checkpoints" / "v3_ctc" / "b3_narrow"
SAVE_DIR = SAVE_DIR.resolve()

START_CHECKPOINT = SAVE_DIR / "epoch_08.pt"

HISTORY_PATH = SAVE_DIR / "training_history.json"

# Continue from Epoch 8
START_EPOCH = 9
END_EPOCH = 12

print("=" * 70)
print("B3 NARROW — EPOCH 9 → 12 RESUME")
print("=" * 70)

print("[PATH] SAVE_DIR       :", SAVE_DIR)
print("[PATH] START CHECKPOINT:", START_CHECKPOINT)
print("[TRAIN] Epochs        :", START_EPOCH, "→", END_EPOCH)


# ============================================================
# SAFETY CHECK
# ============================================================

if not START_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Epoch 8 checkpoint not found:\n{START_CHECKPOINT}"
    )

print("\n[CHECK] Loading Epoch 8 checkpoint...")

checkpoint = torch.load(
    START_CHECKPOINT,
    map_location="cpu",
    weights_only=True
)

print("[CHECK] checkpoint epoch :", checkpoint.get("epoch"))
print("[CHECK] train loss       :", checkpoint.get("train_loss"))
print("[CHECK] val loss         :", checkpoint.get("val_loss"))
print("[CHECK] beam500 CER      :", checkpoint.get("val_beam500_cer"))
print("[CHECK] beam500 WER      :", checkpoint.get("val_beam500_wer"))

if int(checkpoint.get("epoch", -1)) != 8:
    raise RuntimeError(
        "Safety check failed: expected Epoch 8 checkpoint."
    )

print("[CHECK] Epoch 8 checkpoint: VALID")


# ============================================================
# IMPORTANT
# ============================================================
#
# This script is designed to be used together with the existing
# B3 training implementation.
#
# It imports the model/dataset/training definitions from the
# existing robust script and changes only the resume behavior.
#
# ============================================================

SOURCE_SCRIPT = BASE_DIR / "train_b3_narrow_ROBUST_RESUME_EPOCH8_FINAL.py"

if not SOURCE_SCRIPT.exists():
    raise FileNotFoundError(
        f"\nExisting B3 training script not found:\n{SOURCE_SCRIPT}\n"
        "\nMake sure this file is in the same directory."
    )


# ============================================================
# LOAD EXISTING TRAINING SCRIPT
# ============================================================

print("\n[INFO] Loading existing B3 training implementation...")

source = SOURCE_SCRIPT.read_text(encoding="utf-8")

# ------------------------------------------------------------
# We execute definitions from the existing script without
# allowing its main() to execute.
# ------------------------------------------------------------

namespace = {
    "__file__": str(SOURCE_SCRIPT),
    "__name__": "__b3_resume_import__",
}

# Prevent the original main guard from running.
source_for_import = source.replace(
    'if __name__ == "__main__":',
    'if False:'
)

exec(compile(
    source_for_import,
    str(SOURCE_SCRIPT),
    "exec"
), namespace)


# ============================================================
# REQUIRED OBJECTS
# ============================================================

required = [
    "build_model",
    "build_dataloaders",
    "validate",
]

missing = [
    x for x in required
    if x not in namespace
]

if missing:
    raise RuntimeError(
        "Existing B3 script is missing required functions: "
        + ", ".join(missing)
    )


build_model = namespace["build_model"]
build_dataloaders = namespace["build_dataloaders"]
validate = namespace["validate"]


# ============================================================
# OPTIONAL OBJECTS FROM EXISTING SCRIPT
# ============================================================

for name in [
    "DEVICE",
    "VOCAB",
    "INV_VOCAB",
    "DECODER",
    "BEAM_WIDTH",
    "TRAINABLE_LAYERS",
    "EPOCHS",
]:
    if name in namespace:
        globals()[name] = namespace[name]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\n[DEVICE]", DEVICE)

if DEVICE.type == "cuda":
    print(
        "[GPU]",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# BUILD MODEL
# ============================================================

print("\n[MODEL] Building B3 model...")

model = build_model()

model = model.to(DEVICE)


# ============================================================
# LOAD EPOCH 8
# ============================================================

print("\n[MODEL] Loading Epoch 8 weights...")

state_dict = None

for key in [
    "model_state_dict",
    "state_dict",
    "model",
]:
    if key in checkpoint:
        state_dict = checkpoint[key]
        print("[LOAD] Using key:", key)
        break

if state_dict is None:
    raise RuntimeError(
        "Could not find model state dict in Epoch 8 checkpoint."
    )

missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=True
)

print("[LOAD] Missing keys   :", len(missing_keys))
print("[LOAD] Unexpected keys:", len(unexpected_keys))

if missing_keys or unexpected_keys:
    raise RuntimeError(
        "Strict checkpoint loading failed."
    )

print("[LOAD] Epoch 8 weights loaded successfully.")


# ============================================================
# VERIFY TRAINABLE LAYERS
# ============================================================

print("\n[VERIFY] Trainable parameters:")

trainable = 0
total = 0

for name, param in model.named_parameters():

    total += param.numel()

    if param.requires_grad:
        trainable += param.numel()

        print(
            "  TRAINABLE:",
            name,
            param.numel()
        )

print("\n[VERIFY]")
print("Total parameters    :", total)
print("Trainable parameters:", trainable)
print(
    "Trainable ratio     :",
    f"{100.0 * trainable / total:.4f}%"
)


# ============================================================
# DATA
# ============================================================

print("\n[DATA] Building dataloaders...")

loaders = build_dataloaders()

if isinstance(loaders, dict):

    train_loader = loaders["train"]
    val_loader = loaders["val"]

else:

    train_loader, val_loader = loaders[:2]


print("[DATA] Train samples:", len(train_loader.dataset))
print("[DATA] Val samples  :", len(val_loader.dataset))


# ============================================================
# OPTIMIZER
# ============================================================

print("\n[OPTIMIZER] Creating fresh optimizer...")

encoder_params = []
adapter_params = []
head_params = []

for name, param in model.named_parameters():

    if not param.requires_grad:
        continue

    lname = name.lower()

    if "adapter" in lname:
        adapter_params.append(param)

    elif (
        "ctc" in lname
        or "classifier" in lname
        or "lm_head" in lname
        or "head" in lname
    ):
        head_params.append(param)

    else:
        encoder_params.append(param)


param_groups = []

if encoder_params:
    param_groups.append({
        "params": encoder_params,
        "lr": 5e-6,
    })

if adapter_params:
    param_groups.append({
        "params": adapter_params,
        "lr": 2e-4,
    })

if head_params:
    param_groups.append({
        "params": head_params,
        "lr": 2e-4,
    })


optimizer = torch.optim.AdamW(
    param_groups,
    weight_decay=0.02
)


# ============================================================
# AMP
# ============================================================

use_bf16 = (
    DEVICE.type == "cuda"
    and torch.cuda.is_bf16_supported()
)

if use_bf16:

    print("[AMP] Using BF16")

    scaler = None

    autocast_dtype = torch.bfloat16

else:

    print("[AMP] Using FP32")

    scaler = None

    autocast_dtype = torch.float32


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def atomic_torch_save(obj, target):

    target = Path(target)

    tmp = target.with_suffix(
        target.suffix + ".tmp"
    )

    if tmp.exists():
        tmp.unlink()

    torch.save(
        obj,
        tmp
    )

    # Verify temporary checkpoint BEFORE replacing target.
    print(
        f"[CHECKPOINT] Verifying {tmp.name}..."
    )

    test = torch.load(
        tmp,
        map_location="cpu",
        weights_only=True
    )

    if "model_state_dict" not in test:
        raise RuntimeError(
            "Checkpoint verification failed: "
            "model_state_dict missing."
        )

    del test

    tmp.replace(target)

    print(
        f"[CHECKPOINT] Saved safely: {target.name}"
    )


def update_hardlink(alias, target):

    alias = Path(alias)
    target = Path(target)

    tmp_alias = alias.with_suffix(
        alias.suffix + ".tmp"
    )

    if tmp_alias.exists():
        tmp_alias.unlink()

    try:
        os.link(
            target,
            tmp_alias
        )

        tmp_alias.replace(alias)

    except Exception:

        if tmp_alias.exists():
            tmp_alias.unlink()

        shutil.copy2(
            target,
            alias
        )


# ============================================================
# HISTORY
# ============================================================

if HISTORY_PATH.exists():

    history = json.loads(
        HISTORY_PATH.read_text(
            encoding="utf-8"
        )
    )

else:

    history = []


def update_history(record):

    global history

    epoch = int(record["epoch"])

    history = [
        x for x in history
        if int(x["epoch"]) != epoch
    ]

    history.append(record)

    history.sort(
        key=lambda x: int(x["epoch"])
    )

    tmp = HISTORY_PATH.with_suffix(
        ".json.tmp"
    )

    tmp.write_text(
        json.dumps(
            history,
            indent=2
        ),
        encoding="utf-8"
    )

    tmp.replace(
        HISTORY_PATH
    )


# ============================================================
# BEST CER
# ============================================================

existing_beam_cers = [
    float(x["beam_cer"])
    for x in history
    if "beam_cer" in x
]

if existing_beam_cers:

    best_beam_cer = min(existing_beam_cers)

else:

    best_beam_cer = float(
        checkpoint["val_beam500_cer"]
    )


print(
    "\n[BEST] Current best beam CER:",
    f"{best_beam_cer * 100:.4f}%"
)


# ============================================================
# TRAINING
# ============================================================

GRAD_ACCUM = 16
MAX_GRAD_NORM = 1.0


for epoch in range(
    START_EPOCH,
    END_EPOCH + 1
):

    print("\n")
    print("=" * 70)
    print(f"EPOCH {epoch}")
    print("=" * 70)

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_loss = 0.0
    samples_seen = 0
    optimizer_steps = 0

    start_time = time.time()

    for step, batch in enumerate(
        train_loader,
        start=1
    ):

        try:

            # Existing B3 training implementation should provide
            # the tensors in the batch.
            #
            # Handle common dictionary format.

            if isinstance(batch, dict):

                inputs = batch.get(
                    "input_features",
                    batch.get("features")
                )

                labels = batch.get(
                    "labels"
                )

                input_lengths = batch.get(
                    "input_lengths"
                )

                label_lengths = batch.get(
                    "label_lengths"
                )

            else:

                inputs, labels = batch[:2]

                input_lengths = (
                    batch[2]
                    if len(batch) > 2
                    else None
                )

                label_lengths = (
                    batch[3]
                    if len(batch) > 3
                    else None
                )


            if inputs is None or labels is None:
                raise RuntimeError(
                    "Could not identify inputs/labels in batch."
                )


            inputs = inputs.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            if use_bf16:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16
                ):

                    outputs = model(
                        inputs
                    )

            else:

                outputs = model(
                    inputs
                )


            # ------------------------------------------------
            # Extract logits
            # ------------------------------------------------

            if isinstance(outputs, dict):

                logits = outputs.get(
                    "logits"
                )

            elif hasattr(
                outputs,
                "logits"
            ):

                logits = outputs.logits

            else:

                logits = outputs


            if logits is None:
                raise RuntimeError(
                    "Model did not return logits."
                )


            # ------------------------------------------------
            # CTC
            # ------------------------------------------------

            log_probs = F.log_softmax(
                logits,
                dim=-1
            )

            # Expected model layout:
            # [B, T, V]
            #
            # CTC requires:
            # [T, B, V]

            log_probs = log_probs.transpose(
                0,
                1
            )


            if input_lengths is None:

                input_lengths = torch.full(
                    (logits.size(0),),
                    logits.size(1),
                    dtype=torch.long,
                    device=DEVICE
                )

            else:

                input_lengths = input_lengths.to(
                    DEVICE,
                    dtype=torch.long
                )


            if label_lengths is None:

                # Ignore padding/blank values.
                label_lengths = (
                    labels != 0
                ).sum(dim=1).long()

            else:

                label_lengths = label_lengths.to(
                    DEVICE,
                    dtype=torch.long
                )


            # Flatten targets for CTC.

            targets = []

            for i in range(
                labels.size(0)
            ):

                n = int(
                    label_lengths[i].item()
                )

                targets.append(
                    labels[i, :n]
                )

            targets = torch.cat(
                targets
            ).to(
                DEVICE,
                dtype=torch.long
            )


            loss = F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                label_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True
            )


            if not torch.isfinite(loss):

                raise RuntimeError(
                    f"Non-finite loss at step {step}: {loss}"
                )


            loss_scaled = (
                loss / GRAD_ACCUM
            )

            loss_scaled.backward()

            running_loss += float(
                loss.detach().item()
            )

            samples_seen += int(
                logits.size(0)
            )


            # ------------------------------------------------
            # Optimizer step
            # ------------------------------------------------

            if (
                step % GRAD_ACCUM == 0
                or step == len(train_loader)
            ):

                # Correct partial accumulation scaling.
                remainder = (
                    step % GRAD_ACCUM
                )

                if (
                    remainder != 0
                    and step == len(train_loader)
                ):

                    correction = (
                        GRAD_ACCUM
                        / remainder
                    )

                    for param in model.parameters():

                        if param.grad is not None:

                            param.grad.mul_(
                                correction
                            )


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                optimizer_steps += 1


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if step % 500 == 0:

                avg = (
                    running_loss / step
                )

                print(
                    f"[Epoch {epoch}] "
                    f"step={step}/{len(train_loader)} "
                    f"loss={avg:.6f}"
                )


            del outputs
            del logits
            del log_probs
            del loss

            if step % 100 == 0:

                gc.collect()

                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()


        except Exception as e:

            print(
                f"\n[ERROR] Training failed "
                f"at Epoch {epoch}, step {step}"
            )

            raise


    # ========================================================
    # TRAIN SUMMARY
    # ========================================================

    train_loss = (
        running_loss
        / max(1, len(train_loader))
    )

    elapsed = (
        time.time()
        - start_time
    )

    print("\n")
    print(
        f"[EPOCH {epoch}] "
        f"train loss = {train_loss:.6f}"
    )

    print(
        f"[EPOCH {epoch}] "
        f"optimizer steps = {optimizer_steps}"
    )

    print(
        f"[EPOCH {epoch}] "
        f"time = {elapsed / 60:.2f} min"
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    print(
        f"\n[VALIDATION] Epoch {epoch}"
    )

    model.eval()

    with torch.no_grad():

        validation_result = validate(
            model,
            val_loader,
            VOCAB,
            DECODER,
            INV_VOCAB
        )


    # --------------------------------------------------------
    # Support dictionary result
    # --------------------------------------------------------

    if isinstance(
        validation_result,
        dict
    ):

        val_loss = float(
            validation_result.get(
                "loss",
                validation_result.get(
                    "val_loss"
                )
            )
        )

        greedy_cer = float(
            validation_result.get(
                "greedy_cer"
            )
        )

        greedy_wer = float(
            validation_result.get(
                "greedy_wer"
            )
        )

        beam_cer = float(
            validation_result.get(
                "beam_cer",
                validation_result.get(
                    "beam500_cer"
                )
            )
        )

        beam_wer = float(
            validation_result.get(
                "beam_wer",
                validation_result.get(
                    "beam500_wer"
                )
            )
        )

    else:

        raise RuntimeError(
            "Unexpected validation result format."
        )


    print("\n")
    print("=" * 70)
    print(f"EPOCH {epoch} RESULTS")
    print("=" * 70)

    print(
        f"Train loss       : {train_loss:.6f}"
    )

    print(
        f"Val loss         : {val_loss:.6f}"
    )

    print(
        f"Greedy CER       : {greedy_cer * 100:.4f}%"
    )

    print(
        f"Greedy WER       : {greedy_wer * 100:.4f}%"
    )

    print(
        f"Beam-500 CER     : {beam_cer * 100:.4f}%"
    )

    print(
        f"Beam-500 WER     : {beam_wer * 100:.4f}%"
    )


    # ========================================================
    # CHECKPOINT PAYLOAD
    # ========================================================

    payload = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "train_loss":
            train_loss,

        "val_loss":
            val_loss,

        "val_greedy_cer":
            greedy_cer,

        "val_greedy_wer":
            greedy_wer,

        "val_beam500_cer":
            beam_cer,

        "val_beam500_wer":
            beam_wer,

        "beam_width":
            500,

        "optimizer_steps":
            optimizer_steps,

        "elapsed_seconds":
            elapsed,

        "config": {
            "trainable_layers": [22, 23],
            "encoder_lr": 5e-6,
            "adapter_lr": 2e-4,
            "head_lr": 2e-4,
            "weight_decay": 0.02,
            "grad_accum": 16,
            "max_grad_norm": 1.0,
            "amp_bf16": use_bf16,
        }
    }


    # ========================================================
    # SAVE NEW EPOCH FIRST
    # ========================================================

    epoch_path = SAVE_DIR / (
        f"epoch_{epoch:02d}.pt"
    )

    print(
        f"\n[CHECKPOINT] Saving {epoch_path.name}"
    )

    atomic_torch_save(
        payload,
        epoch_path
    )


    # ========================================================
    # UPDATE LAST ONLY AFTER EPOCH CHECKPOINT IS VERIFIED
    # ========================================================

    last_path = SAVE_DIR / "last.pt"

    update_hardlink(
        last_path,
        epoch_path
    )

    print(
        "[CHECKPOINT] last.pt updated."
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    improved = (
        beam_cer < best_beam_cer
    )

    if improved:

        best_beam_cer = beam_cer

        best_path = SAVE_DIR / "best.pt"

        update_hardlink(
            best_path,
            epoch_path
        )

        print(
            f"[BEST] NEW BEST! "
            f"Beam CER = {beam_cer * 100:.4f}%"
        )

    else:

        print(
            f"[BEST] Not improved. "
            f"Best remains "
            f"{best_beam_cer * 100:.4f}%"
        )


    # ========================================================
    # HISTORY
    # ========================================================

    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "loss": val_loss,
        "greedy_cer": greedy_cer,
        "greedy_wer": greedy_wer,
        "beam_cer": beam_cer,
        "beam_wer": beam_wer
    }

    update_history(
        record
    )

    print(
        "[HISTORY] Updated."
    )

    print(
        "[HISTORY] Epochs:",
        [
            x["epoch"]
            for x in history
        ]
    )


    # ========================================================
    # CLEANUP
    # ========================================================

    del payload

    gc.collect()

    if DEVICE.type == "cuda":

        torch.cuda.empty_cache()

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print(
            f"[GPU] allocated={allocated:.2f} GB "
            f"reserved={reserved:.2f} GB"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 70)
print("B3 EPOCH 9 → 12 COMPLETE")
print("=" * 70)

print(
    "Best Beam-500 CER:",
    f"{best_beam_cer * 100:.4f}%"
)

print(
    "Checkpoint directory:",
    SAVE_DIR
)

print(
    "\nIMPORTANT:"
)

print(
    "Official test set was NOT used during training."
)

print(
    "Run the official test ONLY after selecting the final"
    " validation checkpoint."
)