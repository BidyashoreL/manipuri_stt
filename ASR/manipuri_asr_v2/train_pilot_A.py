
$ASR = (Resolve-Path "..").Path
$EXP = Join-Path $ASR "experiments\pilot_A"
New-Item -ItemType Directory -Force -Path $EXP | Out-Null
$SCRIPT = Join-Path $EXP "train_pilot_A.py"

@'
import os, sys, json, time, random, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import bitsandbytes as bnb
from jiwer import cer, wer

ASR = Path(__file__).resolve().parents[2]
V2 = ASR / "manipuri_asr_v2"
TRAIN_CSV = ASR / "xlit" / "audio_audit" / "final_candidate" / "train_16k.csv"
VAL_CSV = ASR / "xlit" / "audio_audit" / "final_candidate" / "validation_16k.csv"
INITIAL_CKPT = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"
OUT = ASR / "experiments" / "pilot_A"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
TRAIN_LIMIT = 256
VAL_LIMIT = 128
EPOCHS = 2
LR = 1e-5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for this pilot.")

torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda")
print("ASR root:", ASR)
print("Device:", torch.cuda.get_device_name(0))
print("Train manifest:", TRAIN_CSV)
print("Validation manifest:", VAL_CSV)

for p in [TRAIN_CSV, VAL_CSV, INITIAL_CKPT]:
    if not p.exists():
        raise FileNotFoundError(p)

# Import the original B3 implementation with its expected working directory.
sys.path.insert(0, str(V2))
old_cwd = Path.cwd()
os.chdir(V2)
try:
    import train_b3_narrow_E11_E12 as b3
finally:
    os.chdir(old_cwd)

def norm(text):
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split())

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)

required = {"audio_path", "text"}
if not required.issubset(train_df.columns):
    raise ValueError(f"Training CSV columns are {list(train_df.columns)}")
if not required.issubset(val_df.columns):
    raise ValueError(f"Validation CSV columns are {list(val_df.columns)}")

# Deterministic subsets: this pilot does not touch the test set.
train_df = train_df.sample(
    n=min(TRAIN_LIMIT, len(train_df)), random_state=SEED
).reset_index(drop=True)
val_df = val_df.sample(
    n=min(VAL_LIMIT, len(val_df)), random_state=SEED
).reset_index(drop=True)

vocab = b3.load_vocab()
id2char = {int(i): ch for ch, i in vocab.items()}
blank_id = int(vocab.get("<blank>", 0))
unk_id = vocab.get("<unk>")

def encode(text):
    ids = [int(vocab.get(ch, unk_id if unk_id is not None else blank_id))
           for ch in norm(text)]
    # Blank is a CTC alignment symbol, never a transcript character.
    ids = [i for i in ids if i != blank_id]
    if not ids:
        raise ValueError(f"Empty target after encoding: {text!r}")
    return ids

def decode_greedy(logits):
    # logits: [1, frames, vocabulary]
    ids = logits[0].argmax(dim=-1).detach().cpu().tolist()
    out = []
    previous = None
    for idx in ids:
        if idx != previous and idx != blank_id:
            token = id2char.get(idx, "")
            if token not in ("<blank>", "<pad>"):
                out.append(token)
        previous = idx
    return norm("".join(out))

def read_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim != 1:
        raise ValueError(f"Expected mono audio, got shape {audio.shape}: {path}")
    if sr != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sr}: {path}")
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    return audio

print("Loading B3 best checkpoint...")
model = b3.B2Model(len(vocab))
payload = torch.load(INITIAL_CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(payload["model_state_dict"], strict=True)
model.to(device)

# Configuration A: preserve B3 adapter and CTC head; unfreeze encoder layers 20–23.
for p in model.parameters():
    p.requires_grad = False

layers = model.encoder.encoder.layers
if len(layers) < 24:
    raise RuntimeError(f"Expected at least 24 encoder layers, found {len(layers)}")

for layer_idx in range(20, 24):
    for p in layers[layer_idx].parameters():
        p.requires_grad = True

for p in model.adapter.parameters():
    p.requires_grad = True
for p in model.ctc_head.parameters():
    p.requires_grad = True

trainable = [p for p in model.parameters() if p.requires_grad]
trainable_count = sum(p.numel() for p in trainable)
total_count = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_count:,}/{total_count:,}")

model.encoder.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)
optimizer = bnb.optim.AdamW8bit(
    trainable, lr=LR, weight_decay=0.01
)
ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

def run_validation():
    model.eval()
    refs, hyps = [], []
    loss_sum = 0.0
    skipped = 0

    with torch.no_grad():
        for _, row in val_df.iterrows():
            try:
                audio = read_audio(row["audio_path"])
                ref = norm(row["text"])
                target_ids = encode(ref)
                features = b3.make_features(
                    model.processor, [torch.from_numpy(audio.copy())]
                ).to(device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(features)

                log_probs = logits.float().log_softmax(-1).transpose(0, 1)
                input_lengths = torch.tensor(
                    [logits.shape[1]], dtype=torch.long, device=device
                )
                target = torch.tensor(target_ids, dtype=torch.long, device=device)
                target_lengths = torch.tensor(
                    [len(target_ids)], dtype=torch.long, device=device
                )
                loss = ctc_loss(
                    log_probs, target, input_lengths, target_lengths
                )

                if not torch.isfinite(loss):
                    skipped += 1
                    continue

                loss_sum += float(loss.item())
                refs.append(ref)
                hyps.append(decode_greedy(logits))
            except Exception as exc:
                raise RuntimeError(
                    f"Validation failed for {row['audio_path']}: {exc}"
                ) from exc

    if not refs:
        raise RuntimeError("No validation samples were evaluated.")

    return {
        "val_loss": loss_sum / len(refs),
        "val_cer": float(cer(refs, hyps)),
        "val_wer": float(wer(refs, hyps)),
        "evaluated": len(refs),
        "skipped": skipped,
    }

best_cer = float("inf")
history = []
start_all = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    order = list(range(len(train_df)))
    random.Random(SEED + epoch).shuffle(order)
    losses = []
    epoch_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    for step, idx in enumerate(order, start=1):
        row = train_df.iloc[idx]
        audio = read_audio(row["audio_path"])
        target_ids = encode(row["text"])

        features = b3.make_features(
            model.processor, [torch.from_numpy(audio.copy())]
        ).to(device)
        targets = torch.tensor(target_ids, dtype=torch.long, device=device)
        target_lengths = torch.tensor(
            [len(target_ids)], dtype=torch.long, device=device
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(features)

        log_probs = logits.float().log_softmax(-1).transpose(0, 1)
        input_lengths = torch.tensor(
            [logits.shape[1]], dtype=torch.long, device=device
        )
        loss = ctc_loss(
            log_probs, targets, input_lengths, target_lengths
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, sample {row['audio_path']}"
            )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite gradient norm.")
        optimizer.step()
        losses.append(float(loss.item()))

        if step % 25 == 0 or step == len(order):
            print(
                f"Epoch {epoch}/{EPOCHS} | step {step}/{len(order)} "
                f"| loss {np.mean(losses[-25:]):.4f}",
                flush=True
            )

    metrics = run_validation()
    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    record = {
        "epoch": epoch,
        "train_loss": float(np.mean(losses)),
        **metrics,
        "peak_gpu_gib": peak_gib,
        "epoch_seconds": time.time() - epoch_start,
    }
    history.append(record)
    print("EPOCH RESULT:", json.dumps(record, ensure_ascii=False), flush=True)

    # Save a resumable checkpoint only inside the pilot folder.
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": record,
        "config": {
            "seed": SEED, "lr": LR, "train_limit": TRAIN_LIMIT,
            "val_limit": VAL_LIMIT, "layers": [20, 21, 22, 23],
            "precision": "bf16", "optimizer": "AdamW8bit",
            "initial_checkpoint": str(INITIAL_CKPT),
        },
    }
    torch.save(state, OUT / "latest.pt")
    if metrics["val_cer"] < best_cer:
        best_cer = metrics["val_cer"]
        torch.save(state, OUT / "best.pt")
        print("Saved new best pilot checkpoint:", OUT / "best.pt")

    with open(OUT / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

print("\nPILOT COMPLETE")
print("Elapsed minutes:", round((time.time() - start_all) / 60, 2))
print("Best validation CER:", round(best_cer * 100, 4), "%")
print("Results:", OUT)
print("Original B3 checkpoint was read-only:", INITIAL_CKPT)
print("Official test set was not accessed.")
'@ | Set-Content -Encoding utf8 $SCRIPT

Write-Host "Created: $SCRIPT"
