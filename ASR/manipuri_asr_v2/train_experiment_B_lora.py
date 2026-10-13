
$ASR = (Resolve-Path "..").Path
$EXP = Join-Path $ASR "experiments\experiment_B_lora"
New-Item -ItemType Directory -Force -Path $EXP | Out-Null
$SCRIPT = Join-Path $EXP "train_experiment_B_lora.py"

@'
import os, sys, json, time, random, unicodedata, argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
from jiwer import cer, wer

ASR = Path(__file__).resolve().parents[2]
V2 = ASR / "manipuri_asr_v2"
TRAIN_CSV = ASR / "xlit" / "audio_audit" / "final_candidate" / "train_16k.csv"
VAL_CSV = ASR / "xlit" / "audio_audit" / "final_candidate" / "validation_16k.csv"
INITIAL_CKPT = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"
OUT = ASR / "experiments" / "experiment_B_lora"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 2026
EPOCHS = 5
LR = 1e-5
LORA_R = 16
LORA_ALPHA = 32

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True

for path in [TRAIN_CSV, VAL_CSV, INITIAL_CKPT]:
    if not path.exists():
        raise FileNotFoundError(path)

sys.path.insert(0, str(V2))
old_cwd = Path.cwd()
os.chdir(V2)
try:
    import train_b3_narrow_E11_E12 as b3
finally:
    os.chdir(old_cwd)

def norm(text):
    return " ".join(unicodedata.normalize("NFC", str(text)).split())

def read_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    if sr != 16000 or audio.ndim != 1:
        raise ValueError(f"Expected mono 16 kHz audio: {path}; sr={sr}, shape={audio.shape}")
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid audio: {path}")
    return audio

train_df = pd.read_csv(TRAIN_CSV).reset_index(drop=True)
val_df = pd.read_csv(VAL_CSV).reset_index(drop=True)
for name, df in [("train", train_df), ("validation", val_df)]:
    if not {"audio_path", "text"}.issubset(df.columns):
        raise ValueError(f"{name} manifest lacks audio_path/text columns.")
    if df["audio_path"].isna().any() or df["text"].isna().any():
        raise ValueError(f"{name} manifest has missing paths or transcripts.")

vocab = b3.load_vocab()
id2char = {int(i): ch for ch, i in vocab.items()}
blank_id = int(vocab.get("<blank>", 0))
unk_id = vocab.get("<unk>")

def encode(text):
    ids = [
        int(vocab.get(ch, unk_id if unk_id is not None else blank_id))
        for ch in norm(text)
    ]
    ids = [i for i in ids if i != blank_id]
    if not ids:
        raise ValueError(f"Transcript encodes to empty target: {text!r}")
    return ids

def decode_greedy(logits):
    # CTC collapse: merge adjacent repeats, but blanks reset repetition.
    ids = logits[0].argmax(dim=-1).detach().cpu().tolist()
    out, previous = [], None
    for idx in ids:
        if idx != previous and idx != blank_id:
            token = id2char.get(idx, "")
            if token not in ("<blank>", "<pad>"):
                out.append(token)
        previous = idx
    return norm("".join(out))

print("Device:", torch.cuda.get_device_name(0))
print("Training clips:", len(train_df))
print("Validation clips:", len(val_df))
print("Loading original B3 best checkpoint...")

model = b3.B2Model(len(vocab))
base_payload = torch.load(INITIAL_CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(base_payload["model_state_dict"], strict=True)
del base_payload
model.to(device)

# Enable activation checkpointing before wrapping the encoder with PEFT.
model.encoder.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["linear_q", "linear_k", "linear_v", "linear_out"],
    lora_dropout=0.05,
    bias="none",
)
model.encoder = get_peft_model(model.encoder, config)

# Train LoRA plus the existing B3 adapter and CTC head.
for p in model.adapter.parameters():
    p.requires_grad = True
for p in model.ctc_head.parameters():
    p.requires_grad = True

trainable = [p for p in model.parameters() if p.requires_grad]
trainable_count = sum(p.numel() for p in trainable)
total_count = sum(p.numel() for p in model.parameters())
lora_count = sum(
    p.numel() for name, p in model.named_parameters()
    if p.requires_grad and "lora_" in name
)
print(f"LoRA rank: {LORA_R}, alpha: {LORA_ALPHA}")
print(f"Trainable parameters: {trainable_count:,}/{total_count:,} ({100*trainable_count/total_count:.3f}%)")
print(f"LoRA trainable parameters: {lora_count:,}")

optimizer = bnb.optim.AdamW8bit(trainable, lr=LR, weight_decay=0.01)
loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=False)

def trainable_state_dict():
    return {
        name: p.detach().cpu().contiguous()
        for name, p in model.named_parameters() if p.requires_grad
    }

def save_checkpoint(path, epoch, best_cer, history):
    torch.save({
        "epoch": epoch,
        "best_cer": best_cer,
        "trainable_state_dict": trainable_state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "history": history,
        "seed": SEED,
        "config": {
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "targets": ["linear_q", "linear_k", "linear_v", "linear_out"],
            "lr": LR,
            "train_manifest": str(TRAIN_CSV),
            "validation_manifest": str(VAL_CSV),
            "initial_checkpoint": str(INITIAL_CKPT),
        },
    }, path)

def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    current = dict(model.named_parameters())
    saved = state["trainable_state_dict"]
    expected = {n for n, p in current.items() if p.requires_grad}
    if set(saved) != expected:
        raise RuntimeError(
            f"Resume trainable-key mismatch. Missing={len(expected-set(saved))}, "
            f"unexpected={len(set(saved)-expected)}"
        )
    with torch.no_grad():
        for name, tensor in saved.items():
            current[name].copy_(tensor.to(device=current[name].device,
                                          dtype=current[name].dtype))
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state

def run_validation(epoch):
    model.eval()
    refs, hyps = [], []
    losses = []
    predictions = []

    with torch.no_grad():
        for i, row in val_df.iterrows():
            try:
                audio = read_audio(row["audio_path"])
                ref = norm(row["text"])
                target_ids = encode(ref)
                features = b3.make_features(
                    model.processor, [torch.from_numpy(audio.copy())]
                ).to(device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(features)

                if len(target_ids) > logits.shape[1]:
                    raise RuntimeError(
                        f"CTC target longer than output frames at {row['audio_path']}"
                    )

                log_probs = logits.float().log_softmax(-1).transpose(0, 1)
                input_lengths = torch.tensor(
                    [logits.shape[1]], dtype=torch.long, device=device
                )
                targets = torch.tensor(target_ids, dtype=torch.long, device=device)
                target_lengths = torch.tensor(
                    [len(target_ids)], dtype=torch.long, device=device
                )
                loss = loss_fn(log_probs, targets, input_lengths, target_lengths)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite validation loss: {row['audio_path']}")

                hyp = decode_greedy(logits)
                refs.append(ref)
                hyps.append(hyp)
                losses.append(float(loss.item()))
                predictions.append({
                    "audio_path": row["audio_path"],
                    "reference": ref,
                    "prediction": hyp,
                })
            except Exception as exc:
                raise RuntimeError(
                    f"Validation failed at row {i}, path={row['audio_path']}: {exc}"
                ) from exc

            if (i + 1) % 250 == 0:
                print(f"Validation: {i+1}/{len(val_df)}", flush=True)

    pd.DataFrame(predictions).to_csv(
        OUT / f"validation_predictions_epoch_{epoch}.csv", index=False
    )
    return {
        "val_loss": float(np.mean(losses)),
        "val_cer": float(cer(refs, hyps)),
        "val_wer": float(wer(refs, hyps)),
        "evaluated": len(refs),
    }

history, best_cer, start_epoch = [], float("inf"), 1
latest_path = OUT / "latest_lora.pt"
best_path = OUT / "best_lora.pt"

if args.resume:
    if not latest_path.exists():
        raise FileNotFoundError(f"Cannot resume; missing {latest_path}")
    state = load_checkpoint(latest_path)
    history = state.get("history", [])
    best_cer = state.get("best_cer", float("inf"))
    start_epoch = int(state["epoch"]) + 1
    print(f"Resuming from completed epoch {start_epoch-1}")
    del state
elif latest_path.exists() or best_path.exists():
    raise RuntimeError(
        "Experiment files already exist. Use --resume to continue latest, "
        "or move the old experiment folder before starting a fresh run."
    )

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    order = list(range(len(train_df)))
    random.Random(SEED + epoch).shuffle(order)
    epoch_losses = []
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

        if len(target_ids) > logits.shape[1]:
            raise RuntimeError(
                f"CTC target longer than output frames: {row['audio_path']}"
            )

        log_probs = logits.float().log_softmax(-1).transpose(0, 1)
        input_lengths = torch.tensor(
            [logits.shape[1]], dtype=torch.long, device=device
        )
        loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss at epoch={epoch}, path={row['audio_path']}"
            )

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite gradient norm.")
        optimizer.step()
        epoch_losses.append(float(loss.item()))

        if step % 500 == 0 or step == len(order):
            print(
                f"Epoch {epoch}/{EPOCHS} | {step}/{len(order)} "
                f"| recent train loss {np.mean(epoch_losses[-100:]):.4f}",
                flush=True
            )

    metrics = run_validation(epoch)
    record = {
        "epoch": epoch,
        "train_loss": float(np.mean(epoch_losses)),
        **metrics,
        "peak_gpu_gib": torch.cuda.max_memory_allocated() / (1024**3),
        "epoch_seconds": time.time() - epoch_start,
    }
    history.append(record)
    print("EPOCH RESULT:", json.dumps(record), flush=True)

    save_checkpoint(latest_path, epoch, best_cer, history)
    if metrics["val_cer"] < best_cer:
        best_cer = metrics["val_cer"]
        save_checkpoint(best_path, epoch, best_cer, history)
        print("Saved best checkpoint:", best_path, flush=True)

    with open(OUT / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

print("\nEXPERIMENT B COMPLETE")
print("Best validation CER:", round(best_cer * 100, 3), "%")
print("Results directory:", OUT)
print("Original B3 checkpoint was read-only.")
print("Official test set was not accessed.")
'@ | Set-Content -Encoding utf8 $SCRIPT

Write-Host "Created: $SCRIPT"
