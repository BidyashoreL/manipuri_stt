import os, sys, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model

ASR = Path(__file__).resolve().parents[2]
V2 = ASR / "manipuri_asr_v2"
TRAIN_CSV = ASR / "xlit" / "audio_audit" / "final_candidate" / "train_16k.csv"
CKPT = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")

device = torch.device("cuda")
sys.path.insert(0, str(V2))
old_cwd = Path.cwd()
os.chdir(V2)
try:
    import train_b3_narrow_E11_E12 as b3
finally:
    os.chdir(old_cwd)

def norm(text):
    return " ".join(unicodedata.normalize("NFC", str(text)).split())

vocab = b3.load_vocab()
blank_id = int(vocab.get("<blank>", 0))
unk_id = vocab.get("<unk>")

df = pd.read_csv(TRAIN_CSV)
row = df.iloc[0]
audio, sr = sf.read(str(row["audio_path"]), dtype="float32")
if audio.ndim != 1 or sr != 16000 or len(audio) == 0:
    raise ValueError(f"Unexpected audio format: shape={audio.shape}, sr={sr}")

text = norm(row["text"])
target_ids = [
    int(vocab.get(ch, unk_id if unk_id is not None else blank_id))
    for ch in text
]
target_ids = [i for i in target_ids if i != blank_id]
if not target_ids:
    raise ValueError("Transcript encoded to an empty target.")

print("Loading original B3 checkpoint...")
model = b3.B2Model(len(vocab))
payload = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(payload["model_state_dict"], strict=True)
model.to(device)
model.eval()

features = b3.make_features(
    model.processor, [torch.from_numpy(audio.copy())]
).to(device)

# Baseline logits before adding LoRA.
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    original_logits = model(features).float()

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["linear_q", "linear_k", "linear_v", "linear_out"],
    lora_dropout=0.05,
    bias="none",
)
model.encoder = get_peft_model(model.encoder, config)

# Confirm exactly 96 attention projections were adapted.
lora_modules = [
    name for name, module in model.encoder.named_modules()
    if hasattr(module, "lora_A") and hasattr(module, "lora_B")
]
print("LoRA-wrapped modules:", len(lora_modules))
for suffix in ["linear_q", "linear_k", "linear_v", "linear_out"]:
    count = sum(name.endswith(suffix) for name in lora_modules)
    print(f"  {suffix}: {count}")

if len(lora_modules) != 96:
    raise RuntimeError(
        f"Expected 96 LoRA-wrapped projections, got {len(lora_modules)}"
    )

# Keep the B3 adapter and CTC head trainable alongside LoRA.
for p in model.adapter.parameters():
    p.requires_grad = True
for p in model.ctc_head.parameters():
    p.requires_grad = True

trainable = [p for p in model.parameters() if p.requires_grad]
trainable_count = sum(p.numel() for p in trainable)
total_count = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_count:,}/{total_count:,}")
print(f"Trainable percentage: {100*trainable_count/total_count:.3f}%")

# LoRA's initial B matrices should make the wrapped model equivalent
# to the original model before any optimizer update.
model.eval()
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    wrapped_logits = model(features).float()

if original_logits.shape != wrapped_logits.shape:
    raise RuntimeError(
        f"Logit shape changed: {original_logits.shape} vs {wrapped_logits.shape}"
    )

max_diff = (original_logits - wrapped_logits).abs().max().item()
print("Initial logits max absolute difference:", max_diff)
if not torch.allclose(original_logits, wrapped_logits, atol=0.03, rtol=0.03):
    raise RuntimeError(
        "LoRA-wrapped model does not reproduce original B3 logits closely."
    )

optimizer = bnb.optim.AdamW8bit(
    trainable, lr=1e-5, weight_decay=0.01
)
loss_fn = nn.CTCLoss(blank=blank_id, zero_infinity=True)
targets = torch.tensor(target_ids, dtype=torch.long, device=device)
target_lengths = torch.tensor([len(target_ids)], dtype=torch.long, device=device)

model.train()
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    logits = model(features)

log_probs = logits.float().log_softmax(-1).transpose(0, 1)
input_lengths = torch.tensor(
    [logits.shape[1]], dtype=torch.long, device=device
)
loss = loss_fn(log_probs, targets, input_lengths, target_lengths)

if not torch.isfinite(loss):
    raise RuntimeError(f"Non-finite CTC loss: {loss.item()}")

loss.backward()

groups = {
    "LoRA": [],
    "B3 adapter": [],
    "CTC head": [],
}
for name, p in model.named_parameters():
    if p.requires_grad and p.grad is not None:
        if "lora_" in name:
            groups["LoRA"].append(p.grad)
        elif name.startswith("adapter."):
            groups["B3 adapter"].append(p.grad)
        elif name.startswith("ctc_head."):
            groups["CTC head"].append(p.grad)

for name, grads in groups.items():
    count = sum(g.numel() for g in grads)
    finite = all(torch.isfinite(g).all().item() for g in grads)
    norm = torch.sqrt(sum((g.float() ** 2).sum() for g in grads)).item() if grads else 0
    print(f"{name}: gradient elements={count}, norm={norm:.6f}, finite={finite}")
    if not grads or not finite:
        raise RuntimeError(f"Missing or invalid gradients for {name}.")

torch.nn.utils.clip_grad_norm_(trainable, 1.0)
optimizer.step()

print("CTC loss:", float(loss.item()))
print("Peak GPU allocated GiB:", round(
    torch.cuda.max_memory_allocated() / (1024**3), 3
))
print("PASS: LoRA forward/backward and one optimizer update succeeded.")
print("No checkpoint saved; original B3 checkpoint was not modified.")
