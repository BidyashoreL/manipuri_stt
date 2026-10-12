import os, sys
from pathlib import Path
import torch
import torch.nn as nn

ASR = Path(__file__).resolve().parents[2]
V2 = ASR / "manipuri_asr_v2"
CKPT = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"

sys.path.insert(0, str(V2))
old = Path.cwd()
os.chdir(V2)
try:
    import train_b3_narrow_E11_E12 as b3
finally:
    os.chdir(old)

vocab = b3.load_vocab()
model = b3.B2Model(len(vocab))
payload = torch.load(CKPT, map_location="cpu", weights_only=True)
model.load_state_dict(payload["model_state_dict"], strict=True)

print("\n=== ENCODER LINEAR MODULE NAMES ===")
names = []
for name, module in model.encoder.named_modules():
    if isinstance(module, nn.Linear):
        names.append(name)
        if any(part in name.lower() for part in
               ("att", "query", "key", "value", "proj", "linear")):
            print(f"{name}: {module.in_features} -> {module.out_features}")

print("\n=== PROPOSED TARGET CHECK ===")
for target in ["q_proj", "k_proj", "v_proj", "out_proj"]:
    matches = [n for n in names if n.split(".")[-1] == target]
    print(f"{target}: {len(matches)} exact matches")
    for name in matches[:5]:
        print("  ", name)

print("\n=== POSSIBLE ATTENTION PROJECTIONS ===")
for suffix in ["linear_q", "linear_k", "linear_v", "linear_out"]:
    matches = [n for n in names if n.split(".")[-1] == suffix]
    print(f"{suffix}: {len(matches)} exact matches")
    for name in matches[:3]:
        print("  ", name)

print("\nDiagnostic complete. No training performed.")
