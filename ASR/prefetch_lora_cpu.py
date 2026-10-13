from pathlib import Path
from safetensors import safe_open
import torch

ROOT = Path(r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt")

ADAPTER_DIR = (
    ROOT
    / "ASR"
    / "checkpoints"
    / "epoch_6_broken_objective"
    / "qwen_lora"
)

sf = next(ADAPTER_DIR.glob("adapter_model*.safetensors"))

print("[1] Adapter:", sf.name)
print("[2] Size GB:", round(sf.stat().st_size / (1024**3), 3))

print("[3] Opening safetensors on CPU...")

lora_tensors = {}

with safe_open(
    str(sf),
    framework="pt",
    device="cpu",
) as f:

    keys = [
        k for k in f.keys()
        if ".lora_A." in k or ".lora_B." in k
    ]

    print("[4] LoRA tensors:", len(keys))

    if len(keys) != 504:
        raise RuntimeError(
            f"Expected 504 LoRA tensors, found {len(keys)}"
        )

    for i, key in enumerate(keys, 1):

        tensor = f.get_tensor(key)

        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Non-finite tensor: {key}"
            )

        lora_tensors[key] = tensor

        if i % 50 == 0:
            print(
                f"[5] Loaded {i}/504 "
                f"CPU tensors"
            )

print("[6] safe_open closed.")
print("[7] Loaded tensors:", len(lora_tensors))

total_elements = sum(
    t.numel()
    for t in lora_tensors.values()
)

total_bytes = sum(
    t.numel() * t.element_size()
    for t in lora_tensors.values()
)

print(
    "[8] Total elements:",
    total_elements
)

print(
    "[9] Total tensor memory:",
    round(total_bytes / (1024**2), 2),
    "MB"
)

first_key = next(iter(lora_tensors))

print("[10] First tensor:")
print("    ", first_key)
print("    shape =", tuple(lora_tensors[first_key].shape))
print("    dtype =", lora_tensors[first_key].dtype)
print("    device =", lora_tensors[first_key].device)

print("[11] PREFETCH TEST SUCCESS")
