from pathlib import Path
from safetensors import safe_open
import torch

ROOT = Path(r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt")

ADAPTER_DIR = ROOT / "ASR" / "checkpoints" / "epoch_6_broken_objective" / "qwen_lora"

print("[1] Adapter directory:")
print("   ", ADAPTER_DIR)

files = list(ADAPTER_DIR.glob("adapter_model*.safetensors"))

print("[2] Matching files:", len(files))

if not files:
    raise FileNotFoundError(
        f"No adapter_model*.safetensors found in {ADAPTER_DIR}"
    )

sf = files[0]

print("[3] Adapter:", sf.name)
print("[4] Size GB:", round(sf.stat().st_size / (1024**3), 3))

print("[5] Opening safetensors on CPU...")

with safe_open(
    str(sf),
    framework="pt",
    device="cpu",
) as f:

    print("[6] safe_open SUCCESS")

    keys = [
        k for k in f.keys()
        if ".lora_A." in k or ".lora_B." in k
    ]

    print("[7] LoRA keys:", len(keys))

    if not keys:
        raise RuntimeError("No LoRA tensors found.")

    key = keys[0]

    print("[8] First key:")
    print("   ", key)

    print("[9] Calling get_tensor...")

    tensor = f.get_tensor(key)

    print("[10] get_tensor SUCCESS")
    print("[11] Shape:", tuple(tensor.shape))
    print("[12] Dtype:", tensor.dtype)
    print("[13] Device:", tensor.device)
    print("[14] Finite:", bool(torch.isfinite(tensor).all()))
    print("[15] Min:", tensor.min().item())
    print("[16] Max:", tensor.max().item())

print("[17] safe_open closed successfully.")
print("[18] CPU-ONLY SAFETENSOR TEST COMPLETE")
