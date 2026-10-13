
from pathlib import Path
import torch
from safetensors import safe_open

epoch6_adapter = Path(r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt\ASR\checkpoints\epoch_6_broken_objective\qwen_lora")
out_path = Path(r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt\ASR\temp_lora_preload\legacy_lora_tensors.pt")

files = [
    f for f in epoch6_adapter.glob("*.safetensors")
    if "adapter_model" in f.name
]

if not files:
    raise FileNotFoundError(
        "No Epoch-6 adapter_model*.safetensors found."
    )

tensors = {}

with safe_open(
    str(files[0]),
    framework="pt",
    device="cpu",
) as f:
    for key in f.keys():
        if ".lora_A." in key or ".lora_B." in key:
            tensors[key] = f.get_tensor(key).contiguous()

if len(tensors) != 504:
    raise RuntimeError(
        f"Expected 504 LoRA tensors, got {len(tensors)}"
    )

torch.save(tensors, out_path)
print(f"Saved {len(tensors)} LoRA tensors to {out_path}")
