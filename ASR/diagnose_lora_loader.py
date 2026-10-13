import sys
from pathlib import Path

ROOT = Path(r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt")
sys.path.insert(0, str(ROOT / "ASR"))

print("[1] Importing recovery module...")
import train_epoch7_recovery as r

print("[2] Loading tokenizer...")
tokenizer = r.load_tokenizer()
print("[3] Tokenizer:", len(tokenizer))

print("[4] Temporarily disabling legacy LoRA loader...")

original_loader = r.load_legacy_lora_weights

def bypass_loader(model):
    print("[BYPASS] Legacy LoRA loader skipped.")
    return None

r.load_legacy_lora_weights = bypass_loader

print("[5] Building Qwen without legacy LoRA loading...")
model = r.build_qwen(tokenizer)

print("[6] build_qwen() SUCCESS")

# Restore original function
r.load_legacy_lora_weights = original_loader

print("[7] Opening Epoch-6 adapter...")
sf = list(r.EPOCH6_ADAPTER.glob("adapter_model*.safetensors"))[0]
print("[8] Adapter:", sf.name)

from safetensors import safe_open
import torch

with safe_open(str(sf), framework="pt", device="cpu") as f:

    keys = [
        k for k in f.keys()
        if ".lora_A." in k or ".lora_B." in k
    ]

    print("[9] LoRA key count:", len(keys))

    legacy_key = keys[0]
    print("[10] First legacy key:")
    print("    ", legacy_key)

    parts = legacy_key.split(".")
    layers_idx = parts.index("layers")
    layer_idx = int(parts[layers_idx + 1])

    module_parts = parts[layers_idx + 2:]
    lora_type = module_parts[-2]
    module_path_parts = module_parts[:-2]

    print("[11] Layer:", layer_idx)
    print("[12] LoRA type:", lora_type)
    print("[13] Module path:", module_path_parts)

    print("[14] Getting transformer layer...")
    layer = model.base_model.model.model.layers[layer_idx]
    print("[15] Layer SUCCESS")

    target_module = layer

    for part in module_path_parts:
        print("[16] getattr:", part)
        target_module = getattr(target_module, part)

    print("[17] Target module:", type(target_module))

    print("[18] Getting LoRA container...")
    lora_container = getattr(target_module, lora_type)
    print("[19] LoRA container:", type(lora_container))

    print("[20] Getting default adapter...")
    default_lora = getattr(lora_container, "default")
    print("[21] Default LoRA:", type(default_lora))

    print("[22] Getting target weight...")
    target_parameter = default_lora.weight

    print("[23] Target shape:", target_parameter.shape)
    print("[24] Target device:", target_parameter.device)
    print("[25] Target dtype:", target_parameter.dtype)

    print("[26] Reading source tensor...")
    tensor = f.get_tensor(legacy_key)

    print("[27] Source shape:", tensor.shape)
    print("[28] Source dtype:", tensor.dtype)
    print("[29] Source finite:", bool(torch.isfinite(tensor).all()))

    print("[30] Moving source to CUDA...")
    tensor_device = tensor.to(
        device=target_parameter.device,
        dtype=target_parameter.dtype,
    )

    print("[31] Source moved:", tensor_device.device)
    print("[32] Source moved shape:", tensor_device.shape)

    print("[33] COPYING FIRST TENSOR...")

    with torch.no_grad():
        target_parameter.copy_(tensor_device)

    print("[34] FIRST TENSOR COPY SUCCESS")

print("[35] Diagnostic completed successfully.")
