"""
Run this BEFORE downloading Qwen2.5-3B. It tests the exact thing that's
uncertain on your GPU (RTX 5060 Laptop, Blackwell, sm_120, compute capability
12.0 on Windows): whether bitsandbytes' 4-bit (NF4) CUDA kernel actually
executes on your card. No model weights are downloaded -- this builds a tiny
Linear4bit layer directly and runs a forward pass on it.

Usage:
    (venv) PS ...\manipuri_stt> python ASR\verify_bnb_4bit.py
"""

import sys

import torch

print("=" * 60)
print("Environment")
print("=" * 60)
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Compute capability:", torch.cuda.get_device_capability(0))
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM: {total_vram_gb:.1f} GB")

try:
    import bitsandbytes as bnb
    print("bitsandbytes:", bnb.__version__)
except ImportError as e:
    print("\nbitsandbytes is not installed / failed to import:", e)
    print("Run: pip install -U bitsandbytes")
    sys.exit(1)

if not torch.cuda.is_available():
    print("\nNo CUDA device visible to torch -- stop here and fix that first.")
    sys.exit(1)

print("\n" + "=" * 60)
print("Test 1: NF4 4-bit Linear layer forward pass (no download)")
print("=" * 60)

try:
    torch.manual_seed(0)
    ref_linear = torch.nn.Linear(512, 512, bias=False).half().cuda()

    q_linear = bnb.nn.Linear4bit(
        512, 512, bias=False,
        compute_dtype=torch.float16,
        quant_type="nf4",
    )
    q_linear.load_state_dict(ref_linear.state_dict(), strict=False)
    q_linear = q_linear.cuda()

    x = torch.randn(4, 512, dtype=torch.float16, device="cuda")
    y = q_linear(x)
    torch.cuda.synchronize()

    assert torch.isfinite(y).all(), "Output contains NaN/Inf"
    print("PASSED -- NF4 forward pass ran and produced finite output.")
    print("Output shape:", tuple(y.shape))
    nf4_ok = True
except Exception as e:
    print("FAILED -- NF4 4-bit kernel did not run on this GPU:")
    print(f"  {type(e).__name__}: {e}")
    nf4_ok = False

print("\n" + "=" * 60)
print("Test 2: 4-bit + backward pass through a LoRA-style adapter")
print("=" * 60)
print("(This mirrors your real training loop: frozen 4-bit base weight,")
print(" trainable fp16 LoRA A/B matrices around it, gradient must flow")
print(" into the LoRA matrices only.)")

if nf4_ok:
    try:
        lora_r = 8
        A = torch.nn.Linear(512, lora_r, bias=False).half().cuda()
        B = torch.nn.Linear(lora_r, 512, bias=False).half().cuda()
        A.weight.requires_grad_(True)
        B.weight.requires_grad_(True)

        x = torch.randn(4, 512, dtype=torch.float16, device="cuda")
        with torch.no_grad():
            base_out = q_linear(x)
        lora_out = B(A(x))
        out = base_out + lora_out
        loss = out.float().pow(2).mean()
        loss.backward()

        assert A.weight.grad is not None and torch.isfinite(A.weight.grad).all()
        assert B.weight.grad is not None and torch.isfinite(B.weight.grad).all()
        print("PASSED -- gradients flowed into the LoRA-style matrices correctly.")
    except Exception as e:
        print("FAILED at the LoRA-style backward pass:")
        print(f"  {type(e).__name__}: {e}")
else:
    print("SKIPPED (Test 1 failed first)")

print("\n" + "=" * 60)
print("Verdict")
print("=" * 60)
if nf4_ok:
    print("bitsandbytes 4-bit quantization WORKS on this GPU/driver/CUDA combo.")
    print("You're clear to proceed with the notebook's section 11 as written.")
else:
    print("bitsandbytes 4-bit quantization is BROKEN on this exact setup right now.")
    print("Don't burn time downloading Qwen2.5-3B yet. Options, in order of effort:")
    print("  1. pip install -U bitsandbytes   (get the newest wheel, then rerun this script)")
    print("  2. Check `pip show bitsandbytes` is >=0.45.3 -- older wheels predate sm_120 kernels")
    print("  3. If still broken: WSL2 + Ubuntu + the Linux bitsandbytes wheel tends to pick up")
    print("     Blackwell kernel support faster than native Windows wheels")
    print("  4. Last resort: skip 4-bit, load Qwen in bf16/fp16 and rely on LoRA + gradient")
    print("     checkpointing alone -- viable on 8GB only with a smaller LoRA rank and short")
    print("     audio clips; report back the actual OOM point if you hit one")