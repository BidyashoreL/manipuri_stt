import sys
import pickle
from pathlib import Path
from safetensors import safe_open

if len(sys.argv) != 3:
    raise SystemExit("Usage: extract_meitei_rows.py <safetensors> <output_pickle>")

sf_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

EMBED_KEY = "base_model.model.model.embed_tokens.weight"
LM_KEY = "base_model.model.lm_head.weight"

START = 151665
END = 151709

print("[SUBPROCESS] Opening Epoch-6 safetensors...")
print("[SUBPROCESS] File:", sf_path)

with safe_open(
    str(sf_path),
    framework="pt",
    device="cpu",
) as f:

    keys = set(f.keys())

    if EMBED_KEY not in keys:
        raise KeyError(f"Missing {EMBED_KEY}")

    if LM_KEY not in keys:
        raise KeyError(f"Missing {LM_KEY}")

    embed_rows = (
        f.get_slice(EMBED_KEY)[START:END]
        .float()
        .contiguous()
    )

    lm_rows = (
        f.get_slice(LM_KEY)[START:END]
        .float()
        .contiguous()
    )

if tuple(embed_rows.shape) != (44, 2048):
    raise RuntimeError(
        f"Unexpected embedding shape: {tuple(embed_rows.shape)}"
    )

if tuple(lm_rows.shape) != (44, 2048):
    raise RuntimeError(
        f"Unexpected LM-head shape: {tuple(lm_rows.shape)}"
    )

difference = (
    embed_rows - lm_rows
).abs().max().item()

print(
    "[SUBPROCESS] Embed rows:",
    tuple(embed_rows.shape),
)

print(
    "[SUBPROCESS] LM-head rows:",
    tuple(lm_rows.shape),
)

print(
    "[SUBPROCESS] Max embed/lm difference:",
    difference,
)

if difference > 1e-4:
    print(
        "[SUBPROCESS] WARNING: embedding and LM-head rows differ."
    )

with open(out_path, "wb") as f:
    pickle.dump(
        embed_rows,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print("[SUBPROCESS] Meitei-row extraction complete.")
