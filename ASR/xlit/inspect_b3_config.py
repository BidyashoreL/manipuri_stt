from pathlib import Path
import torch

p = Path(r"..\checkpoints\v3_ctc\b3_narrow\b3_narrow_epoch11_BEST.pt")
# Current folder is ASR\xlit, so the path above may not resolve.
# Resolve from the ASR directory instead:
p = Path.cwd().parent / "checkpoints" / "v3_ctc" / "b3_narrow" / "b3_narrow_epoch11_BEST.pt"

obj = torch.load(p, map_location="cpu", weights_only=True)

print("Checkpoint:", p)
print("Epoch:", obj.get("epoch"))
print("Beam width:", obj.get("beam_width"))
print("Normalization:", obj.get("normalization"))
print("Trainable layers:", obj.get("trainable_layers"))
print("Available keys:", list(obj.keys()))
