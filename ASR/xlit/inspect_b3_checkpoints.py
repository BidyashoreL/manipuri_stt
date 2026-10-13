from pathlib import Path
import torch

folder = Path(r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt\ASR\checkpoints\v3_ctc\b3_narrow")
files = [
    "b3_narrow_epoch11_BEST.pt",
    "best.pt",
    "last.pt",
    "epoch_10.pt",
    "epoch_09.pt",
    "epoch_08.pt",
]

for name in files:
    path = folder / name
    if not path.exists():
        print(f"\nMISSING: {name}")
        continue

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        print(f"\n--- {name} ---")
        if isinstance(obj, dict):
            for key in (
                "epoch", "train_loss", "val_loss",
                "val_greedy_cer", "val_greedy_wer",
                "val_beam500_cer", "val_beam500_wer"
            ):
                if key in obj:
                    print(f"{key}: {obj[key]}")
            print("Top-level keys:", list(obj.keys())[:25])
        else:
            print("Object type:", type(obj).__name__)
    except Exception as e:
        print(f"Inspection failed: {type(e).__name__}: {e}")
