from pathlib import Path
import torch

p = Path(r"..\checkpoints\v3_ctc\b3_narrow")

for f in sorted(p.glob("epoch_*.pt")):
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)

        print("=" * 70)
        print(f.name)
        print("epoch      :", ckpt.get("epoch"))
        print("train_loss :", ckpt.get("train_loss"))
        print("val_loss   :", ckpt.get("val_loss"))
        print("greedy CER :", ckpt.get("val_greedy_cer"))
        print("greedy WER :", ckpt.get("val_greedy_wer"))
        print("beam CER   :", ckpt.get("val_beam500_cer"))
        print("beam WER   :", ckpt.get("val_beam500_wer"))
        print("beam width :", ckpt.get("beam_width"))

    except Exception as e:
        print("=" * 70)
        print(f.name, "CORRUPT/UNREADABLE")
        print(type(e).__name__, e)

print("\nDONE")
