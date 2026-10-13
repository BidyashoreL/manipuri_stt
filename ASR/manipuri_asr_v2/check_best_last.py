import torch
from pathlib import Path

p = Path(r"..\checkpoints\v3_ctc\b3_narrow")

for name in ["best.pt", "last.pt"]:
    f = p / name

    print("\n" + "="*60)
    print(name)

    ckpt = torch.load(f, map_location="cpu", weights_only=True)

    print("epoch      :", ckpt.get("epoch"))
    print("train_loss :", ckpt.get("train_loss"))
    print("val_loss   :", ckpt.get("val_loss"))
    print("greedy CER :", ckpt.get("val_greedy_cer"))
    print("greedy WER :", ckpt.get("val_greedy_wer"))
    print("beam CER   :", ckpt.get("val_beam500_cer"))
    print("beam WER   :", ckpt.get("val_beam500_wer"))
    print("beam width :", ckpt.get("beam_width"))
