from pathlib import Path
import torch

p = Path(r"..\checkpoints\v3_ctc\b3_narrow")

results = []

for f in p.glob("epoch_*.pt"):
    try:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)

        results.append({
            "file": f.name,
            "epoch": ckpt.get("epoch"),
            "train_loss": ckpt.get("train_loss"),
            "val_loss": ckpt.get("val_loss"),
            "cer": ckpt.get("val_beam500_cer"),
            "wer": ckpt.get("val_beam500_wer"),
            "time": f.stat().st_mtime
        })

    except Exception as e:
        print(f"{f.name}: ERROR - {e}")

print("\n" + "="*80)
print("CHECKPOINT SUMMARY")
print("="*80)

for r in sorted(results, key=lambda x: x["epoch"]):
    print(
        f"Epoch {r['epoch']:>2} | "
        f"Val Loss: {r['val_loss']:.6f} | "
        f"Beam CER: {r['cer']:.6f} | "
        f"Beam WER: {r['wer']:.6f} | "
        f"{r['file']}"
    )

best_cer = min(results, key=lambda x: x["cer"])
best_wer = min(results, key=lambda x: x["wer"])
latest = max(results, key=lambda x: x["epoch"])

print("\n" + "="*80)
print("BEST CHECKPOINTS")
print("="*80)

print(f"BEST CER : {best_cer['file']} | Epoch {best_cer['epoch']} | {best_cer['cer']:.4%}")
print(f"BEST WER : {best_wer['file']} | Epoch {best_wer['epoch']} | {best_wer['wer']:.4%}")
print(f"LAST EPOCH: {latest['file']} | Epoch {latest['epoch']}")

print("="*80)
