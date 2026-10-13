from pathlib import Path
import json
import torch

SAVE_DIR = Path(r"..\checkpoints\v3_ctc\b3_narrow")

history_path = SAVE_DIR / "training_history.json"
ckpt_path = SAVE_DIR / "epoch_08.pt"

history = json.loads(history_path.read_text(encoding="utf-8"))

ckpt = torch.load(
    ckpt_path,
    map_location="cpu",
    weights_only=True
)

epoch8 = {
    "epoch": 8,
    "train_loss": float(ckpt["train_loss"]),
    "loss": float(ckpt["val_loss"]),
    "greedy_cer": float(ckpt["val_greedy_cer"]),
    "greedy_wer": float(ckpt["val_greedy_wer"]),
    "beam_cer": float(ckpt["val_beam500_cer"]),
    "beam_wer": float(ckpt["val_beam500_wer"]),
}

history = [x for x in history if int(x["epoch"]) != 8]
history.append(epoch8)
history.sort(key=lambda x: int(x["epoch"]))

tmp = history_path.with_suffix(".json.tmp")
tmp.write_text(
    json.dumps(history, indent=2),
    encoding="utf-8"
)
tmp.replace(history_path)

print("=" * 60)
print("HISTORY REPAIRED")
print("=" * 60)
print(json.dumps(epoch8, indent=2))
print()
print("History epochs:", [x["epoch"] for x in history])
