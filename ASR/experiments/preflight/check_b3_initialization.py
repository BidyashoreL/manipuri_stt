from pathlib import Path
import os, sys, json, gc
import torch
import pandas as pd
import soundfile as sf

ASR = Path(__file__).resolve().parents[2]
V2 = ASR / "manipuri_asr_v2"
MANIFEST = ASR / "xlit" / "audio_audit" / "final_candidate" / "validation_16k.csv"
CHECKPOINT = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"

# The original B3 script defines its paths from the current directory.
sys.path.insert(0, str(V2))
old_cwd = Path.cwd()
os.chdir(V2)
try:
    import train_b3_narrow_E11_E12 as b3
finally:
    os.chdir(old_cwd)

def main():
    print("=" * 65)
    print("B3 CHECKPOINT INITIALIZATION TEST — NO TRAINING")
    print("=" * 65)
    print("ASR root:", ASR)
    print("Device:", b3.DEVICE)

    assert MANIFEST.exists(), f"Missing manifest: {MANIFEST}"
    assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"

    vocab = b3.load_vocab()
    print("Vocabulary size:", len(vocab))

    model = b3.B2Model(len(vocab))
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    result = model.load_state_dict(payload["model_state_dict"], strict=True)
    print("Checkpoint epoch:", payload["epoch"])
    print("Strict checkpoint load:", result)

    # Experiment A: same learned B3 model, now unfreeze layers 20–23.
    for p in model.parameters():
        p.requires_grad = False

    for layer_idx in [20, 21, 22, 23]:
        for p in model.encoder.encoder.layers[layer_idx].parameters():
            p.requires_grad = True

    for p in model.adapter.parameters():
        p.requires_grad = True
    for p in model.ctc_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable percentage: {100 * trainable / total:.3f}%")

    # Enable checkpointing for the upcoming training harness.
    model.encoder.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled.")

    # Test BF16 forward on one real validation audio; no weight updates.
    df = pd.read_csv(MANIFEST)
    row = df.iloc[0]
    audio_path = Path(row["audio_path"])
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim != 1:
        raise RuntimeError(f"Expected mono audio, got shape {audio.shape}")
    if sr != 16000:
        raise RuntimeError(f"Expected 16000 Hz, got {sr}")

    audio_tensor = torch.from_numpy(audio.copy())
    features = b3.make_features(model.processor, [audio_tensor])
    features = features.to(b3.DEVICE)

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=(b3.DEVICE.type == "cuda")
    ):
        logits = model(features)

    print("Audio:", audio_path.name)
    print("Audio seconds:", round(len(audio) / sr, 3))
    print("Logits shape:", tuple(logits.shape))
    print("Logits dtype:", logits.dtype)
    print("All logits finite:", bool(torch.isfinite(logits).all().item()))

    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits detected.")

    print("Peak GPU allocated GiB:",
          round(torch.cuda.max_memory_allocated() / 1024**3, 3)
          if torch.cuda.is_available() else "CPU")
    print("\nPASS: checkpoint loaded strictly and BF16 forward succeeded.")
    print("No training step was performed; best.pt remains unchanged.")

    del model, payload, logits, features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
