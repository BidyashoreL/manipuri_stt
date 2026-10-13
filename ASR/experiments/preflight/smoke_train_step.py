from pathlib import Path
import os, sys, gc, unicodedata, re
import torch
import torch.nn as nn
import pandas as pd
import soundfile as sf
import bitsandbytes as bnb

ASR = Path(__file__).resolve().parents[2]
V2 = ASR / "manipuri_asr_v2"
MANIFEST = ASR / "xlit" / "audio_audit" / "final_candidate" / "train_16k.csv"
CHECKPOINT = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"

sys.path.insert(0, str(V2))
old_cwd = Path.cwd()
os.chdir(V2)
try:
    import train_b3_narrow_E11_E12 as b3
finally:
    os.chdir(old_cwd)

def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()

def main():
    assert torch.cuda.is_available(), "CUDA is required for this test."
    assert torch.cuda.is_bf16_supported(), "BF16 is unsupported."
    assert hasattr(bnb.optim, "AdamW8bit"), "8-bit AdamW is unavailable."

    df = pd.read_csv(MANIFEST)
    row = df.iloc[0]
    audio_path = Path(row["audio_path"])
    audio, sr = sf.read(audio_path, dtype="float32")
    assert sr == 16000 and audio.ndim == 1, (sr, audio.shape)

    vocab = b3.load_vocab()
    text = normalize(row["text"])
    unk_id = vocab["<unk>"]
    target_ids = [vocab.get(ch, unk_id) for ch in text]
    target_ids = [i for i in target_ids if i != vocab["<blank>"]]
    assert target_ids, "Transcript encoded to an empty target."

    print("Loading B3 best checkpoint...")
    model = b3.B2Model(len(vocab))
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload

    # Experiment A parameter policy: train layers 20–23 + adapter + CTC head.
    for p in model.parameters():
        p.requires_grad = False
    for idx in [20, 21, 22, 23]:
        for p in model.encoder.encoder.layers[idx].parameters():
            p.requires_grad = True
    for p in model.adapter.parameters():
        p.requires_grad = True
    for p in model.ctc_head.parameters():
        p.requires_grad = True

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(trainable, lr=1e-5, weight_decay=0.01)

    # Non-reentrant checkpointing supports gradients when early layers are frozen.
    model.encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()

    features = b3.make_features(
        model.processor, [torch.from_numpy(audio.copy())]
    ).to(b3.DEVICE)

    targets = torch.tensor(target_ids, dtype=torch.long, device=b3.DEVICE)
    target_lengths = torch.tensor([len(target_ids)], dtype=torch.long)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(features)

    log_probs = logits.float().log_softmax(dim=-1).transpose(0, 1)
    input_lengths = torch.tensor([logits.shape[1]], dtype=torch.long)

    loss = ctc(
        log_probs,
        targets,
        input_lengths,
        target_lengths
    )

    print("Audio:", audio_path.name)
    print("Transcript characters:", len(text))
    print("Encoded target length:", len(target_ids))
    print("Output frames:", logits.shape[1])
    print("Logits dtype:", logits.dtype)
    print("CTC loss:", float(loss.detach().cpu()))

    if not torch.isfinite(loss):
        raise RuntimeError("Non-finite CTC loss.")

    loss.backward()

    grads = [
        p.grad for p in trainable
        if p.grad is not None
    ]
    if not grads:
        raise RuntimeError("No trainable parameter received gradients.")

    finite_grads = all(torch.isfinite(g).all().item() for g in grads)
    if not finite_grads:
        raise RuntimeError("Non-finite gradients detected.")

    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    print("Parameters with gradients:", len(grads))
    print("Gradient norm:", float(grad_norm))
    print("All gradients finite:", finite_grads)

    optimizer.step()
    print("8-bit AdamW step: PASS")
    print("Peak GPU allocated GiB:",
          round(torch.cuda.max_memory_allocated() / 1024**3, 3))
    print("\nPASS: one optimizer step completed.")
    print("No checkpoint saved; original best.pt was not modified.")

    del model, optimizer, logits, log_probs, loss, features
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
