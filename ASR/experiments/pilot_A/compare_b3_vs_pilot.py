import os, sys, json, time, random, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from jiwer import cer, wer

ASR = Path(__file__).resolve().parents[2]
V2 = ASR / "manipuri_asr_v2"
VAL_CSV = ASR / "xlit" / "audio_audit" / "final_candidate" / "validation_16k.csv"
ORIGINAL_CKPT = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"
PILOT_CKPT = ASR / "experiments" / "pilot_A" / "best.pt"
OUT = ASR / "experiments" / "pilot_A"

SEED = 2026
VAL_LIMIT = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required for evaluation.")

device = torch.device("cuda")

sys.path.insert(0, str(V2))
old_cwd = Path.cwd()
os.chdir(V2)
try:
    import train_b3_narrow_E11_E12 as b3
finally:
    os.chdir(old_cwd)

def norm(text):
    text = unicodedata.normalize("NFC", str(text))
    return " ".join(text.split())

val_df = pd.read_csv(VAL_CSV)
val_df = val_df.sample(
    n=min(VAL_LIMIT, len(val_df)), random_state=SEED
).reset_index(drop=True)

vocab = b3.load_vocab()
id2char = {int(i): ch for ch, i in vocab.items()}
blank_id = int(vocab.get("<blank>", 0))
unk_id = vocab.get("<unk>")

def encode(text):
    ids = [int(vocab.get(ch, unk_id if unk_id is not None else blank_id))
           for ch in norm(text)]
    ids = [i for i in ids if i != blank_id]
    return ids

def decode_greedy(logits):
    ids = logits[0].argmax(dim=-1).detach().cpu().tolist()
    out = []
    previous = None
    for idx in ids:
        if idx != previous and idx != blank_id:
            token = id2char.get(idx, "")
            if token not in ("<blank>", "<pad>"):
                out.append(token)
        # Update on every frame, including blanks.
        previous = idx
    return norm("".join(out))

def read_audio(path):
    audio, sr = sf.read(str(path), dtype="float32")
    return audio

ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

def eval_model(ckpt_path, label):
    print(f"Evaluating {label}...")
    model = b3.B2Model(len(vocab))
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    refs, hyps = [], []
    loss_sum = 0.0

    with torch.no_grad():
        for _, row in val_df.iterrows():
            audio = read_audio(row["audio_path"])
            ref = norm(row["text"])
            target_ids = encode(ref)
            features = b3.make_features(
                model.processor, [torch.from_numpy(audio.copy())]
            ).to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(features)

            log_probs = logits.float().log_softmax(-1).transpose(0, 1)
            input_lengths = torch.tensor([logits.shape[1]], dtype=torch.long, device=device)
            target = torch.tensor(target_ids, dtype=torch.long, device=device)
            target_lengths = torch.tensor([len(target_ids)], dtype=torch.long, device=device)
            
            loss = ctc_loss(log_probs, target, input_lengths, target_lengths)
            if torch.isfinite(loss):
                loss_sum += float(loss.item())
            
            refs.append(ref)
            hyps.append(decode_greedy(logits))

    m_cer = float(cer(refs, hyps))
    m_wer = float(wer(refs, hyps))
    m_loss = loss_sum / len(refs)
    
    del model
    torch.cuda.empty_cache()
    
    return {
        "model": label,
        "loss": round(m_loss, 4),
        "cer": round(m_cer * 100, 2),
        "wer": round(m_wer * 100, 2)
    }

orig_res = eval_model(ORIGINAL_CKPT, "Original B3")
pilot_res = eval_model(PILOT_CKPT, "Pilot A (Fine-Tuned)")

print("\n" + "="*55)
print("HEAD-TO-HEAD EVALUATION (128 MATCHED VAL CLIPS)")
print("="*55)
print(f"{'Metric':<10} | {'Original B3':<14} | {'Pilot A':<14} | {'Delta':<10}")
print("-" * 55)
for metric in ["loss", "cer", "wer"]:
    o_val = orig_res[metric]
    p_val = pilot_res[metric]
    diff = round(p_val - o_val, 2)
    sign = "+" if diff > 0 else ""
    unit = "%" if metric in ["cer", "wer"] else ""
    print(f"{metric.upper():<10} | {o_val}{unit:<13} | {p_val}{unit:<13} | {sign}{diff}{unit}")
print("="*55)

with open(OUT / "comparison.json", "w", encoding="utf-8") as f:
    json.dump({"original": orig_res, "pilot_A": pilot_res}, f, indent=2)

print(f"\nSaved comparison summary to: {OUT / 'comparison.json'}")
