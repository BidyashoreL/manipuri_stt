"""
Local training entrypoint. Config-driven (see config.json), runs the same
staged approach as the notebook: one-example sanity check -> 100-example
overfit test -> full training -> checkpoint each epoch -> light validation
eval each epoch.

Run from the manipuri_stt root:
    (venv) PS ...\\manipuri_stt> python ASR\\train.py

The model-loading and forward-pass functions here are copied verbatim from
the notebook (cells under sections 7, 9, 10, 11, 12, 14) -- not rewritten
from memory, to avoid reintroducing bugs we already found and fixed once.

PATCHED: bf16 instead of fp16 (fixes NaN loss on RTX 5060/Blackwell), plus
NaN-loss and non-finite-gradient guards so a bad example can never silently
corrupt the model weights again.
"""

import json
import random
import sys
import traceback
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio  # only torchaudio.functional.resample is used -- NOT torchaudio.load()
from jiwer import cer, wer
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

ROOT = Path(cfg["data_root"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE != "cuda":
    print("WARNING: no CUDA device found. This pipeline needs a GPU (4-bit QLoRA).")
    sys.exit(1)

SAMPLE_RATE = cfg["audio_sample_rate"]
MAX_AUDIO_SECONDS = cfg["max_audio_seconds"]
MAX_TEXT_TOKENS = cfg["max_text_tokens"]
SEED = cfg["seed"]
EARLY_STOPPING_PATIENCE = cfg.get("early_stopping_patience", 3)  # Default to 3 if not in config

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CHECKPOINT_DIR = ROOT / cfg["checkpoint_dir"]
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
print("Data root:", ROOT)
print("Checkpoint dir:", CHECKPOINT_DIR)

# ---------------------------------------------------------------------------
# Load splits
# ---------------------------------------------------------------------------
train_df = pd.read_csv(ROOT / cfg["train_csv"], encoding="utf-8-sig")
val_df = pd.read_csv(ROOT / cfg["val_csv"], encoding="utf-8-sig")
test_df = pd.read_csv(ROOT / cfg["test_csv"], encoding="utf-8-sig")
print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

# audio_path in these CSVs is already a full, verified path (from
# build_unified_corpus.py) -- no resolve_audio_path() needed here.
train_df = train_df[train_df["audio_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
val_df = val_df[val_df["audio_path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)
print(f"after existence re-check: train={len(train_df)}  val={len(val_df)}")

all_text = "".join(train_df["text"].tolist())
unique_chars = sorted(set(all_text))
print("Unique characters in train text:", len(unique_chars))

# ---------------------------------------------------------------------------
# Wav2Vec2-BERT 2.0 (frozen) -- verbatim from notebook section 7
# ---------------------------------------------------------------------------
from transformers import AutoFeatureExtractor, AutoModel

print("\nLoading Wav2Vec2-BERT 2.0 ...")
feature_extractor = AutoFeatureExtractor.from_pretrained(cfg["wav2vec_model"])
audio_encoder = AutoModel.from_pretrained(cfg["wav2vec_model"]).to(DEVICE)
for p in audio_encoder.parameters():
    p.requires_grad = False
audio_encoder.eval()
hidden_size = audio_encoder.config.hidden_size
print("Wav2Vec2-BERT hidden size:", hidden_size)

# ---------------------------------------------------------------------------
# Audio projector -- verbatim from notebook section 9
# ---------------------------------------------------------------------------
class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1024, llm_dim=2048, stride=2):
        super().__init__()
        self.downsample = nn.Conv1d(audio_dim, audio_dim, kernel_size=stride, stride=stride)
        self.proj = nn.Linear(audio_dim, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.downsample(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return self.norm(x)


projector = AudioProjector(audio_dim=hidden_size, llm_dim=2048).to(DEVICE)
print("Trainable projector params:", sum(p.numel() for p in projector.parameters()))

# ---------------------------------------------------------------------------
# Tokenizer + Meitei character analysis -- verbatim from notebook section 10
# ---------------------------------------------------------------------------
from transformers import AutoTokenizer

print("\nLoading Qwen tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(cfg["qwen_model"], use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

missing_chars = [c for c in unique_chars if c.strip() and c not in tokenizer.get_vocab()]
print("Missing as single tokens:", len(missing_chars), "/", len(unique_chars))

if cfg["add_meitei_chars"] and missing_chars:
    added = tokenizer.add_tokens(missing_chars)
    print("Added:", added, "-> new vocab size:", len(tokenizer))
else:
    print("No new characters added.")

# ---------------------------------------------------------------------------
# Qwen in 4-bit + LoRA -- verbatim from notebook section 11
# PATCHED: bfloat16 instead of float16. Your RTX 5060 (Blackwell) supports
# bf16 natively, and bf16's exponent range matches fp32's -- this removes
# the overflow-to-NaN failure mode that fp16 has on long training runs.
# ---------------------------------------------------------------------------
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

print("\nLoading Qwen2.5-3B-Instruct in 4-bit ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
qwen = AutoModelForCausalLM.from_pretrained(
    cfg["qwen_model"], quantization_config=bnb_config, device_map="auto", dtype=torch.bfloat16,
)
qwen.resize_token_embeddings(len(tokenizer))
qwen.config.use_cache = False
qwen = prepare_model_for_kbit_training(qwen)
if cfg["gradient_checkpointing"]:
    qwen.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
qwen = get_peft_model(qwen, lora_config)
qwen.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Batch construction + training step -- verbatim from notebook sections 12, 14
# ---------------------------------------------------------------------------
def load_audio_16k(path, max_seconds=MAX_AUDIO_SECONDS):
    # soundfile, not torchaudio.load() -- recent torchaudio versions require
    # a separate torchcodec install for .load() to work at all; this avoids
    # that dependency entirely.
    wav_np, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = torch.from_numpy(wav_np)
    if wav.ndim > 1:
        wav = wav.mean(dim=-1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav[: int(max_seconds * SAMPLE_RATE)]
    return wav.numpy()


def build_single_example(row):
    wav = load_audio_16k(row["audio_path"])
    x = feature_extractor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    x = {k: v.to(DEVICE) for k, v in x.items()}

    with torch.no_grad():
        h = audio_encoder(**x).last_hidden_state
    audio_emb = projector(h.float())

    target = tokenizer(row["text"], add_special_tokens=True, truncation=True,
                        max_length=MAX_TEXT_TOKENS, return_tensors="pt")
    target_ids = target.input_ids.to(DEVICE)

    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    if target_ids.shape[1] < 2:
        raise ValueError("Transcript is too short after tokenization.")

    text_input_ids = torch.cat([
        torch.tensor([[bos_id]], device=DEVICE, dtype=torch.long),
        target_ids[:, :-1],
    ], dim=1)
    text_emb = qwen.get_input_embeddings()(text_input_ids)

    inputs_embeds = torch.cat([audio_emb.to(text_emb.dtype), text_emb], dim=1)
    labels = torch.cat([
        torch.full((1, audio_emb.shape[1]), -100, dtype=torch.long, device=DEVICE),
        target_ids,
    ], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=DEVICE)

    return inputs_embeds, attention_mask, labels


def train_one_row(row):
    qwen.train()
    projector.train()
    inputs_embeds, attention_mask, labels = build_single_example(row)
    out = qwen(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    return out.loss


@torch.no_grad()
def greedy_transcribe(row, max_new_tokens=MAX_TEXT_TOKENS):
    qwen.eval()
    projector.eval()
    wav = load_audio_16k(row["audio_path"])
    x = feature_extractor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    x = {k: v.to(DEVICE) for k, v in x.items()}
    h = audio_encoder(**x).last_hidden_state
    audio_emb = projector(h.float())

    bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    bos_emb = qwen.get_input_embeddings()(torch.tensor([[bos_id]], device=DEVICE))
    prompt_emb = torch.cat([audio_emb.to(bos_emb.dtype), bos_emb], dim=1)
    mask = torch.ones(prompt_emb.shape[:2], dtype=torch.long, device=DEVICE)

    out_ids = qwen.generate(inputs_embeds=prompt_emb, attention_mask=mask,
                             max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def run_step_safely(fn, row, on_oom_msg):
    """
    OOM on one weird utterance shouldn't kill a multi-hour run.
    PATCHED: also catches a non-finite (NaN/inf) loss, which previously
    slipped through silently, got averaged into a gradient-accumulation
    window, and permanently corrupted the model weights on optimizer.step().
    """
    try:
        result = fn(row)
        if isinstance(result, torch.Tensor) and not torch.isfinite(result).all():
            print(f"  Non-finite loss on {row.get('audio_path', '?')} -- {on_oom_msg}, skipping")
            return None
        return result
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM on {row.get('audio_path', '?')} -- {on_oom_msg}, skipping")
        torch.cuda.empty_cache()
        return None
    except Exception as e:
        print(f"  Error on {row.get('audio_path', '?')}: {e}")
        return None


# ---------------------------------------------------------------------------
# Stage 1: one-example sanity check
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STAGE 1: one-example forward pass")
print("=" * 60)
row0 = train_df.iloc[0]
loss0 = run_step_safely(train_one_row, row0, "sanity check")
if loss0 is None:
    print("FAILED on the very first example. Stopping -- fix this before anything else.")
    sys.exit(1)
print(f"OK -- loss={loss0.item():.4f}")

# ---------------------------------------------------------------------------
# Stage 2: 100-example overfit test (per README: if this can't overfit, don't
# spend GPU time on the full run until it's fixed)
# PATCHED: skip optimizer.step() if the accumulated gradient norm is non-finite.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"STAGE 2: {cfg['overfit_test_size']}-example overfit test")
print("=" * 60)

from torch.optim import AdamW

optimizer = AdamW([
    {"params": projector.parameters(), "lr": cfg["projector_lr"]},
    {"params": [p for p in qwen.parameters() if p.requires_grad], "lr": cfg["qwen_lr"]},
], weight_decay=0.01)

tiny = train_df.head(cfg["overfit_test_size"]).copy().reset_index(drop=True)
GRAD_ACCUM = cfg["gradient_accumulation_steps"]
history = []
optimizer.zero_grad(set_to_none=True)

for epoch in range(cfg["overfit_test_epochs"]):
    running, steps = 0.0, 0
    for i, (_, row) in enumerate(tqdm(tiny.iterrows(), total=len(tiny), desc=f"Overfit epoch {epoch+1}")):
        loss = run_step_safely(train_one_row, row, "overfit test")
        if loss is None:
            continue
        loss = loss / GRAD_ACCUM
        loss.backward()
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(tiny):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(projector.parameters()) + [p for p in qwen.parameters() if p.requires_grad], 1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                print(f"  Skipping optimizer step -- non-finite grad norm ({grad_norm})")
            optimizer.zero_grad(set_to_none=True)
        running += float(loss.detach()) * GRAD_ACCUM
        steps += 1
    avg = running / max(steps, 1)
    history.append(avg)
    print(f"Overfit epoch {epoch+1}: avg loss = {avg:.4f}")

print("Overfit loss history:", history)
if len(history) >= 2 and history[-1] >= history[0]:
    print("\nWARNING: loss did not decrease over the overfit test. Something is likely")
    print("broken upstream (gradient flow, learning rate, data). Recommend stopping")
    print("here and investigating before the full run.")
    resp = input("Continue to full training anyway? [y/N] ").strip().lower()
    if resp != "y":
        sys.exit(1)
else:
    print("OK -- loss decreased. Pipeline can learn from this data.")

# ---------------------------------------------------------------------------
# Stage 3: full training
# PATCHED: same non-finite grad-norm guard as Stage 2.
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STAGE 3: full training")
print("=" * 60)


@torch.no_grad()
def quick_eval(df_subset, n):
    subset = df_subset.head(n)
    refs, hyps = [], []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Validating"):
        try:
            pred = greedy_transcribe(row)
        except Exception as e:
            print(f"  eval error on {row['audio_path']}: {e}")
            continue
        refs.append(row["text"])
        hyps.append(pred)
    if not refs:
        return float("nan"), float("nan")
    return cer(refs, hyps), wer(refs, hyps)


optimizer.zero_grad(set_to_none=True)
epoch_log = []

# Early stopping variables
best_val_wer = float("inf")
patience_counter = 0

for epoch in range(cfg["num_epochs"]):
    shuffled = train_df.sample(frac=1.0, random_state=SEED + epoch).reset_index(drop=True)
    running, steps = 0.0, 0

    for i, (_, row) in enumerate(tqdm(shuffled.iterrows(), total=len(shuffled), desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")):
        loss = run_step_safely(train_one_row, row, "training step")
        if loss is None:
            continue
        loss = loss / GRAD_ACCUM
        loss.backward()
        if (i + 1) % GRAD_ACCUM == 0 or (i + 1) == len(shuffled):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(projector.parameters()) + [p for p in qwen.parameters() if p.requires_grad], 1.0)
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                print(f"  Skipping optimizer step -- non-finite grad norm ({grad_norm})")
            optimizer.zero_grad(set_to_none=True)
        running += float(loss.detach()) * GRAD_ACCUM
        steps += 1

    avg_loss = running / max(steps, 1)
    val_cer, val_wer = quick_eval(val_df, cfg["val_eval_subset_per_epoch"])
    print(f"\nEpoch {epoch+1}: train_loss={avg_loss:.4f}  val_CER={val_cer:.4f}  val_WER={val_wer:.4f}")
    epoch_log.append({"epoch": epoch + 1, "train_loss": avg_loss, "val_CER": val_cer, "val_WER": val_wer})

    # Save epoch checkpoint
    epoch_dir = CHECKPOINT_DIR / f"epoch_{epoch+1}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), epoch_dir / "audio_projector.pt")
    qwen.save_pretrained(epoch_dir / "qwen_lora")
    tokenizer.save_pretrained(epoch_dir / "tokenizer")
    print(f"Saved checkpoint: {epoch_dir}")

    # -----------------------------------------------------------------------
    # Early Stopping Check
    # -----------------------------------------------------------------------
    if not np.isnan(val_wer):
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            patience_counter = 0
            print(f"New best validation WER: {best_val_wer:.4f}! Resetting patience counter.")

            # Save a dedicated 'best_model' copy
            best_dir = CHECKPOINT_DIR / "best_model"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(epoch_dir, best_dir)
            print(f"Updated best model at: {best_dir}")
        else:
            patience_counter += 1
            print(f"Validation WER did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch+1} epochs! Best WER was {best_val_wer:.4f}.")
                break
    else:
        print("Validation failed to compute WER. Skipping early stopping check for this epoch.")

pd.DataFrame(epoch_log).to_csv(CHECKPOINT_DIR / "training_log.csv", index=False)
print("\nDone. Full training log:", CHECKPOINT_DIR / "training_log.csv")
print("Run a full test-set evaluation separately once you've picked the best epoch.")