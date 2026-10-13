import json
import random
import sys
import re
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import soundfile as sf
import torchaudio
from jiwer import cer, wer
from tqdm.auto import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from torch.optim import AdamW

# ---------------------------------------------------------------------------
# Config & Setup
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
EARLY_STOPPING_PATIENCE = cfg.get("early_stopping_patience", 3)
RESUME_FROM = cfg.get("resume_from_checkpoint")
RESUME_START_EPOCH = int(cfg.get("resume_start_epoch", 0))

TASK_PREFIX = cfg.get("task_prefix", "Transcribe Manipuri speech to Meitei Mayek text:")
USE_CLEANUP = bool(cfg.get("use_decode_cleanup", True))
USE_PHONETIC_POSTPROCESS = bool(cfg.get("use_phonetic_postprocess", False))
PHONETIC_MAP_PATH = cfg.get("phonetic_map_path", "")
VAL_EVAL_SUBSET = int(cfg.get("val_eval_subset_per_epoch", 100))
OVERFIT_TEST_SIZE = int(cfg.get("overfit_test_size", 20))
OVERFIT_TEST_EPOCHS = int(cfg.get("overfit_test_epochs", 3))

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

CHECKPOINT_DIR = ROOT / cfg["checkpoint_dir"]
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RESUME_DIR = CHECKPOINT_DIR / RESUME_FROM if RESUME_FROM else None
if RESUME_DIR and not RESUME_DIR.exists():
    print(f"ERROR: resume_from_checkpoint is set to '{RESUME_FROM}' but {RESUME_DIR} does not exist.")
    sys.exit(1)

print("Device:", DEVICE)
print("Data root:", ROOT)
print("Checkpoint dir:", CHECKPOINT_DIR)
if RESUME_DIR:
    print("Resuming from:", RESUME_DIR)

# ---------------------------------------------------------------------------
# Optional Phonetic Engine
# ---------------------------------------------------------------------------
engine = None
if USE_PHONETIC_POSTPROCESS:
    try:
        from phonetic_engine import ManipuriPhoneticEngine
        map_path = Path(PHONETIC_MAP_PATH) if PHONETIC_MAP_PATH else None
        engine = ManipuriPhoneticEngine(map_path if (map_path and map_path.exists()) else None)
        print("Phonetic postprocess: enabled")
    except Exception as e:
        print(f"WARNING: could not enable phonetic postprocess ({e}). Continuing without it.")
        engine = None
        USE_PHONETIC_POSTPROCESS = False

# ---------------------------------------------------------------------------
# Load Splits & Verify Audio Paths
# ---------------------------------------------------------------------------
train_df = pd.read_csv(ROOT / cfg["train_csv"], encoding="utf-8-sig")
val_df = pd.read_csv(ROOT / cfg["val_csv"], encoding="utf-8-sig")
test_df = pd.read_csv(ROOT / cfg["test_csv"], encoding="utf-8-sig")
print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

def resolve_audio_path(p):
    path_obj = Path(p)
    return path_obj if path_obj.is_absolute() else ROOT / path_obj

train_df = train_df[train_df["audio_path"].apply(lambda p: resolve_audio_path(p).exists())].reset_index(drop=True)
val_df = val_df[val_df["audio_path"].apply(lambda p: resolve_audio_path(p).exists())].reset_index(drop=True)
print(f"after existence re-check: train={len(train_df)}  val={len(val_df)}")

all_text = "".join(train_df["text"].astype(str).tolist())
unique_chars = sorted(set(all_text))
print("Unique characters in train text:", len(unique_chars))

# ---------------------------------------------------------------------------
# Wav2Vec2-BERT 2.0 (Frozen Encoder)
# ---------------------------------------------------------------------------
print("\nLoading Wav2Vec2-BERT 2.0 ...")
feature_extractor = AutoFeatureExtractor.from_pretrained(cfg["wav2vec_model"])
audio_encoder = AutoModel.from_pretrained(cfg["wav2vec_model"]).to(DEVICE)
for p in audio_encoder.parameters():
    p.requires_grad = False
audio_encoder.eval()
hidden_size = audio_encoder.config.hidden_size
print("Wav2Vec2-BERT hidden size:", hidden_size)

# ---------------------------------------------------------------------------
# Audio Projector (1D Subsampling Conv)
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
if RESUME_DIR:
    projector_path = RESUME_DIR / "audio_projector.pt"
    if projector_path.exists():
        projector.load_state_dict(torch.load(projector_path, map_location=DEVICE))
        print("Loaded projector weights from checkpoint.")
print("Trainable projector params:", sum(p.numel() for p in projector.parameters()))

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
tokenizer_path = RESUME_DIR / "tokenizer" if RESUME_DIR else None
if tokenizer_path and tokenizer_path.exists():
    print("\nLoading tokenizer from checkpoint ...")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    print("Vocab size from checkpoint:", len(tokenizer))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
else:
    print("\nLoading Qwen tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["qwen_model"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    missing_chars = [c for c in unique_chars if c.strip() and c not in tokenizer.get_vocab()]
    print("Missing as single tokens:", len(missing_chars), "/", len(unique_chars))

    if cfg.get("add_meitei_chars", False) and missing_chars:
        added = tokenizer.add_tokens(missing_chars)
        print("Added:", added, "-> new vocab size:", len(tokenizer))

im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
print(f"eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})   <|im_end|> id: {im_end_id}")
if tokenizer.eos_token_id != im_end_id:
    print("WARNING: eos_token_id does NOT match <|im_end|>'s id. "
          "generate() calls below stop on BOTH ids -- leaving this warning "
          "so you know why, in case it needs further attention.")
STOP_TOKEN_IDS = list({tid for tid in [tokenizer.eos_token_id, im_end_id] if tid is not None})

# ---------------------------------------------------------------------------
# Pre-flight Filter
# ---------------------------------------------------------------------------
print("\nRunning pre-flight tokenization check on train/val...")

def _token_len(text):
    ids = tokenizer(str(text), add_special_tokens=True, truncation=True, max_length=MAX_TEXT_TOKENS).input_ids
    return len(ids)

def apply_token_filter(df, name):
    lens = df["text"].apply(_token_len)
    valid = lens >= 2
    dropped = int((~valid).sum())
    if dropped:
        print(f"Dropping {dropped} row(s) from {name} (tokenization too short).")
    return df[valid].reset_index(drop=True)

train_df = apply_token_filter(train_df, "train")
val_df = apply_token_filter(val_df, "val")
print(f"after token filter: train={len(train_df)}  val={len(val_df)}")

# ---------------------------------------------------------------------------
# Qwen2.5 (4-bit QLoRA)
# ---------------------------------------------------------------------------
print("\nLoading Qwen2.5-3B-Instruct in 4-bit ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
qwen = AutoModelForCausalLM.from_pretrained(
    cfg["qwen_model"],
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16,
)

qwen.resize_token_embeddings(len(tokenizer))
qwen.config.use_cache = False
qwen = prepare_model_for_kbit_training(qwen)

if cfg.get("gradient_checkpointing", True):
    qwen.gradient_checkpointing_enable()

if RESUME_DIR and (RESUME_DIR / "qwen_lora").exists():
    qwen = PeftModel.from_pretrained(qwen, str(RESUME_DIR / "qwen_lora"), is_trainable=True)
    print("Loaded LoRA adapter from checkpoint.")
else:
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    qwen = get_peft_model(qwen, lora_config)

qwen.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Prompts & Bad Words Setup
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = "You are a helpful speech recognition assistant."
prompt_text = (
    f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
    f"<|im_start|>user\n{TASK_PREFIX}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
TASK_PREFIX_IDS = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)

BAD_STRINGS = [
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>"
]
BAD_WORDS_IDS = [ids for s in BAD_STRINGS if (ids := tokenizer.encode(s, add_special_tokens=False))]

CTRL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")
MULTISPACE_RE = re.compile(r"\s+")

def clean_hyp_text(text: str) -> str:
    if not text:
        return ""
    x = CTRL_TOKEN_RE.sub(" ", text)
    x = x.replace("\ufffd", " ")
    x = MULTISPACE_RE.sub(" ", x).strip()
    return x

def normalize_for_eval(text: str) -> str:
    x = clean_hyp_text(text) if USE_CLEANUP else text
    if USE_PHONETIC_POSTPROCESS and engine is not None:
        try:
            x = engine.clean_sentence(x)
        except Exception:
            pass
    return x.strip()

# ---------------------------------------------------------------------------
# Forward & Inference Processing
# ---------------------------------------------------------------------------
def load_audio_16k(path, max_seconds=MAX_AUDIO_SECONDS):
    resolved = resolve_audio_path(path)
    wav_np, sr = sf.read(str(resolved), dtype="float32", always_2d=False)
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

    target_text = str(row["text"]).strip() + "<|im_end|>"
    # PATCHED: truncation now matches the pre-flight filter's assumption --
    # previously this call had no length cap at all, so an unusually long
    # row could slip past the filter's check and still train untruncated.
    target_ids = tokenizer(
        target_text, add_special_tokens=False, truncation=True,
        max_length=MAX_TEXT_TOKENS, return_tensors="pt"
    ).input_ids.to(DEVICE)

    prefix_emb = qwen.get_input_embeddings()(TASK_PREFIX_IDS)
    target_emb = qwen.get_input_embeddings()(target_ids)

    inputs_embeds = torch.cat([prefix_emb.to(target_emb.dtype), audio_emb.to(target_emb.dtype), target_emb], dim=1)

    ignore_len = prefix_emb.shape[1] + audio_emb.shape[1]
    labels = torch.cat([
        torch.full((1, ignore_len), -100, dtype=torch.long, device=DEVICE),
        target_ids
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

    prefix_emb = qwen.get_input_embeddings()(TASK_PREFIX_IDS)
    prompt_emb = torch.cat([prefix_emb.to(audio_emb.dtype), audio_emb], dim=1)
    mask = torch.ones(prompt_emb.shape[:2], dtype=torch.long, device=DEVICE)

    out_ids = qwen.generate(
        inputs_embeds=prompt_emb,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=STOP_TOKEN_IDS,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        bad_words_ids=BAD_WORDS_IDS if BAD_WORDS_IDS else None,
    )
    raw = tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()
    return raw

def run_step_safely(fn, row, on_oom_msg):
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
# Training Loop
# ---------------------------------------------------------------------------
optimizer = AdamW(
    [
        {"params": projector.parameters(), "lr": cfg["projector_lr"]},
        {"params": [p for p in qwen.parameters() if p.requires_grad], "lr": cfg["qwen_lr"]},
    ],
    weight_decay=0.01,
)
GRAD_ACCUM = cfg["gradient_accumulation_steps"]

print("\n" + "=" * 60)
print("STAGE 1: one-example forward pass")
print("=" * 60)
row0 = train_df.iloc[0]
loss0 = run_step_safely(train_one_row, row0, "sanity check")
if loss0 is None:
    print("FAILED on the very first example. Stopping.")
    sys.exit(1)
print(f"OK -- loss={loss0.item():.4f}")

print("\n" + "=" * 60)
print(f"STAGE 2: {OVERFIT_TEST_SIZE}-example overfit test")
print("=" * 60)

tiny = train_df.head(OVERFIT_TEST_SIZE).copy().reset_index(drop=True)
history = []
optimizer.zero_grad(set_to_none=True)

for epoch in range(OVERFIT_TEST_EPOCHS):
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
    print("\nWARNING: loss did not decrease over the overfit test. The resumed weights")
    print("may be fighting the corrected objective. Recommend investigating before")
    print("committing to a long full run.")
    resp = input("Continue to full training anyway? [y/N] ").strip().lower()
    if resp != "y":
        sys.exit(1)
else:
    print("OK -- loss decreased. Pipeline can learn under the corrected objective.")

optimizer = AdamW(
    [
        {"params": projector.parameters(), "lr": cfg["projector_lr"]},
        {"params": [p for p in qwen.parameters() if p.requires_grad], "lr": cfg["qwen_lr"]},
    ],
    weight_decay=0.01,
)

print("\n" + "=" * 60)
print("STAGE 3: full training")
print("=" * 60)

@torch.no_grad()
def quick_eval(df_subset, n):
    subset = df_subset.head(n)
    refs_raw, hyps_raw = [], []
    refs_norm, hyps_norm = [], []

    for idx, (_, row) in enumerate(tqdm(subset.iterrows(), total=len(subset), desc="Validating")):
        try:
            pred_raw = greedy_transcribe(row)
        except Exception as e:
            print(f"  eval error on {row['audio_path']}: {e}")
            continue

        ref_raw = str(row["text"])
        pred_norm = normalize_for_eval(pred_raw)
        ref_norm = normalize_for_eval(ref_raw)

        refs_raw.append(ref_raw)
        hyps_raw.append(pred_raw)
        refs_norm.append(ref_norm)
        hyps_norm.append(pred_norm)

        if idx < 3:
            print(f"\n  REF(raw): {ref_raw}")
            print(f"  HYP(raw): {pred_raw}")
            print(f"  REF(norm): {ref_norm}")
            print(f"  HYP(norm): {pred_norm}")

    if not refs_raw:
        return float("nan"), float("nan"), float("nan"), float("nan")

    raw_cer = cer(refs_raw, hyps_raw)
    raw_wer = wer(refs_raw, hyps_raw)
    norm_cer = cer(refs_norm, hyps_norm)
    norm_wer = wer(refs_norm, hyps_norm)
    return raw_cer, raw_wer, norm_cer, norm_wer

start_epoch = 0
best_val_wer = float("inf")
patience_counter = 0

if RESUME_DIR:
    state_path = RESUME_DIR / "trainer_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_epoch = int(state.get("epoch", 0))
        best_val_wer = float(state.get("best_val_wer", float("inf")))
        patience_counter = int(state.get("patience_counter", 0))
        print(
            f"Resumed state: start_epoch={start_epoch}  "
            f"best_val_wer={best_val_wer:.4f}  patience={patience_counter}/{EARLY_STOPPING_PATIENCE}"
        )
    else:
        start_epoch = RESUME_START_EPOCH
        print(f"No trainer_state.json found -- using resume_start_epoch from config: {start_epoch}")

    optimizer_path = RESUME_DIR / "optimizer.pt"
    if optimizer_path.exists():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location=DEVICE))
        print("Loaded optimizer state from checkpoint.")

if start_epoch >= cfg["num_epochs"]:
    print(f"\nWARNING: start_epoch ({start_epoch}) >= num_epochs ({cfg['num_epochs']}) in config.json.")
    print("There are no new epochs to run. Increase num_epochs before resuming.")
    sys.exit(1)

optimizer.zero_grad(set_to_none=True)
epoch_log = []

for epoch in range(start_epoch, cfg["num_epochs"]):
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
                list(projector.parameters()) + [p for p in qwen.parameters() if p.requires_grad], 1.0
            )
            if torch.isfinite(grad_norm):
                optimizer.step()
            else:
                print(f"  Skipping optimizer step -- non-finite grad norm ({grad_norm})")
            optimizer.zero_grad(set_to_none=True)

        running += float(loss.detach()) * GRAD_ACCUM
        steps += 1

    avg_loss = running / max(steps, 1)
    raw_cer, raw_wer, val_cer, val_wer = quick_eval(val_df, min(VAL_EVAL_SUBSET, len(val_df)))

    print(
        f"\nEpoch {epoch+1}: train_loss={avg_loss:.4f}  "
        f"raw_CER={raw_cer:.4f} raw_WER={raw_wer:.4f}  "
        f"norm_CER={val_cer:.4f} norm_WER={val_wer:.4f}"
    )

    epoch_log.append({
        "epoch": epoch + 1,
        "train_loss": avg_loss,
        "raw_CER": raw_cer,
        "raw_WER": raw_wer,
        "norm_CER": val_cer,
        "norm_WER": val_wer,
    })

    epoch_dir = CHECKPOINT_DIR / f"epoch_{epoch+1}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    torch.save(projector.state_dict(), epoch_dir / "audio_projector.pt")
    qwen.save_pretrained(str(epoch_dir / "qwen_lora"))
    tokenizer.save_pretrained(str(epoch_dir / "tokenizer"))
    torch.save(optimizer.state_dict(), epoch_dir / "optimizer.pt")

    is_best = (not np.isnan(val_wer)) and (val_wer < best_val_wer)
    if is_best:
        best_val_wer = val_wer
        patience_counter = 0
    else:
        patience_counter += 1

    (epoch_dir / "trainer_state.json").write_text(
        json.dumps(
            {
                "epoch": epoch + 1,
                "best_val_wer": best_val_wer,
                "patience_counter": patience_counter,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if is_best:
        print(f"New best validation norm_WER: {best_val_wer:.4f}! Resetting patience counter.")
        best_dir = CHECKPOINT_DIR / "best_model"
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(epoch_dir, best_dir)
        print(f"Updated best model at: {best_dir}")
    else:
        print(f"Validation WER did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs! Best WER was {best_val_wer:.4f}.")
            break

log_path = CHECKPOINT_DIR / f"training_log_from_epoch_{start_epoch+1}.csv"
pd.DataFrame(epoch_log).to_csv(log_path, index=False)
print("\nDone. Training log:", log_path)