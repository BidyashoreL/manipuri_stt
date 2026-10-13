"""
Manipuri ASR V2 — B0/B1 CTC trainer.

B0:
    cached W2V-BERT features -> Linear CTC

B1:
    cached W2V-BERT features -> residual bottleneck adapter -> CTC

Design:
- Uses existing ASR/combined_dataset/{all,train,val,test}.csv
- Uses existing ASR/feature_cache/<source>/<manifest_index:06d>.pt
- Does NOT touch the Qwen experiment.
- Validation CER is the primary selection metric.
- Early stopping + resumable checkpoints.
- Retains top-N checkpoints.
- Optional greedy checkpoint soup after training.
- Official 410-example test is not used for model selection.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Paths / reproducibility
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
ASR = ROOT / "ASR"

ALL_CSV = ASR / "combined_dataset" / "all.csv"
TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"
TEST_CSV = ASR / "combined_dataset" / "test.csv"
CACHE_ROOT = ASR / "feature_cache"
DEFAULT_RUN_ROOT = ASR / "checkpoints" / "v2_ctc"

SEED = 42
FEATURE_DIM = 1024
CACHE_T = 32


@dataclass
class Config:
    mode: str = "b1"
    epochs: int = 20
    batch_size: int = 4
    grad_accum: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.08
    max_grad_norm: float = 1.0
    patience: int = 4
    min_delta: float = 1e-4
    num_workers: int = 4
    prefetch_factor: int = 2
    amp: bool = True
    adapter_bottleneck: int = 256
    top_k_checkpoints: int = 5
    soup: bool = True
    soup_top_k: int = 5
    log_every: int = 100


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.split())
    return text.strip()


def levenshtein(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                cur[-1] + 1,
                prev[j] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(list(ref), list(hyp)) / len(ref)


def wer(ref: str, hyp: str) -> float:
    r = normalize_text(ref).split()
    h = normalize_text(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(r, h) / len(r)


# ---------------------------------------------------------------------------
# Cache index
# ---------------------------------------------------------------------------

def build_cache_index() -> Dict[Tuple[str, int], Path]:
    index: Dict[Tuple[str, int], Path] = {}
    count = 0

    if not CACHE_ROOT.exists():
        raise FileNotFoundError(f"Feature cache not found: {CACHE_ROOT}")

    for source_dir in CACHE_ROOT.iterdir():
        if not source_dir.is_dir():
            continue
        source = source_dir.name
        for p in source_dir.glob("*.pt"):
            try:
                manifest_index = int(p.stem)
            except ValueError:
                continue
            index[(source, manifest_index)] = p
            count += 1

    print(f"[CACHE] Indexed feature files: {count:,}")
    return index


def attach_cache_indices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    all_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # The current cache was built from all.csv in global row order.
    # We therefore create stable lookup keys and map split rows back to
    # their all.csv manifest index. No audio_path resolution is required.
    lookup: Dict[Tuple, int] = {}

    for i, row in all_df.reset_index(drop=True).iterrows():
        source = str(row.get("source", ""))
        if source == "indicvoices":
            key = (
                "indicvoices",
                str(row.get("parquet_file", "")),
                int(row.get("parquet_row", -1)),
            )
        else:
            key = (
                source,
                str(row.get("audio_path", "")),
            )
        lookup[key] = i

    def map_split(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        indices = []

        for _, row in df.iterrows():
            source = str(row.get("source", ""))
            if source == "indicvoices":
                key = (
                    "indicvoices",
                    str(row.get("parquet_file", "")),
                    int(row.get("parquet_row", -1)),
                )
            else:
                key = (
                    source,
                    str(row.get("audio_path", "")),
                )
            indices.append(lookup.get(key, -1))

        df["_cache_record_index"] = indices
        return df

    return map_split(train_df), map_split(val_df)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SPECIAL_BLANK = "<blank>"
SPECIAL_UNK = "<unk>"


def build_vocab(train_df: pd.DataFrame, out_dir: Path):
    chars = set()

    for text in train_df["text"].astype(str):
        text = normalize_text(text)
        chars.update(text)

    # Never silently discard script characters. Whitespace is represented
    # as the literal space character and is therefore part of the vocabulary.
    chars = sorted(c for c in chars if c != "\x00")

    vocab = {
        SPECIAL_BLANK: 0,
        SPECIAL_UNK: 1,
    }

    for ch in chars:
        if ch not in vocab:
            vocab[ch] = len(vocab)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[VOCAB] Size: {len(vocab)}")
    print(f"[VOCAB] Train characters: {len(chars)}")
    return vocab


def load_vocab(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CachedCTCDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cache_index, vocab):
        self.df = df.reset_index(drop=True)
        self.cache_index = cache_index
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def encode_text(self, text: str):
        ids = []
        for ch in normalize_text(text):
            ids.append(self.vocab.get(ch, self.vocab[SPECIAL_UNK]))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        source = str(row["source"])
        manifest_index = int(row["_cache_record_index"])
        cache_path = self.cache_index.get((source, manifest_index))

        if cache_path is None:
            raise RuntimeError(
                f"Missing cache feature for row={idx}, "
                f"source={source}, manifest_index={manifest_index}"
            )

        payload = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=True,
        )

        if isinstance(payload, dict):
            feat = payload["features"]
        else:
            feat = payload

        if not isinstance(feat, torch.Tensor):
            raise TypeError(f"Invalid feature payload: {cache_path}")

        feat = feat.float().contiguous()

        if feat.ndim != 2 or feat.shape[-1] != FEATURE_DIM:
            raise RuntimeError(
                f"Expected [T,{FEATURE_DIM}], got {tuple(feat.shape)} "
                f"from {cache_path}"
            )

        target = self.encode_text(str(row["text"]))

        return {
            "features": feat,
            "feature_length": feat.shape[0],
            "target": target,
            "text": normalize_text(str(row["text"])),
        }


def collate_fn(batch):
    bsz = len(batch)
    max_t = max(x["feature_length"] for x in batch)
    feat = torch.zeros(bsz, max_t, FEATURE_DIM, dtype=torch.float32)

    input_lengths = []
    targets = []
    target_lengths = []
    texts = []

    for i, item in enumerate(batch):
        t = item["feature_length"]
        feat[i, :t] = item["features"]
        input_lengths.append(t)

        targets.append(item["target"])
        target_lengths.append(int(item["target"].numel()))
        texts.append(item["text"])

    targets = torch.cat(targets, dim=0) if targets else torch.empty(0, dtype=torch.long)

    return {
        "features": feat,
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "targets": targets,
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "texts": texts,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ResidualBottleneckAdapter(nn.Module):
    """
    B1 adapter.

    Frozen W2V-BERT features
        -> LayerNorm
        -> Linear 1024 -> bottleneck
        -> GELU
        -> Linear bottleneck -> 1024
        -> residual
        -> LayerNorm
    """

    def __init__(self, dim=1024, bottleneck=256):
        super().__init__()
        self.norm_in = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.norm_out = nn.LayerNorm(dim)

        # Start close to identity so B1 is stable at initialization.
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = self.norm_in(x)
        z = self.up(self.act(self.down(z)))
        return self.norm_out(x + z)


class CTCASR(nn.Module):
    def __init__(self, vocab_size: int, mode: str, bottleneck: int):
        super().__init__()
        self.mode = mode
        self.vocab_size = vocab_size

        if mode == "b0":
            self.adapter = nn.Identity()
        elif mode == "b1":
            self.adapter = ResidualBottleneckAdapter(
                FEATURE_DIM,
                bottleneck,
            )
        else:
            raise ValueError("mode must be b0 or b1")

        self.dropout = nn.Dropout(0.10)
        self.ctc_head = nn.Linear(FEATURE_DIM, vocab_size)

    def forward(self, features):
        x = self.adapter(features)
        x = self.dropout(x)
        return self.ctc_head(x)


# ---------------------------------------------------------------------------
# Decoding / metrics
# ---------------------------------------------------------------------------

def decode_ids(ids, vocab):
    inv = {int(v): k for k, v in vocab.items()}
    blank = vocab[SPECIAL_BLANK]
    unk = vocab[SPECIAL_UNK]

    out = []
    prev = None

    for token in ids:
        token = int(token)

        if token == blank:
            prev = None
            continue

        if token == prev:
            continue

        ch = inv.get(token, SPECIAL_UNK)
        if ch == SPECIAL_UNK:
            ch = "�"

        out.append(ch)
        prev = token

    return "".join(out)


@torch.no_grad()
def evaluate(model, loader, vocab, device, amp_enabled, show_examples=5):
    model.eval()

    total_loss = 0.0
    total_items = 0
    refs = []
    hyps = []

    ctc_loss_fn = nn.CTCLoss(
        blank=vocab[SPECIAL_BLANK],
        zero_infinity=True,
    )

    for batch_idx, batch in enumerate(loader):
        features = batch["features"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        use_amp = amp_enabled and device.type == "cuda"

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(features)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = ctc_loss_fn(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

        total_loss += float(loss.item()) * len(batch["texts"])
        total_items += len(batch["texts"])

        pred = logits.argmax(dim=-1).detach().cpu().tolist()

        for ref, hyp_ids in zip(batch["texts"], pred):
            hyp = decode_ids(hyp_ids, vocab)
            refs.append(ref)
            hyps.append(hyp)

    avg_loss = total_loss / max(total_items, 1)

    char_errors = 0
    char_total = 0
    word_errors = 0
    word_total = 0

    for ref, hyp in zip(refs, hyps):
        rchars = list(normalize_text(ref))
        hchars = list(normalize_text(hyp))
        char_errors += levenshtein(rchars, hchars)
        char_total += max(len(rchars), 1)

        rwords = normalize_text(ref).split()
        hwords = normalize_text(hyp).split()
        word_errors += levenshtein(rwords, hwords)
        word_total += max(len(rwords), 1)

    val_cer = char_errors / max(char_total, 1)
    val_wer = word_errors / max(word_total, 1)

    print(
        f"[VAL] loss={avg_loss:.4f} "
        f"CER={val_cer:.4f} "
        f"WER={val_wer:.4f}"
    )

    if show_examples:
        print("[VAL] Examples:")
        for i in range(min(show_examples, len(refs))):
            print(f"  REF: {refs[i]}")
            print(f"  HYP: {hyps[i]}")

    return {
        "loss": avg_loss,
        "cer": val_cer,
        "wer": val_wer,
        "refs": refs,
        "hyps": hyps,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def atomic_torch_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(
    run_dir,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_cer,
    patience_count,
    metrics,
    config,
):
    ckpt_dir = run_dir / f"epoch_{epoch:03d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    atomic_torch_save(model.state_dict(), ckpt_dir / "model.pt")
    atomic_torch_save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
    atomic_torch_save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

    if scaler is not None:
        atomic_torch_save(scaler.state_dict(), ckpt_dir / "scaler.pt")

    state = {
        "epoch": epoch,
        "best_cer": best_cer,
        "patience_count": patience_count,
        "metrics": metrics,
        "config": asdict(config),
    }

    (ckpt_dir / "state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ckpt_dir


def load_checkpoint(
    ckpt_dir,
    model,
    optimizer,
    scheduler,
    scaler,
    device,
):
    ckpt_dir = Path(ckpt_dir)

    model.load_state_dict(
        torch.load(ckpt_dir / "model.pt", map_location=device, weights_only=True)
    )
    optimizer.load_state_dict(
        torch.load(ckpt_dir / "optimizer.pt", map_location=device, weights_only=True)
    )
    scheduler.load_state_dict(
        torch.load(ckpt_dir / "scheduler.pt", map_location=device, weights_only=True)
    )

    scaler_path = ckpt_dir / "scaler.pt"
    if scaler is not None and scaler_path.exists():
        scaler.load_state_dict(
            torch.load(scaler_path, map_location=device, weights_only=True)
        )

    state = json.loads((ckpt_dir / "state.json").read_text(encoding="utf-8"))
    return state


def save_best(run_dir, ckpt_dir):
    best = run_dir / "best"
    if best.exists():
        shutil.rmtree(best)
    shutil.copytree(ckpt_dir, best)


def save_last(run_dir, ckpt_dir):
    last = run_dir / "last"
    if last.exists():
        shutil.rmtree(last)
    shutil.copytree(ckpt_dir, last)


def prune_checkpoints(run_dir, keep_top_k):
    entries = []

    for p in run_dir.glob("epoch_*"):
        state_path = p / "state.json"
        if not state_path.exists():
            continue

        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            entries.append((float(state["metrics"]["cer"]), p))
        except Exception:
            pass

    entries.sort(key=lambda x: x[0])

    keep = {p for _, p in entries[:keep_top_k]}

    for _, p in entries[keep_top_k:]:
        if p not in keep:
            shutil.rmtree(p, ignore_errors=True)

    return [p for _, p in entries[:keep_top_k]]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = (step - warmup_steps) / max(
            1,
            total_steps - warmup_steps,
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Soup
# ---------------------------------------------------------------------------

def load_model_weights(path: Path, device):
    return torch.load(
        path,
        map_location=device,
        weights_only=True,
    )


def average_state_dicts(states):
    keys = states[0].keys()
    out = {}

    for k in keys:
        tensors = [s[k] for s in states]

        if not torch.is_floating_point(tensors[0]):
            out[k] = tensors[0].clone()
            continue

        out[k] = sum(tensors) / len(tensors)

    return out


@torch.no_grad()
def greedy_soup(
    model,
    candidate_dirs,
    val_loader,
    vocab,
    device,
    amp_enabled,
    run_dir,
):
    if len(candidate_dirs) < 2:
        print("[SOUP] Fewer than 2 candidates; skipping.")
        return None

    # Candidate order is already validation-CER sorted.
    candidate_dirs = list(candidate_dirs)

    best_state = load_model_weights(
        candidate_dirs[0] / "model.pt",
        device,
    )

    model.load_state_dict(best_state)
    base = evaluate(
        model,
        val_loader,
        vocab,
        device,
        amp_enabled,
        show_examples=0,
    )

    accepted_states = [best_state]
    soup_state = {
        k: v.clone() if torch.is_tensor(v) else v
        for k, v in best_state.items()
    }
    count = 1
    current_cer = base["cer"]
    accepted = [candidate_dirs[0].name]

    print(f"[SOUP] Starting CER={current_cer:.4f}")

    for candidate_dir in candidate_dirs[1:]:
        candidate_state = load_model_weights(
            candidate_dir / "model.pt",
            device,
        )

        trial_states = accepted_states + [candidate_state]
        trial = average_state_dicts(trial_states)

        model.load_state_dict(trial)

        metrics = evaluate(
            model,
            val_loader,
            vocab,
            device,
            amp_enabled,
            show_examples=0,
        )

        print(
            f"[SOUP] Try {candidate_dir.name}: "
            f"CER={metrics['cer']:.4f} vs current={current_cer:.4f}"
        )

        if metrics["cer"] + 1e-8 < current_cer:
            soup_state = trial
            accepted_states.append(candidate_state)
            current_cer = metrics["cer"]
            count += 1
            accepted.append(candidate_dir.name)
            print("[SOUP] ACCEPT")
        else:
            model.load_state_dict(soup_state)
            print("[SOUP] REJECT")

    soup_dir = run_dir / "soup"
    soup_dir.mkdir(parents=True, exist_ok=True)

    atomic_torch_save(soup_state, soup_dir / "model.pt")

    soup_info = {
        "accepted_checkpoints": accepted,
        "num_models": count,
        "validation_cer": current_cer,
        "validation_wer": None,
    }

    model.load_state_dict(soup_state)
    final_metrics = evaluate(
        model,
        val_loader,
        vocab,
        device,
        amp_enabled,
        show_examples=5,
    )
    soup_info["validation_wer"] = final_metrics["wer"]

    (soup_dir / "soup.json").write_text(
        json.dumps(soup_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"[SOUP] Final soup: {count} checkpoints, "
        f"CER={final_metrics['cer']:.4f}, "
        f"WER={final_metrics['wer']:.4f}"
    )

    return soup_dir


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    scaler,
    ctc_loss_fn,
    device,
    cfg,
    epoch,
):
    model.train()

    optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    count = 0
    start = time.time()

    for step, batch in enumerate(loader, 1):
        features = batch["features"].to(device, non_blocking=True)
        input_lengths = batch["input_lengths"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        target_lengths = batch["target_lengths"].to(device, non_blocking=True)

        use_amp = cfg.amp and device.type == "cuda"

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            logits = model(features)
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)

            loss = ctc_loss_fn(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

            scaled_loss = loss / cfg.grad_accum

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        should_step = (
            step % cfg.grad_accum == 0
            or step == len(loader)
        )

        if should_step:
            if scaler is not None:
                scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                cfg.max_grad_norm,
            )

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        running_loss += float(loss.item()) * len(batch["texts"])
        count += len(batch["texts"])

        if step % cfg.log_every == 0:
            mem = 0.0
            if device.type == "cuda":
                mem = torch.cuda.memory_allocated(device) / (1024 ** 3)

            print(
                f"[TRAIN] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"VRAM={mem:.2f}GB"
            )

    elapsed = time.time() - start
    return running_loss / max(count, 1), elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["b0", "b1"], default="b1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-soup", action="store_true")
    parser.add_argument("--soup-top-k", type=int, default=5)
    args = parser.parse_args()

    seed_everything()

    cfg = Config(
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        patience=args.patience,
        num_workers=args.workers,
        soup=not args.no_soup,
        soup_top_k=args.soup_top_k,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("MANIPURI ASR V2 — CTC")
    print("=" * 72)
    print(f"[DEVICE] {device}")

    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(
            f"[GPU] VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"[MODE] {cfg.mode}")
    print("[DATA] Using cached W2V-BERT features; audio paths are not needed for B0/B1.")

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------

    for p in [ALL_CSV, TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not p.exists():
            raise FileNotFoundError(p)

    all_df = pd.read_csv(ALL_CSV, encoding="utf-8-sig")
    train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    val_df = pd.read_csv(VAL_CSV, encoding="utf-8-sig")
    test_df = pd.read_csv(TEST_CSV, encoding="utf-8-sig")

    print(
        f"[DATA] train={len(train_df):,} "
        f"val={len(val_df):,} "
        f"test={len(test_df):,}"
    )

    train_df, val_df = attach_cache_indices(
        train_df,
        val_df,
        all_df,
    )

    cache_index = build_cache_index()

    for name, df in [("train", train_df), ("val", val_df)]:
        missing = 0
        for _, row in df.iterrows():
            key = (
                str(row["source"]),
                int(row["_cache_record_index"]),
            )
            if int(row["_cache_record_index"]) < 0 or key not in cache_index:
                missing += 1

        if missing:
            raise RuntimeError(
                f"{name}: {missing} rows have no cached feature. "
                "Fix cache mapping before training."
            )

    # -----------------------------------------------------------------------
    # Run directory
    # -----------------------------------------------------------------------

    run_name = args.run_name or f"{cfg.mode}_ctc"
    run_dir = DEFAULT_RUN_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(cfg),
                "root": str(ROOT),
                "all_csv": str(ALL_CSV),
                "train_csv": str(TRAIN_CSV),
                "val_csv": str(VAL_CSV),
                "test_csv": str(TEST_CSV),
                "cache_root": str(CACHE_ROOT),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Vocabulary
    # -----------------------------------------------------------------------

    vocab_path = run_dir / "vocab.json"

    if vocab_path.exists():
        vocab = load_vocab(vocab_path)
        print(f"[VOCAB] Loaded existing vocabulary: {len(vocab)}")
    else:
        vocab = build_vocab(train_df, run_dir)

    # -----------------------------------------------------------------------
    # Dataset / loaders
    # -----------------------------------------------------------------------

    train_ds = CachedCTCDataset(train_df, cache_index, vocab)
    val_ds = CachedCTCDataset(val_df, cache_index, vocab)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }

    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = cfg.prefetch_factor

    train_loader = DataLoader(
        train_ds,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    print(
        f"[LOADER] batch={cfg.batch_size}, "
        f"grad_accum={cfg.grad_accum}, "
        f"effective_batch={cfg.batch_size * cfg.grad_accum}, "
        f"workers={cfg.num_workers}"
    )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    model = CTCASR(
        vocab_size=len(vocab),
        mode=cfg.mode,
        bottleneck=cfg.adapter_bottleneck,
    ).to(device)

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())

    print(f"[MODEL] total params: {total:,}")
    print(f"[MODEL] trainable params: {trainable:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / cfg.grad_accum
    )
    total_steps = updates_per_epoch * cfg.epochs

    scheduler = build_scheduler(
        optimizer,
        total_steps,
        cfg.warmup_ratio,
    )

    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if device.type == "cuda" and cfg.amp
        else None
    )

    ctc_loss_fn = nn.CTCLoss(
        blank=vocab[SPECIAL_BLANK],
        zero_infinity=True,
    )

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------

    start_epoch = 1
    best_cer = float("inf")
    patience_count = 0

    if args.resume:
        resume_dir = Path(args.resume)
        state = load_checkpoint(
            resume_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )

        start_epoch = int(state["epoch"]) + 1
        best_cer = float(state["best_cer"])
        patience_count = int(state["patience_count"])

        print(
            f"[RESUME] epoch={state['epoch']} "
            f"best_CER={best_cer:.4f} "
            f"patience={patience_count}"
        )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    history_path = run_dir / "history.jsonl"

    for epoch in range(start_epoch, cfg.epochs + 1):
        print()
        print("=" * 72)
        print(f"EPOCH {epoch}/{cfg.epochs}")
        print("=" * 72)

        train_loss, elapsed = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            ctc_loss_fn,
            device,
            cfg,
            epoch,
        )

        metrics = evaluate(
            model,
            val_loader,
            vocab,
            device,
            cfg.amp,
            show_examples=5,
        )

        metrics_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "cer": metrics["cer"],
            "wer": metrics["wer"],
            "epoch_seconds": elapsed,
            "lr": optimizer.param_groups[0]["lr"],
        }

        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics_record, ensure_ascii=False) + "\n")

        improved = metrics["cer"] < (best_cer - cfg.min_delta)

        if improved:
            best_cer = metrics["cer"]
            patience_count = 0
            print(f"[EARLY STOP] New best CER: {best_cer:.4f}")
        else:
            patience_count += 1
            print(
                f"[EARLY STOP] No improvement. "
                f"patience={patience_count}/{cfg.patience}"
            )

        ckpt_dir = save_checkpoint(
            run_dir,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_cer,
            patience_count,
            metrics_record,
            cfg,
        )

        save_last(run_dir, ckpt_dir)

        if improved:
            save_best(run_dir, ckpt_dir)

        candidates = prune_checkpoints(
            run_dir,
            cfg.top_k_checkpoints,
        )

        print(
            "[CHECKPOINT] "
            f"saved={ckpt_dir.name} "
            f"best={'YES' if improved else 'NO'}"
        )

        if patience_count >= cfg.patience:
            print("[EARLY STOP] Stopping training.")
            break

    # -----------------------------------------------------------------------
    # Final validation selection
    # -----------------------------------------------------------------------

    best_dir = run_dir / "best"

    if not best_dir.exists():
        raise RuntimeError("No best checkpoint was produced.")

    model.load_state_dict(
        load_model_weights(best_dir / "model.pt", device)
    )

    best_metrics = evaluate(
        model,
        val_loader,
        vocab,
        device,
        cfg.amp,
        show_examples=10,
    )

    print()
    print("=" * 72)
    print("BEST CHECKPOINT")
    print("=" * 72)
    print(f"CER: {best_metrics['cer']:.4f}")
    print(f"WER: {best_metrics['wer']:.4f}")
    print(f"Path: {best_dir}")

    # -----------------------------------------------------------------------
    # Greedy soup
    # -----------------------------------------------------------------------

    if cfg.soup:
        candidate_dirs = []
        for p in run_dir.glob("epoch_*"):
            if (p / "model.pt").exists() and (p / "state.json").exists():
                candidate_dirs.append(p)

        def checkpoint_cer(p):
            state = json.loads((p / "state.json").read_text(encoding="utf-8"))
            return float(state["metrics"]["cer"])

        candidate_dirs.sort(key=checkpoint_cer)
        candidate_dirs = candidate_dirs[: cfg.soup_top_k]

        print()
        print("[SOUP] Candidates:")
        for p in candidate_dirs:
            print(
                f"  {p.name}: "
                f"CER={checkpoint_cer(p):.4f}"
            )

        greedy_soup(
            model,
            candidate_dirs,
            val_loader,
            vocab,
            device,
            cfg.amp,
            run_dir,
        )

    # -----------------------------------------------------------------------
    # Final status
    # -----------------------------------------------------------------------

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Run directory: {run_dir}")
    print(f"Best checkpoint: {run_dir / 'best'}")
    print(f"Last checkpoint: {run_dir / 'last'}")
    if cfg.soup:
        print(f"Soup checkpoint: {run_dir / 'soup' / 'model.pt'}")
    print()
    print("IMPORTANT: The official test split was NOT used for selection.")


if __name__ == "__main__":
    main()
