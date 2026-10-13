from pathlib import Path
import os, random, json, time, io, wave, gc, re, unicodedata
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
from pyctcdecode import build_ctcdecoder

ROOT = Path.cwd()
ASR = ROOT.parent

TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"
ENCODER_MODEL = Path(r"E:\hf_models\w2v-bert-2.0")
B2_CHECKPOINT = ASR / "checkpoints" / "v2_ctc" / "b2_partial_encoder" / "regularized" / "FINAL_B2_MODEL.pt"
VOCAB_PATH = ASR / "checkpoints" / "v2_ctc" / "b2_partial_encoder" / "vocab.json"
UNIGRAM_PATH = ROOT / "kenlm_experiment" / "manipuri_unigrams.txt"
SAVE_DIR = ASR / "checkpoints" / "v3_ctc" / "b3_narrow"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAMPLE_RATE = 16000
FEATURE_DIM = 1024
BOTTLENECK = 256
SEED = 42
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR_ENCODER = 5e-6
LR_HEAD = 2e-4
WEIGHT_DECAY = 0.02
MAX_GRAD_NORM = 1.0
EPOCHS = 12
EXPECTED_RESUME_EPOCH = 8
NUM_WORKERS = 0
AMP = True
TRAINABLE_LAYERS = [22, 23]
BEAM_WIDTH = 500
PARQUET_CACHE_SIZE = 4
SPECIAL_BLANK = "<blank>"
SPECIAL_UNK = "<unk>"

USE_BF16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
AMP_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32

def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_text(text):
    if text is None: return ""
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()

def decode_pcm(raw, sample_width):
    if sample_width == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        return (audio - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_width == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        if len(b) % 3: raise ValueError("Invalid 24-bit PCM byte length")
        b = b.reshape(-1, 3)
        audio = b[:,0].astype(np.int32) | (b[:,1].astype(np.int32)<<8) | (b[:,2].astype(np.int32)<<16)
        mask = audio & 0x800000
        audio = audio - (mask << 1)
        return audio.astype(np.float32) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise ValueError(f"Unsupported PCM sample width: {sample_width}")

@lru_cache(maxsize=PARQUET_CACHE_SIZE)
def _read_parquet_cached(parquet_file):
    return pd.read_parquet(str(parquet_file), columns=["audio"])

def load_parquet_audio(parquet_file, row_index):
    parquet_file = str(parquet_file)
    if not os.path.exists(parquet_file): raise FileNotFoundError(parquet_file)
    df = _read_parquet_cached(parquet_file)
    obj = df.iloc[int(row_index)]["audio"]
    if not isinstance(obj, dict): raise ValueError(f"Unexpected audio object: {type(obj)}")
    raw_bytes = obj.get("bytes")
    if raw_bytes is None: raise ValueError("IndicVoices audio object has no bytes")
    with wave.open(io.BytesIO(raw_bytes), "rb") as wf:
        channels, width, sr, frames = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        raw = wf.readframes(frames)
    audio = decode_pcm(raw, width)
    if channels > 1: audio = audio.reshape(-1, channels).mean(axis=1)
    return audio, sr

def normalize_audio(audio, sample_rate):
    audio = torch.as_tensor(audio, dtype=torch.float32)
    if audio.ndim == 2:
        audio = audio.mean(dim=0 if audio.shape[0] <= 8 else 1)
    elif audio.ndim > 2: audio = audio.squeeze()
    if sample_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, SAMPLE_RATE)
    return audio.contiguous()

def resolve_audio(row):
    source = str(row.get("source", "")).strip().lower()
    if source == "indicvoices":
        a, sr = load_parquet_audio(row["parquet_file"], int(row["parquet_row"]))
        return normalize_audio(a, sr)
    path = row.get("audio_path", "")
    if pd.isna(path) or not str(path).strip(): raise RuntimeError(f"Invalid audio_path for source={source}")
    path = str(path).strip()
    if not os.path.exists(path): raise FileNotFoundError(path)
    array, sr = sf.read(path, dtype="float32", always_2d=False)
    return normalize_audio(torch.from_numpy(np.asarray(array)), sr)

def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f: vocab = json.load(f)
    if len(vocab) != 60 or vocab.get(SPECIAL_BLANK) != 0:
        raise RuntimeError(f"Unexpected vocab: size={len(vocab)}, blank={vocab.get(SPECIAL_BLANK)}")
    return vocab

def encode_text(text, vocab):
    return torch.tensor([vocab.get(ch, vocab[SPECIAL_UNK]) for ch in normalize_text(text)], dtype=torch.long)

class AudioCTCDataset(Dataset):
    def __init__(self, df, vocab): self.df, self.vocab = df.reset_index(drop=True), vocab
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {"audio": resolve_audio(row), "target": encode_text(row["text"], self.vocab), "text": normalize_text(row["text"])}

def collate_fn(batch):
    max_audio = max(x["audio"].shape[0] for x in batch)
    audios = torch.zeros(len(batch), max_audio, dtype=torch.float32)
    lengths, targets, target_lengths, texts = [], [], [], []
    for i, x in enumerate(batch):
        n = x["audio"].shape[0]; audios[i,:n] = x["audio"]; lengths.append(n)
        targets.append(x["target"]); target_lengths.append(x["target"].shape[0]); texts.append(x["text"])
    return {"audio": audios, "audio_lengths": torch.tensor(lengths), "targets": torch.cat(targets), "target_lengths": torch.tensor(target_lengths), "texts": texts}

class ResidualBottleneckAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck=256):
        super().__init__()
        self.norm_in=nn.LayerNorm(dim); self.down=nn.Linear(dim,bottleneck); self.act=nn.GELU()
        self.up=nn.Linear(bottleneck,dim); self.norm_out=nn.LayerNorm(dim)
        nn.init.normal_(self.up.weight, mean=0.0, std=1e-3); nn.init.zeros_(self.up.bias)
    def forward(self,x):
        z=self.up(self.act(self.down(self.norm_in(x))))
        return self.norm_out(x+z)

class B2Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.processor=AutoFeatureExtractor.from_pretrained(ENCODER_MODEL, local_files_only=True)
        self.encoder=Wav2Vec2BertModel.from_pretrained(ENCODER_MODEL, dtype=torch.float32, local_files_only=True)
        self.encoder.config.layerdrop=0.0
        for p in self.encoder.parameters(): p.requires_grad=False
        self.adapter=ResidualBottleneckAdapter(FEATURE_DIM,BOTTLENECK)
        self.ctc_head=nn.Linear(FEATURE_DIM,vocab_size)
        self.encoder.to(DEVICE); self.adapter.to(DEVICE); self.ctc_head.to(DEVICE)
        try: self.encoder.gradient_checkpointing_disable()
        except Exception: pass
    def forward(self,input_features):
        hidden=self.encoder(input_features=input_features).last_hidden_state
        return self.ctc_head(self.adapter(hidden))

def load_b2_checkpoint(model):
    ckpt = torch.load(B2_CHECKPOINT, map_location="cpu", weights_only=True)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
    missing,unexpected=model.load_state_dict(state,strict=False)
    if missing: raise RuntimeError("Missing B2 checkpoint keys:\n"+ "\n".join(map(str,missing)))
    if unexpected: print("[WARNING] Unexpected keys:", unexpected)
    print("[INIT] FINAL_B2_MODEL.pt: PASS")

def configure_b3(model):
    for p in model.parameters(): p.requires_grad=False
    for i in TRAINABLE_LAYERS:
        for p in model.encoder.encoder.layers[i].parameters(): p.requires_grad=True
    for p in model.adapter.parameters(): p.requires_grad=True
    for p in model.ctc_head.parameters(): p.requires_grad=True

def print_report(model):
    total=sum(p.numel() for p in model.parameters()); train=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total:,}")
    print(f"Trainable parameters : {train:,}")
    print(f"Trainable ratio      : {100*train/total:.4f}%")
    for i in range(24):
        n=sum(p.numel() for p in model.encoder.encoder.layers[i].parameters() if p.requires_grad)
        print(f"Layer {i:02d} : {'TRAINABLE' if n else 'FROZEN':<10} {n:,}")
    print(f"Adapter trainable    : {sum(p.numel() for p in model.adapter.parameters() if p.requires_grad):,}")
    print(f"CTC head trainable   : {sum(p.numel() for p in model.ctc_head.parameters() if p.requires_grad):,}")

def make_optimizer(model):
    enc=[]
    for i in TRAINABLE_LAYERS: enc += list(model.encoder.encoder.layers[i].parameters())
    head=list(model.adapter.parameters())+list(model.ctc_head.parameters())
    return torch.optim.AdamW([{"params":enc,"lr":LR_ENCODER},{"params":head,"lr":LR_HEAD}],weight_decay=WEIGHT_DECAY,betas=(0.9,0.98))

def make_features(processor,audios):
    audio_list=[w.cpu().numpy().astype(np.float32,copy=False) for w in audios]
    out=processor(audio_list,sampling_rate=SAMPLE_RATE,return_tensors="pt",padding=True)
    features=out.input_features
    del audio_list,out
    return features

def decode_ids(ids, inv_vocab, blank_id=0):
    prev = None
    out = []
    for t in ids:
        t = int(t)
        if t == blank_id:
            prev = None
            continue
        if t == prev:
            continue
        out.append(inv_vocab.get(t, SPECIAL_UNK))
        prev = t
    return "".join(out)

def levenshtein(a,b):
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]

def metrics(refs,hyps):
    ce=ct=we=wt=0
    for r,h in zip(refs,hyps):
        r=normalize_text(r); h=normalize_text(h)
        rc,hc=list(r),list(h); rw,hw=r.split(),h.split()
        ce+=levenshtein(rc,hc); ct+=max(len(rc),1)
        we+=levenshtein(rw,hw); wt+=max(len(rw),1)
    return ce/max(ct,1),we/max(wt,1)

def build_decoder(vocab):
    labels=[k for k,v in sorted(vocab.items(),key=lambda x:x[1])]
    labels=["" if x==SPECIAL_BLANK else x for x in labels]
    with open(UNIGRAM_PATH,"r",encoding="utf-8") as f: unigrams=[x.strip() for x in f if x.strip()]
    print(f"[DECODER] labels={len(labels)}, unigrams={len(unigrams):,}, beam={BEAM_WIDTH}")
    return build_ctcdecoder(labels=labels,unigrams=unigrams)

def set_train_modes(model):
    model.eval()
    for i in TRAINABLE_LAYERS: model.encoder.encoder.layers[i].train()
    model.adapter.train(); model.ctc_head.train()

def train_epoch(model,loader,optimizer,scaler,ctc,epoch):
    set_train_modes(model); optimizer.zero_grad(set_to_none=True)
    total=0.; items=0; steps=0; start=time.time()
    for step,batch in enumerate(loader,1):
        feats=make_features(model.processor,batch["audio"]).to(DEVICE)
        targets=batch["targets"].to(DEVICE); tl=batch["target_lengths"].to(DEVICE)
        enabled=AMP and DEVICE.type=="cuda" and USE_BF16
        with torch.autocast(device_type="cuda",dtype=AMP_DTYPE,enabled=enabled):
            logits=model(feats)
            il=torch.full((logits.shape[0],),logits.shape[1],dtype=torch.long,device=DEVICE)
            lp=logits.float().log_softmax(-1).transpose(0,1)
            loss=ctc(lp,targets,il,tl)
            if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite loss: {loss.item()}")
            remainder = len(loader) % GRAD_ACCUM
            is_final_partial = remainder != 0 and step > len(loader) - remainder
            accum_steps = remainder if is_final_partial else GRAD_ACCUM
            sl = loss / accum_steps
        sl.backward()
        if step%GRAD_ACCUM==0 or step==len(loader):
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM)
            if not torch.isfinite(torch.as_tensor(gn)): raise FloatingPointError("Non-finite gradient norm")
            optimizer.step(); optimizer.zero_grad(set_to_none=True); steps+=1
        total+=loss.item()*len(batch["texts"]); items+=len(batch["texts"])
        if step%100==0:
            mem=torch.cuda.memory_allocated(DEVICE)/1024**3 if DEVICE.type=="cuda" else 0
            print(f"[TRAIN] epoch={epoch} step={step}/{len(loader)} loss={loss.item():.4f} opt={steps} VRAM={mem:.2f}GB",flush=True)
    return total/max(items,1),(time.time()-start)/60,steps

@torch.no_grad()
def validate(model,loader,vocab,decoder,inv_vocab):
    model.eval(); refs=[]; greedy=[]; beam=[]; total=0.; items=0
    ctc=nn.CTCLoss(blank=vocab[SPECIAL_BLANK],zero_infinity=True)
    for bi,batch in enumerate(loader,1):
        feats=make_features(model.processor,batch["audio"]).to(DEVICE)
        targets=batch["targets"].to(DEVICE); tl=batch["target_lengths"].to(DEVICE)
        enabled=AMP and DEVICE.type=="cuda" and USE_BF16
        with torch.autocast(device_type="cuda",dtype=AMP_DTYPE,enabled=enabled):
            logits=model(feats)
            il=torch.full((logits.shape[0],),logits.shape[1],dtype=torch.long,device=DEVICE)
            lp=logits.float().log_softmax(-1).transpose(0,1)
            loss=ctc(lp,targets,il,tl)
        if not torch.isfinite(loss): raise FloatingPointError(f"Non-finite validation loss: {loss.item()}")
        total+=loss.item()*len(batch["texts"]); items+=len(batch["texts"])
        ids=logits.argmax(-1).cpu().tolist()
        lpn=lp.transpose(0,1).cpu().numpy()
        for i,r in enumerate(batch["texts"]):
            refs.append(normalize_text(r))
            greedy.append(normalize_text(decode_ids(ids[i], inv_vocab, vocab[SPECIAL_BLANK])))
            beam.append(normalize_text(decoder.decode(lpn[i],beam_width=BEAM_WIDTH)))
        if bi%100==0: print(f"[VAL] {bi}/{len(loader)}",flush=True)
    gc.collect()
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    greedy_cer, greedy_wer = metrics(refs, greedy)
    beam_cer, beam_wer = metrics(refs, beam)
    return {
        "loss": total / max(items, 1),
        "greedy_cer": greedy_cer,
        "greedy_wer": greedy_wer,
        "beam_cer": beam_cer,
        "beam_wer": beam_wer
    }

def make_payload(model,epoch,m,train_loss):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(m["loss"]),
        "val_greedy_cer": float(m["greedy_cer"]),
        "val_greedy_wer": float(m["greedy_wer"]),
        "val_beam500_cer": float(m["beam_cer"]),
        "val_beam500_wer": float(m["beam_wer"]),
        "trainable_layers": TRAINABLE_LAYERS,
        "beam_width": BEAM_WIDTH,
        "normalization": "NFC + whitespace collapse",
    }


def disk_free_gb(path=SAVE_DIR):
    import shutil
    return shutil.disk_usage(str(path)).free / (1024**3)

def atomic_torch_save(payload, path):
    """
    Save safely using a temporary file and atomic replacement.
    Requires enough free disk for one additional full checkpoint.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        try:
            tmp.unlink()
        except Exception:
            pass

    print(f"[SAVE] free disk before save: {disk_free_gb():.2f} GB")
    torch.save(payload, tmp)
    if path.exists():
        path.unlink()
    tmp.replace(path)

def make_hardlink(src, dst):
    """
    Create a zero-additional-space Windows hardlink when possible.
    This is critical because each model checkpoint is ~2.3 GB.
    """
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        print(f"[HARDLINK] {dst.name} -> {src.name} (no extra model storage)")
        return True
    except OSError as exc:
        print(f"[WARNING] Hardlink failed for {dst}: {exc}")
        return False

def save_last_and_aliases(model, epoch, m, train_loss, save_epoch_alias=True, save_best=False):
    """
    Save exactly ONE physical model copy as last.pt.
    epoch_XX.pt and best.pt are hardlinks to that same physical file.
    This prevents the old script from consuming ~2.3 GB three times per epoch.
    """
    payload = make_payload(model, epoch, m, train_loss)
    last_path = SAVE_DIR / "last.pt"

    atomic_torch_save(payload, last_path)
    print(f"[CHECKPOINT] LAST: {last_path}")

    if save_epoch_alias:
        epoch_path = SAVE_DIR / f"epoch_{epoch:02d}.pt"
        if epoch_path != last_path:
            make_hardlink(last_path, epoch_path)

    if save_best:
        best_path = SAVE_DIR / "best.pt"
        if best_path != last_path:
            make_hardlink(last_path, best_path)

def validate_checkpoint(path, expected_min_epoch=0):
    """
    Verify that a checkpoint is actually readable before trusting it.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        epoch = int(ckpt.get("epoch", 0))
        state = ckpt.get(
            "model_state_dict",
            ckpt.get("state_dict", ckpt.get("model", ckpt))
        )
        if not isinstance(state, dict) or len(state) == 0:
            raise RuntimeError("checkpoint contains no usable model state")
        if epoch < expected_min_epoch:
            return None
        return ckpt
    except Exception as exc:
        print(f"[CHECKPOINT] INVALID: {path} -> {exc}")
        return None

def cleanup_old_duplicate_checkpoints(keep_epochs):
    """
    Remove old full-size checkpoint files from the previous script.
    Do NOT remove checkpoints listed in keep_epochs.
    Hardlinks created by this script do not cost extra disk space.
    """
    import glob

    keep = {f"epoch_{e:02d}.pt" for e in keep_epochs}
    keep.update({"last.pt", "best.pt"})

    for p in sorted(SAVE_DIR.glob("epoch_*.pt")):
        if p.name not in keep:
            try:
                p.unlink()
                print(f"[CLEANUP] Removed old checkpoint: {p.name}")
            except Exception as exc:
                print(f"[WARNING] Could not remove {p}: {exc}")

def save_history(history, history_path):
    tmp = history_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(history, indent=2), encoding="utf-8")
    tmp.replace(history_path)

def main():
    seed_everything()

    print("="*72)
    print("MANIPURI ASR V2 — B3-NARROW ROBUST RESUME")
    print("="*72)
    print("[PLAN] Continue from the latest VALID checkpoint")
    print("[PLAN] Target final epoch:", EPOCHS)

    print(f"[DEVICE] {DEVICE}")
    if DEVICE.type == "cuda":
        print("[GPU]", torch.cuda.get_device_name(0))
        print("[VRAM]", f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    print(f"[AMP] BF16 supported : {USE_BF16}")
    print(f"[AMP] dtype          : {AMP_DTYPE}")
    print(f"[DISK] Free before startup: {disk_free_gb():.2f} GB")

    for p in [TRAIN_CSV, VAL_CSV, ENCODER_MODEL, B2_CHECKPOINT, VOCAB_PATH, UNIGRAM_PATH]:
        if not p.exists():
            raise FileNotFoundError(f"Required path missing: {p}")
        print("[PATH] PASS:", p)

    train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    val_df = pd.read_csv(VAL_CSV, encoding="utf-8-sig")

    if len(train_df) != 7768 or len(val_df) != 864:
        raise RuntimeError(
            f"Expected train=7768,val=864; got {len(train_df)},{len(val_df)}"
        )

    vocab = load_vocab()
    print(f"[DATA] train={len(train_df):,} val={len(val_df):,} vocab={len(vocab)}")

    train_loader = DataLoader(
        AudioCTCDataset(train_df, vocab),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=DEVICE.type == "cuda",
    )

    val_loader = DataLoader(
        AudioCTCDataset(val_df, vocab),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=DEVICE.type == "cuda",
    )

    # ------------------------------------------------------------
    # Build the original B2 architecture, then configure B3.
    # ------------------------------------------------------------
    model = B2Model(len(vocab))
    load_b2_checkpoint(model)
    configure_b3(model)
    print_report(model)

    optimizer = make_optimizer(model)

    # BF16 does not use GradScaler.
    scaler = None

    ctc = nn.CTCLoss(
        blank=vocab[SPECIAL_BLANK],
        zero_infinity=True
    )

    decoder = build_decoder(vocab)
    inv_vocab = {int(v): k for k, v in vocab.items()}

    history_path = SAVE_DIR / "training_history.json"
    history = []
    best = float("inf")

    # ------------------------------------------------------------
    # Load existing history.
    # ------------------------------------------------------------
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text(encoding="utf-8"))
            if history:
                best = min(
                    float(x.get("beam_cer", float("inf")))
                    for x in history
                )
                print(
                    f"[HISTORY] {len(history)} epochs found | "
                    f"best Beam-500 CER={best*100:.4f}%"
                )
        except Exception as exc:
            print(f"[WARNING] Could not read training history: {exc}")
            history = []
            best = float("inf")

    # ------------------------------------------------------------
    # IMPORTANT:
    # The previous run successfully created epoch_08.pt, but crashed
    # while writing last.pt. Therefore, search for the newest VALID
    # checkpoint instead of blindly trusting last.pt.
    # ------------------------------------------------------------
    candidate_paths = []

    last_path = SAVE_DIR / "last.pt"
    if last_path.exists():
        candidate_paths.append(last_path)

    for p in sorted(
        SAVE_DIR.glob("epoch_*.pt"),
        key=lambda x: int(re.search(r"epoch_(\d+)", x.name).group(1)),
        reverse=True,
    ):
        candidate_paths.append(p)

    resume_path = None
    resume_ckpt = None
    completed = 0

    for candidate in candidate_paths:
        ckpt = validate_checkpoint(
            candidate,
            expected_min_epoch=EXPECTED_RESUME_EPOCH
        )
        if ckpt is not None:
            candidate_epoch = int(ckpt.get("epoch", 0))
            if candidate_epoch > completed:
                completed = candidate_epoch
                resume_path = candidate
                resume_ckpt = ckpt

    if resume_path is None:
        raise RuntimeError(
            "No valid B3 checkpoint at or beyond Epoch "
            f"{EXPECTED_RESUME_EPOCH} was found. "
            "Training was NOT started."
        )

    print()
    print("[RESUME] VALID checkpoint selected")
    print("[RESUME] Path      :", resume_path)
    print("[RESUME] Completed :", completed)
    print("[RESUME] Next      :", completed + 1)

    resume_state = resume_ckpt.get(
        "model_state_dict",
        resume_ckpt.get(
            "state_dict",
            resume_ckpt.get("model", resume_ckpt)
        )
    )

    missing, unexpected = model.load_state_dict(
        resume_state,
        strict=False
    )

    if missing:
        raise RuntimeError(
            "Missing resume keys:\n" + "\n".join(map(str, missing))
        )

    if unexpected:
        print("[WARNING] Unexpected resume keys:", unexpected)

    # Reconcile the best metric with the selected checkpoint itself.
    # This matters because the previous run crashed before it could
    # append Epoch 8 to training_history.json.
    resume_beam_cer = resume_ckpt.get("val_beam500_cer", None)
    if resume_beam_cer is not None:
        resume_beam_cer = float(resume_beam_cer)
        if resume_beam_cer < best:
            best = resume_beam_cer
            print(
                f"[RESUME] Checkpoint contains a newer best: "
                f"Beam-500 CER={best*100:.4f}%"
            )

    start_epoch = completed + 1

    # ------------------------------------------------------------
    # If the selected checkpoint is epoch_08.pt, make last.pt a
    # HARDLINK to it. No extra 2.3 GB copy is created.
    # ------------------------------------------------------------
    if resume_path.name != "last.pt":
        print("[RESUME] Repairing last.pt using a hardlink to the valid checkpoint.")
        if last_path.exists():
            try:
                last_path.unlink()
            except Exception as exc:
                print("[WARNING] Could not remove stale last.pt:", exc)

        make_hardlink(resume_path, last_path)

    # If the selected checkpoint is itself the best checkpoint,
    # replace the old full-size best.pt with a zero-space hardlink.
    if resume_beam_cer is not None and resume_beam_cer <= best:
        best_path = SAVE_DIR / "best.pt"
        if best_path.exists():
            try:
                best_path.unlink()
            except Exception as exc:
                print("[WARNING] Could not remove stale best.pt:", exc)
        make_hardlink(resume_path, best_path)

    # ------------------------------------------------------------
    # Remove duplicate old epoch checkpoint files.
    #
    # Keep the actual selected source epoch. Older epoch_05/06/07
    # files from the previous script were separate 2.3 GB copies.
    # ------------------------------------------------------------
    cleanup_old_duplicate_checkpoints(
        keep_epochs={completed}
    )

    print(f"[DISK] Free after cleanup: {disk_free_gb():.2f} GB")

    if start_epoch > EPOCHS:
        print("\n" + "="*72)
        print("B3-NARROW ALREADY COMPLETE")
        print("="*72)
        print(f"Best Beam-500 CER : {best*100:.4f}%")
        print("Official 410-sample test set: NOT TOUCHED")
        print("Best checkpoint:", SAVE_DIR / "best.pt")
        return

    # ------------------------------------------------------------
    # CONTINUE TRAINING.
    #
    # IMPORTANT: The previous script did NOT save optimizer state.
    # Therefore this is a weight-continuation with a fresh optimizer,
    # exactly as before.
    # ------------------------------------------------------------
    for epoch in range(start_epoch, EPOCHS + 1):

        print("\n" + "="*72)
        print(f"B3-NARROW EPOCH {epoch}/{EPOCHS}")
        print("="*72)

        train_loss, minutes, steps = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            ctc,
            epoch
        )

        print(
            f"[TRAIN] loss={train_loss:.6f} "
            f"time={minutes:.2f} min "
            f"optimizer_steps={steps}"
        )

        gc.collect()

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        # Use the exact validation signature of this script.
        m = validate(
            model,
            val_loader,
            vocab,
            decoder,
            inv_vocab
        )

        print("\n" + "="*72)
        print(f"EPOCH {epoch} RESULTS")
        print("="*72)
        print(f"Train loss       : {train_loss:.6f}")
        print(f"Val loss         : {m['loss']:.6f}")
        print(f"Greedy CER       : {m['greedy_cer']*100:.4f}%")
        print(f"Greedy WER       : {m['greedy_wer']*100:.4f}%")
        print(f"Beam-500 CER     : {m['beam_cer']*100:.4f}%")
        print(f"Beam-500 WER     : {m['beam_wer']*100:.4f}%")
        print("B2 Beam-500 CER  : 20.1736%")
        print("B2 Beam-500 WER  : 71.7230%")

        improved = m["beam_cer"] < best

        # --------------------------------------------------------
        # CRITICAL FIX:
        # Save ONLY ONE physical model file.
        #
        # last.pt = physical checkpoint
        # epoch_XX.pt = hardlink
        # best.pt = hardlink if this epoch is best
        #
        # This prevents 2.3 GB x 3 copies per epoch.
        # --------------------------------------------------------
        save_last_and_aliases(
            model,
            epoch,
            m,
            train_loss,
            save_epoch_alias=True,
            save_best=improved
        )

        if improved:
            best = m["beam_cer"]
            print(
                f"🔥 NEW BEST: Beam-500 CER={best*100:.4f}%"
            )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            **m
        })

        save_history(history, history_path)

        print(
            f"[DISK] Free after epoch {epoch}: "
            f"{disk_free_gb():.2f} GB"
        )

    print("\n" + "="*72)
    print("B3-NARROW COMPLETE")
    print("="*72)
    print(f"Best Beam-500 CER : {best*100:.4f}%")
    print("B2 baseline       : 20.1736%")
    print(
        f"Absolute change   : "
        f"{20.1736-best*100:+.4f} pp"
    )
    print("Official 410-sample test set: NOT TOUCHED")
    print("Best checkpoint:", SAVE_DIR / "best.pt")
    print("Last checkpoint:", SAVE_DIR / "last.pt")


if __name__=="__main__":
    main()
