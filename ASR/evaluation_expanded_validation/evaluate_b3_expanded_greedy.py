from pathlib import Path
import sys, json, unicodedata, re, time, gc
import numpy as np
import pandas as pd
import torch
import soundfile as sf
from torch.utils.data import Dataset, DataLoader

ASR = Path.cwd().parent
V2 = ASR / "manipuri_asr_v2"
sys.path.insert(0, str(V2))

import train_b3_narrow_E11_E12 as b3

MANIFEST = ASR / "xlit" / "audio_audit" / "final_candidate" / "validation_16k.csv"
CHECKPOINT = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"
OUT = ASR / "evaluation_expanded_validation"
OUT.mkdir(parents=True, exist_ok=True)

def norm(s):
    if s is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()

def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (ca != cb)
            ))
        previous = current
    return previous[-1]

def score(ref, hyp):
    ref, hyp = norm(ref), norm(hyp)
    char_errors = edit_distance(list(ref), list(hyp))
    word_errors = edit_distance(ref.split(), hyp.split())
    return char_errors, max(len(ref), 1), word_errors, max(len(ref.split()), 1)

class CleanWavDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = Path(str(row["audio_path"]))
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sr}: {path}")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(audio):
            raise ValueError(f"Empty audio: {path}")
        return audio, norm(row["text"]), str(path), str(row.get("source", ""))

def main():
    print("=" * 70)
    print("B3 BEST CHECKPOINT — EXPANDED VALIDATION GREEDY EVALUATION")
    print("=" * 70)
    print("Manifest:", MANIFEST)
    print("Checkpoint:", CHECKPOINT)
    print("Output:", OUT)
    print("Device:", b3.DEVICE)

    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    df = pd.read_csv(MANIFEST, encoding="utf-8-sig")
    required = {"audio_path", "text"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest needs columns {required}; found {list(df.columns)}")
    if len(df) != 1693:
        raise ValueError(f"Expected 1693 rows; found {len(df)}")
    if df["audio_path"].isna().any() or df["text"].isna().any():
        raise ValueError("Manifest contains missing audio paths or transcripts.")
    if df["audio_path"].duplicated().any():
        raise ValueError("Duplicate audio paths detected; stop and inspect manifest.")

    for p in df["audio_path"].astype(str):
        if not Path(p).is_file():
            raise FileNotFoundError(p)

    vocab = b3.load_vocab()
    print("[VOCAB] size:", len(vocab), "blank:", vocab.get("<blank>"))
    if len(vocab) != 60 or vocab.get("<blank>") != 0:
        raise RuntimeError("Vocabulary does not match expected B3 configuration.")

    model = b3.B2Model(len(vocab))
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if "model_state_dict" not in ckpt:
        raise RuntimeError("Checkpoint has no model_state_dict.")
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    model.to(b3.DEVICE)
    model.eval()

    print("[CHECKPOINT] epoch:", ckpt.get("epoch"))
    print("[CHECKPOINT] recorded greedy CER:", ckpt.get("val_greedy_cer"))
    print("[CHECKPOINT] recorded beam CER:", ckpt.get("val_beam500_cer"))
    print("[LOAD] all checkpoint keys matched.")

    inv_vocab = {int(v): k for k, v in vocab.items()}
    loader = DataLoader(
        CleanWavDataset(df), batch_size=1, shuffle=False,
        num_workers=0, collate_fn=lambda x: x[0]
    )

    results = []
    char_err = char_total = word_err = word_total = 0
    start = time.time()

    with torch.inference_mode():
        for i, (audio, reference, path, source) in enumerate(loader, 1):
            features = b3.make_features(model.processor, [torch.from_numpy(audio)]).to(b3.DEVICE)
            use_amp = b3.DEVICE.type == "cuda" and b3.USE_BF16
            with torch.autocast(
                device_type="cuda",
                dtype=b3.AMP_DTYPE,
                enabled=use_amp
            ):
                logits = model(features)

            ids = logits.argmax(-1)[0].detach().cpu().tolist()
            hypothesis = norm(b3.decode_ids(ids, inv_vocab, blank_id=vocab["<blank>"]))
            ce, ct, we, wt = score(reference, hypothesis)
            char_err += ce
            char_total += ct
            word_err += we
            word_total += wt

            results.append({
                "index": i,
                "audio_path": path,
                "source": source,
                "reference": reference,
                "prediction_greedy": hypothesis,
                "char_errors": ce,
                "reference_chars": ct,
                "word_errors": we,
                "reference_words": wt,
            })

            if i % 25 == 0 or i == len(loader):
                print(
                    f"[{i}/{len(loader)}] "
                    f"CER={100*char_err/char_total:.3f}% "
                    f"WER={100*word_err/word_total:.3f}%",
                    flush=True
                )

            del features, logits
            if i % 100 == 0:
                gc.collect()
                if b3.DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

    result_df = pd.DataFrame(results)
    result_path = OUT / "greedy_predictions_1693.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    summary = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)),
        "manifest": str(MANIFEST),
        "num_clips": len(results),
        "decoder": "greedy CTC",
        "cer": char_err / max(char_total, 1),
        "wer": word_err / max(word_total, 1),
        "cer_percent": 100 * char_err / max(char_total, 1),
        "wer_percent": 100 * word_err / max(word_total, 1),
        "elapsed_minutes": (time.time() - start) / 60,
        "test_benchmark_used": False,
    }
    summary_path = OUT / "greedy_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== FINAL GREEDY RESULTS ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nPredictions:", result_path)
    print("Summary:", summary_path)

if __name__ == "__main__":
    main()
