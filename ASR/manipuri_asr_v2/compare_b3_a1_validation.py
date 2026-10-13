from pathlib import Path
import gc, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pyctcdecode import build_ctcdecoder
import train_b3_a1_specaugment as tr

# Validation-only: this script never opens test.csv or any test checkpoint.
ASR = tr.ASR
VAL_CSV = ASR / "combined_dataset" / "val.csv"
E11 = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "best.pt"
A1_DIR = ASR / "checkpoints" / "v3_ctc" / "b3_a1_specaugment"
A1_E4 = A1_DIR / "a1_epoch_04.pt"
A1_E5 = A1_DIR / "a1_epoch_05.pt"
OUT = tr.ROOT / "b3_a1_validation_comparison"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = tr.DEVICE
BEAM = 500
EXPECTED_VAL = 864


def build_model(vocab_size):
    model = tr.B2Model(vocab_size)

    # E11 is a complete model checkpoint.
    # It restores the encoder, adapter and CTC head.
    tr.load_e11(model)

    tr.configure_b3(model)

    return model


def overlay_a1(model, path, expected_epoch):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != tr.CHECKPOINT_FORMAT:
        raise RuntimeError(f"Wrong checkpoint format: {path}")
    if int(ckpt.get("epoch", -1)) != expected_epoch:
        raise RuntimeError(f"Wrong epoch in {path}: {ckpt.get('epoch')}")
    state = ckpt.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"Empty trainable state: {path}")

    tr.configure_b3(model)
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    saved = set(state)
    if expected != saved:
        raise RuntimeError(
            f"Trainable-key mismatch for {path.name}: "
            f"missing={len(expected-saved)}, unexpected={len(saved-expected)}"
        )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or (set(missing) & expected):
        raise RuntimeError(
            f"Unsafe A1 overlay: unexpected={unexpected}, "
            f"missing trainable={set(missing) & expected}"
        )
    tr.configure_b3(model)
    print(f"[A1 OVERLAY PASS] {path.name}: epoch {expected_epoch}, {len(saved)} tensors")


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + (ca != cb)))
        prev = cur
    return prev[-1]


def metrics(refs, hyps):
    ce = ct = we = wt = 0
    for r, h in zip(refs, hyps):
        r, h = tr.normalize_text(r), tr.normalize_text(h)
        ce += lev(list(r), list(h)); ct += max(len(r), 1)
        rw, hw = r.split(), h.split()
        we += lev(rw, hw); wt += max(len(rw), 1)
    return {
        "cer": ce / max(ct, 1), "wer": we / max(wt, 1),
        "char_errors": ce, "char_total": ct,
        "word_errors": we, "word_total": wt,
    }


def make_decoder(vocab):
    labels = [k for k, v in sorted(vocab.items(), key=lambda kv: kv[1])]
    labels = ["" if x == tr.SPECIAL_BLANK else x for x in labels]
    with open(tr.UNIGRAM_PATH, encoding="utf-8") as f:
        unigrams = [x.strip() for x in f if x.strip()]
    print(f"[DECODER] {len(labels)} labels, {len(unigrams):,} unigrams, beam={BEAM}")
    return build_ctcdecoder(labels=labels, unigrams=unigrams)


@torch.inference_mode()
def evaluate(name, model, loader, vocab, decoder):
    model.eval()
    inverse = {int(v): k for k, v in vocab.items()}
    refs, greedy, beam = [], [], []
    total = len(loader.dataset)

    for n, batch in enumerate(loader, 1):
        feats = tr.make_features(model.processor, batch["audio"]).to(DEVICE)
        bf16 = DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            logits = model(feats)
        ids = logits.argmax(-1).cpu().tolist()
        lp = logits.float().log_softmax(-1).cpu().numpy()

        for i, ref in enumerate(batch["texts"]):
            refs.append(tr.normalize_text(ref))
            greedy.append(tr.normalize_text(
                tr.decode_ids(ids[i], inverse, blank_id=vocab[tr.SPECIAL_BLANK])
            ))
            beam.append(tr.normalize_text(decoder.decode(lp[i], beam_width=BEAM)))

        if n % 50 == 0 or n == total:
            print(f"[{name}] {n}/{total}")
        del feats, logits, lp
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    result = {"model": name, "n": len(refs),
              "greedy": metrics(refs, greedy), "beam500": metrics(refs, beam)}
    pred = pd.DataFrame({
        f"{name}_greedy": greedy,
        f"{name}_beam500": beam,
    })
    del model
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return result, pred


def main():
    print("B3 E11 vs A1 E4 vs A1 E5 — VALIDATION ONLY")
    print("Official test set will NOT be accessed.")
    for p in [VAL_CSV, E11, A1_E4, A1_E5, tr.ENCODER_MODEL,
              tr.B2_CHECKPOINT, tr.VOCAB_PATH, tr.UNIGRAM_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)
        print("[PATH PASS]", p)

    df = pd.read_csv(VAL_CSV, encoding="utf-8-sig")
    if len(df) != EXPECTED_VAL:
        raise RuntimeError(f"Expected {EXPECTED_VAL} validation rows; found {len(df)}")
    vocab = tr.load_vocab()
    loader = DataLoader(
        tr.AudioCTCDataset(df, vocab), batch_size=1, shuffle=False,
        num_workers=0, collate_fn=tr.collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    decoder = make_decoder(vocab)

    runs = [
        ("B3_E11", E11, None, 11),
        ("A1_E4", A1_E4, 4, 4),
        ("A1_E5", A1_E5, 5, 5),
    ]
    results, prediction_parts = [], []
    for name, path, a1_epoch, epoch in runs:
        print("\n" + "=" * 65 + "\n", name, "\n" + "=" * 65)
        model = build_model(len(vocab))
        if a1_epoch is not None:
            overlay_a1(model, path, epoch)
        else:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(ckpt, dict) and int(ckpt.get("epoch", epoch)) != epoch:
                raise RuntimeError(f"E11 checkpoint epoch mismatch: {ckpt.get('epoch')}")
        result, pred = evaluate(name, model, loader, vocab, decoder)
        results.append(result)
        prediction_parts.append(pred)
        print(f"Greedy CER/WER: {result['greedy']['cer']*100:.4f}% / "
              f"{result['greedy']['wer']*100:.4f}%")
        print(f"Beam-{BEAM} CER/WER: {result['beam500']['cer']*100:.4f}% / "
              f"{result['beam500']['wer']*100:.4f}%")

    # Row order is preserved because all runs use the same non-shuffled loader.
    combined = pd.concat(prediction_parts, axis=1)
    combined.insert(0, "row_index", np.arange(len(combined)))
    combined.insert(1, "reference", [tr.normalize_text(x) for x in df["text"]])
    combined.to_csv(OUT / "validation_predictions_b3_a1.csv",
                    index=False, encoding="utf-8-sig")

    summary = [{
        "model": x["model"], "n": x["n"],
        "greedy_CER_percent": x["greedy"]["cer"] * 100,
        "greedy_WER_percent": x["greedy"]["wer"] * 100,
        "beam500_CER_percent": x["beam500"]["cer"] * 100,
        "beam500_WER_percent": x["beam500"]["wer"] * 100,
    } for x in results]
    pd.DataFrame(summary).to_csv(
        OUT / "validation_summary_b3_a1.csv", index=False, encoding="utf-8-sig"
    )
    with open(OUT / "validation_metrics_b3_a1.json", "w", encoding="utf-8") as f:
        json.dump({
            "evaluation": "validation_only",
            "validation_rows": len(df),
            "normalization": "NFC + whitespace collapse",
            "decoder": "pyctcdecode unigram decoder",
            "beam_width": BEAM,
            "official_test_accessed": False,
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print("\nFINAL SUMMARY")
    print(pd.DataFrame(summary).to_string(index=False))
    print("\nSaved outputs to:", OUT)
    print("[SAFETY] Official 410-sample test set was not accessed.")


if __name__ == "__main__":
    main()
