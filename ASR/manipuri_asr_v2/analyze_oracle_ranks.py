import json
import unicodedata
import numpy as np
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent
NBEST_PATH = ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "nbest" / "b3_4gram_pooled100_val.json"

def normalize_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split()).strip()

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]

def calc_cer(ref, hyp):
    if not ref: return 0.0 if not hyp else 1.0
    return edit_distance(list(ref), list(hyp)) / len(ref)

def calc_wer(ref, hyp):
    ref_w = ref.split()
    hyp_w = hyp.split()
    if not ref_w: return 0.0 if not hyp_w else 1.0
    return edit_distance(ref_w, hyp_w) / len(ref_w)

print("Loading pooled N-best dataset...")
with open(NBEST_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = data["samples"] if isinstance(data, dict) else data
total_samples = len(samples)
print(f"Loaded {total_samples} samples.")

# Track rank distributions
best_cer_ranks = []
best_wer_ranks = []

# Cumulative totals for Top-K curves
k_thresholds = [1, 2, 3, 5, 10, 20, 30, 40, 50, 100]
topk_cer_errs = {k: 0 for k in k_thresholds}
topk_wer_errs = {k: 0 for k in k_thresholds}

total_ref_chars = 0
total_ref_words = 0

for sample in samples:
    ref = normalize_text(sample.get("reference", sample.get("text", "")))
    hyps = sample.get("hypotheses", sample.get("nbest", []))
    if not hyps: continue

    ref_chars = len(ref)
    ref_words = len(ref.split())
    total_ref_chars += ref_chars
    total_ref_words += ref_words

    cers, wers = [], []
    for h in hyps:
        htext = normalize_text(h.get("text", ""))
        cers.append(calc_cer(ref, htext))
        wers.append(calc_wer(ref, htext))

    # Rank of best CER / WER candidate (1-indexed)
    best_cer_idx = int(np.argmin(cers))
    best_wer_idx = int(np.argmin(wers))
    
    best_cer_ranks.append(best_cer_idx + 1)
    best_wer_ranks.append(best_wer_idx + 1)

    # Top-K oracle errors
    for k in k_thresholds:
        sub_hyps = hyps[:k]
        sub_texts = [normalize_text(h.get("text", "")) for h in sub_hyps]
        
        # Best CER within Top-K
        sub_cer_errs = [edit_distance(list(ref), list(t)) for t in sub_texts]
        topk_cer_errs[k] += min(sub_cer_errs)

        # Best WER within Top-K
        sub_wer_errs = [edit_distance(ref.split(), t.split()) for t in sub_texts]
        topk_wer_errs[k] += min(sub_wer_errs)

# =====================================================================
# REPORTING
# =====================================================================

print("\n" + "="*70)
print("ORACLE RANK DISTRIBUTION SUMMARY")
print("="*70)

def print_bucket_distribution(ranks, metric_name):
    print(f"\n--- Best {metric_name} Candidate Rank Distribution ---")
    buckets = [
        ("Rank 1", lambda r: r == 1),
        ("Rank 2", lambda r: r == 2),
        ("Rank 3", lambda r: r == 3),
        ("Rank 4–5", lambda r: 4 <= r <= 5),
        ("Rank 6–10", lambda r: 6 <= r <= 10),
        ("Rank 11–20", lambda r: 11 <= r <= 20),
        ("Rank 21–30", lambda r: 21 <= r <= 30),
        ("Rank 31–50", lambda r: 31 <= r <= 50),
        ("Rank 51+", lambda r: r > 50),
    ]
    for label, fn in buckets:
        cnt = sum(1 for r in ranks if fn(r))
        pct = 100 * cnt / total_samples
        print(f"  {label:<12}: {cnt:4d} / {total_samples} ({pct:6.2f}%)")

print_bucket_distribution(best_cer_ranks, "CER")
print_bucket_distribution(best_wer_ranks, "WER")

print("\n" + "="*70)
print("TOP-K ORACLE CURVE")
print("="*70)
print(f"{'Pool Cutoff':<15} | {'Oracle CER (%)':<15} | {'Oracle WER (%)':<15}")
print("-" * 50)

for k in k_thresholds:
    cer_pct = 100 * topk_cer_errs[k] / total_ref_chars
    wer_pct = 100 * topk_wer_errs[k] / total_ref_words
    print(f"Top-{k:<10} | {cer_pct:14.4f}% | {wer_pct:14.4f}%")

print("="*70)