# Meiteilon ASR — Handoff Memory & Layout Plan

*Read this first on any new machine or new session before touching any script. It exists because this project has already been running across two machines with drifting file versions — this document is the single source of truth for "what is actually true right now," overriding anything a stale file or an old chat transcript implies.*

---

## 1. What this project is

A research-grade ASR system for Meiteilon (Manipuri) using a novel audio-LLM pipeline — no prior published Manipuri ASR work uses this paradigm (existing work is CTC/hybrid), so the architecture itself is the core novelty claim for the paper.

```
16kHz audio
    │
    ▼
Wav2Vec2-BERT 2.0 (facebook/w2v-bert-2.0)   — frozen, bf16, 1024-d output
    │
    ▼
Audio Projector   — Conv1d(stride 2) + Linear + LayerNorm, ~5.25M trainable params, 1024 → 2048
    │
    ▼
Qwen2.5-3B-Instruct   — 4-bit NF4 quantized, QLoRA (r=16), tokenizer extended with 44 Meitei Mayek chars
    │
    ▼
Meitei Mayek text (target script — see §4)
```

## 2. Hardware — two confirmed machines, do not conflate them

| | Machine A | Machine B |
|---|---|---|
| GPU | RTX 5060 Laptop | RTX 2050 |
| Architecture | Blackwell, compute capability 12.0 (sm_120) | Ampere, compute capability 8.6 |
| VRAM | 8 GB dedicated | 4 GB dedicated |
| System RAM | — | 16 GB (Acer Aspire A715-79G) |
| OS path seen | `C:\Users\Churchil Ningthoujam\Downloads\manipuri_stt-...` | `D:\boom\manipuri_stt-...(2)\...` |
| Status | **Completed a real 3-epoch training run successfully** (§5) | Currently blocked on VRAM (§6) |

Anything copied between machines needs re-verification, not blind trust — `bitsandbytes` version, `venv`, downloaded model weights, and any feature cache are **not portable as-is** because cache fingerprints and CUDA/driver specifics differ per machine.

## 3. Known, verified environment gotchas (don't rediscover these)

- **bitsandbytes + Blackwell (sm_120) + Windows**: historically broken; fixed from `bitsandbytes>=0.45.3`. Verified working on Machine A via a dedicated NF4 forward+backward test — do not assume, re-run the same style of test on any new machine.
- **`torchaudio.load()`** on recent versions requires a separate `torchcodec` package or it crashes outright. Workaround already in use: load via `soundfile` instead, only use `torchaudio.functional.resample`.
- **PEFT's `modules_to_save=["embed_tokens", "lm_head"]` is a footgun**: it makes the *entire* embedding/LM-head matrix trainable, not just new vocabulary rows. This is what caused the RTX 2050 OOM (§6) — confirmed by independent arithmetic: `151,709 × 2048 × 2 ≈ 621.4M`, matching the reported trainable-parameter count almost exactly.
- **PowerShell console mojibake**: the Windows terminal does not render UTF-8 (Meetei Mayek / Bengali) correctly by default. Never trust what a script prints to console for non-ASCII text — write to a UTF-8 file and open it in VS Code/Notepad++ instead. This exact issue caused a real, confirmed script-identity mixup during the data audit (see §4).
- **Any "diagnostic" pasted from another source** (another AI session, another tool) has repeatedly contained real factual errors when checked against the actual running files — wrong learning-rate values, claims that code was missing when it was already present, line numbers that don't exist in the actual file. Standing rule for this project: **read the real file before patching it. Every time this rule was skipped, a real bug was later found.**

## 4. Data pipeline — verified, working, don't redo

- **12,007 total audio files** across three pools, 0 missing on disk (verified programmatically):
  - `manipuri_1k_project`: 1,000 files, **native Meetei Mayek** script
  - `benchmark`: 100 files, **native Meetei Mayek** script
  - `manipuri_benchmark_extracted`: 10,907 files (**91% of the corpus**), **Bengali-script transliterated** Manipuri — confirmed by decoding actual Unicode code points, not by appearance
- Transliteration tool `hoomexsun/xlit` (MIT-licensed, cloned and test-run against real project data, confirmed genuine Meetei Mayek Unicode output) unifies all three pools into one script.
- `unified_ground_truth.csv` → `train.csv` (10,199) / `val.csv` (1,207) / `test.csv` (601) via source-stratified split. **Not speaker-disjoint** — no speaker/dialect labels exist in the source data; flag this as a limitation in the paper.
- A `checkpoints/whisper-manipuri/` directory exists in the project root — **ownership still unconfirmed** (Baka's own prior work, or inherited?). Do not treat it as a citable baseline until this is settled.

## 5. Machine A (RTX 5060) — real training result, correctly diagnosed

A real 3-epoch run completed on real data. Key facts, because an external "diagnostic" got several of these wrong when checked against the actual log:

- **No NaN occurred.** Loss was stable: `7.6103 → 7.6017 → 7.5929`.
- Validation CER trended down each epoch: `14.05 → 8.94 → 2.10` — real, if noisy (only 10 validation examples were evaluated per epoch, which is too small a sample to trust precisely).
- Epoch 2 was anomalously slow (both training and validation) — almost certainly an external system factor (thermal throttling, background process), not a code bug.
- Checkpoints landed in `ASR\checkpoints\epoch_1` / `epoch_2` / `epoch_3` / `best_model` (note: **not** `manipuri_stt\checkpoints\`, which only holds the unrelated `whisper-manipuri` baseline).
- Legitimate patches applied on top of the original `train.py`: bf16 instead of fp16, finite-loss/finite-gradient guards placed correctly (before `backward()`/`optimizer.step()`), early stopping with best-model tracking, checkpoint resume support, a pre-flight filter for un-tokenizable transcripts, and a frozen-encoder embedding cache (verified via an instrumented test: exactly one encoder call per unique file on epoch 1, zero on epoch 2).
- **"Epoch 6" was asked about but never confirmed to exist.** The resume logic reloads weights correctly but does not continue epoch *numbering* — a naive resume-and-continue would have overwritten `epoch_1`–`3`, not extended to `4`–`6`. Check the actual checkpoint folder contents before assuming a 6-epoch history exists anywhere.

## 6. Machine B (RTX 2050) — v2 system, currently blocked

A separate, larger "v2" codebase exists under `ASR\v2\` (`train_cached_preflight.py`, now reportedly 3,200+ lines and growing; `feature_cache.py`; `precompute_features.py`; `encoder_cache_benchmark.py`) built to fit training within 4GB VRAM via a Soup-inspired resource-orchestration philosophy (hardware detection, feature caching, preflight validation before committing to a long run).

**What's confirmed working on this machine:** hardware detection, dataset loading/validation (10,199/1,207/601, matching Machine A), frozen encoder feature caching (~11.4k features cached, correctly moving W2V-BERT's cost out of the training loop), projector (5,250,048 params) and Qwen 4-bit+LoRA loading, forward pass, and one backward pass with gradients flowing correctly into projector/LoRA/new-embedding rows.

**What's actually blocking progress — two distinct, unresolved issues:**

1. **Root cause of the OOM, diagnosed and the fix designed, but the fix is NOT YET IMPLEMENTED in the real file.** `modules_to_save=["embed_tokens","lm_head"]` made ~621M parameters trainable instead of the intended 44 new token rows, pushing peak VRAM to ~7.9GB on a 4GB card. **The correct fix is a split-embedding design** — keep the original embedding frozen exactly as-is and add a small separate trainable `nn.Parameter` of shape `(44, 2048)` for only the new tokens (same idea for the output head). This cuts Adam's optimizer-state burden from ~621M to ~90K parameters. A concrete implementation was written and handed over, but **has not yet been applied to the real, current `train_cached_preflight.py`** — the version available for editing was confirmed stale (a "line 3259" crash was reported, and no error text or line-765 content matched anything findable in the file actually on hand). **Next step here: get the real current file (not a possibly-reformatted export), apply the split-embedding patch, re-run preflight.**
2. **A separate, still-open issue**: a feature-cache fingerprint mismatch (`load_feature_cache` raising on a hash mismatch between the cache's recorded fingerprint and a freshly computed one). Best working hypothesis — not yet confirmed — is that the cache was built under one encoder dtype config (fp32, from an earlier version of `encoder_cache_benchmark.py`) and is now being checked against a corrected bf16 config, so the mismatch is the safety check correctly doing its job rather than a bug to route around. Two live options: delete and rebuild the (likely cheap, preflight-scale) cache under the current config, or get the actual `feature_cache.py` / `precompute_features.py` to confirm precisely what changed.

## 7. Proposed clean layout for a fresh/consolidated machine

The current state (two machines, a "v1" script set and a larger "v2" system, multiple checkpoint locations, at least one confirmed stale-file incident) has enough sprawl that a straight file copy risks carrying the confusion forward. Recommended structure when setting up anywhere new:

```
manipuri_stt/
├── HANDOFF_MEMORY.md              ← this document, kept current
├── config.json                    ← ONE canonical config; note which machine/VRAM it targets
├── requirements.txt
│
├── data/
│   ├── raw/
│   │   ├── manipuri_1k_project/
│   │   ├── manipuri_benchmark_extracted/
│   │   └── benchmark/
│   ├── unified_ground_truth.csv
│   ├── train.csv / val.csv / test.csv
│   └── flagged_for_review.csv     ← transliteration QC output
│
├── scripts/
│   ├── data_pipeline/             ← audit_datasets.py, build_unified_corpus.py,
│   │                                 validate_unified_corpus.py, split_dataset.py
│   ├── env_checks/                ← verify_gpu.py, verify_bnb_4bit.py
│   └── xlit/                      ← cloned transliteration tool, kept separate/read-only
│
├── training/
│   ├── v1_rtx5060/                ← the proven, working pipeline (train.py + its embed_cache/)
│   │   ├── train.py
│   │   ├── precompute_embeddings.py
│   │   └── embed_cache/           ← fingerprint/config tied to THIS pipeline only
│   └── v2_rtx2050/                ← the memory-constrained variant, clearly separated
│       ├── train_cached_preflight.py
│       ├── feature_cache.py
│       ├── precompute_features.py
│       └── cache/features/        ← fingerprint/config tied to THIS pipeline only — never mix with v1's cache
│
├── checkpoints/
│   ├── whisper_baseline/          ← renamed from whisper-manipuri once ownership is confirmed, with a note on provenance
│   ├── v1_rtx5060/
│   │   ├── epoch_1/ epoch_2/ epoch_3/ best_model/
│   │   └── training_log.csv
│   └── v2_rtx2050/                ← kept empty/separate until a successful run actually completes
│
└── docs/
    ├── Meiteilon_ASR_Architecture_and_Research_Roadmap.md
    └── PROJECT_STATUS.md
```

The load-bearing rule behind this layout: **v1 and v2 never share a cache directory or a checkpoint directory.** Every fingerprint mismatch and every "which file actually ran" confusion so far traces back to artifacts from different pipeline versions sitting in the same place.

## 8. Immediate next steps, in order

1. Get the actual current `train_cached_preflight.py` (or at minimum `stage_two_overfit()`, the optimizer construction, and the tokenizer/`resize_token_embeddings` logic) and apply the split-embedding fix for real.
2. Resolve the feature-cache fingerprint mismatch — confirm the dtype-drift hypothesis or find the real cause, then either rebuild the cache or fix the config drift.
3. Re-run the Machine B preflight with both fixes in place; confirm trainable-parameter count drops from ~651M to roughly `30M (LoRA) + ~90K (new embeddings) + ~5.25M (projector)`.
4. Once Machine B trains successfully, evaluate against the Machine A 3-epoch baseline using the same held-out test set (601 rows) and CER as the primary metric (not WER — whitespace segmentation isn't reliable for Meetei Mayek).
5. Confirm `checkpoints/whisper-manipuri/` provenance before citing it as a baseline in the paper.
6. Diagnose the still-open `<`/`>` characters found in `manipuri_1k_project` during the original data audit (possible annotation tags needing stripping).
