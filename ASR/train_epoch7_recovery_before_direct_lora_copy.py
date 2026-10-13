"""
Epoch-7 Recovery Trainer
========================

Purpose
-------
Continue from the Epoch-6 checkpoint while fixing the catastrophic
full-vocabulary embedding/lm_head training problem.

Preserved from Epoch-6
----------------------
- Qwen2.5-3B-Instruct
- 4-bit NF4 + double quantization
- W2V-BERT 2.0 frozen encoder
- Epoch-6 AudioProjector architecture
- Epoch-6 tokenizer
- 44 Meitei Mayek tokens
- ChatML prompt
- Epoch-6 task prefix
- prefix -> audio -> target objective
- target text + <|im_end|>
- LoRA configuration

Changed for Epoch-7
-------------------
- Full embedding matrix is frozen.
- Full LM-head matrix is frozen.
- Only the 44 Meitei Mayek rows are trainable.
- The 44 Meitei rows are shared between input embedding and output head.
- Epoch-6 optimizer state is NOT loaded.
- Epoch-6 LoRA weights ARE warm-started.
- Epoch-6 projector weights ARE warm-started.

Modes
-----
Dry run:
    python .\ASR\train_epoch7_recovery.py --dry-run

Overfit test:
    python .\ASR\train_epoch7_recovery.py --overfit

Full training:
    python .\ASR\train_epoch7_recovery.py --train
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from jiwer import cer, wer
from tqdm.auto import tqdm

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from safetensors import safe_open
from torch.optim import AdamW


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path(
    r"E:\manipuri_stt1\manipuri_stt4"
    r"\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt"
)

ASR = ROOT / "ASR"

EPOCH6 = (
    ASR
    / "checkpoints"
    / "epoch_6_broken_objective"
)

EPOCH6_ADAPTER = EPOCH6 / "qwen_lora"
EPOCH6_PROJECTOR = EPOCH6 / "audio_projector.pt"
EPOCH6_TOKENIZER = EPOCH6 / "tokenizer"

TRAIN_CSV = ASR / "train.csv"
VAL_CSV = ASR / "val.csv"
TEST_CSV = ASR / "test.csv"

CACHE_DIR = (
    ASR
    / "v2"
    / "cache"
    / "features"
)

OUTPUT_ROOT = (
    ASR
    / "checkpoints"
    / "epoch_7_recovery"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# MODEL CONSTANTS
# ============================================================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ENCODER_MODEL = "facebook/w2v-bert-2.0"

BASE_VOCAB_SIZE = 151936
FINAL_VOCAB_SIZE = 151709

HIDDEN_SIZE = 2048
AUDIO_DIM = 1024

MEITEI_START = 151665
MEITEI_END = 151709
MEITEI_COUNT = 44

SYSTEM_PROMPT = (
    "You are a helpful speech recognition assistant."
)

# IMPORTANT:
# This is the task_prefix used by the original Epoch-6 backup.
TASK_PREFIX = (
    "Transcribe Manipuri speech to Meitei Mayek text:"
)

MAX_AUDIO_SECONDS = 30.0
MAX_TEXT_TOKENS = 256

SEED = 42

PROJECTOR_LR = 5e-4
QWEN_LR = 2e-4

GRAD_ACCUM = 1

OVERFIT_SIZE = 20
OVERFIT_EPOCHS = 3

VAL_SUBSET = 100

EARLY_STOPPING_PATIENCE = 3


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    COMPUTE_DTYPE = torch.bfloat16
else:
    COMPUTE_DTYPE = torch.float32


# ============================================================================
# REPRODUCIBILITY
# ============================================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================================
# HELPERS
# ============================================================================

def gb(x: int) -> float:
    return x / (1024 ** 3)


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def cuda_report(label: str):
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print(
        f"[GPU] {label}: "
        f"allocated={allocated:.2f} GB "
        f"reserved={reserved:.2f} GB"
    )


# ============================================================================
# AUDIO PROJECTOR
# EXACT EPOCH-6 ARCHITECTURE
# ============================================================================

class AudioProjector(nn.Module):

    def __init__(
        self,
        audio_dim=1024,
        llm_dim=2048,
        stride=2,
    ):
        super().__init__()

        self.downsample = nn.Conv1d(
            audio_dim,
            audio_dim,
            kernel_size=stride,
            stride=stride,
        )

        self.proj = nn.Linear(
            audio_dim,
            llm_dim,
        )

        self.norm = nn.LayerNorm(
            llm_dim,
        )

    def forward(self, x):

        # [B,T,C] -> [B,C,T]
        x = x.transpose(1, 2)

        x = self.downsample(x)

        # [B,C,T] -> [B,T,C]
        x = x.transpose(1, 2)

        x = self.proj(x)

        x = self.norm(x)

        return x


# ============================================================================
# FROZEN QWEN EMBEDDING + 44 TRAINABLE MEITEI ROWS
# ============================================================================

class MeiteiEmbedding(nn.Module):
    """
    Frozen original Qwen embedding.

    Only the 44 Meitei Mayek rows are trainable.
    """

    def __init__(
        self,
        frozen_embedding: nn.Module,
        trainable_rows: torch.Tensor,
        start_id: int,
    ):
        super().__init__()

        self.frozen_embedding = frozen_embedding

        for p in self.frozen_embedding.parameters():
            p.requires_grad_(False)

        self.start_id = start_id

        self.trainable_rows = nn.Parameter(
            trainable_rows.detach().clone()
        )

    def forward(self, input_ids):

        base = self.frozen_embedding(input_ids)

        mask = (
            (input_ids >= self.start_id)
            &
            (
                input_ids
                <
                self.start_id
                + self.trainable_rows.shape[0]
            )
        )

        if not bool(mask.any()):
            return base

        local_ids = (
            input_ids[mask]
            - self.start_id
        )

        replacement = self.trainable_rows[
            local_ids
        ]

        output = base.clone()

        output[mask] = replacement

        return output


# ============================================================================
# FROZEN LM HEAD + 44 TRAINABLE MEITEI OUTPUT ROWS
# ============================================================================

class MeiteiLMHead(nn.Module):
    """
    Keeps the original Qwen LM head frozen.

    Only the 44 Meitei output columns use the shared
    trainable Meitei embedding parameter.

    We intentionally do NOT clone the full LM-head weight.
    """

    def __init__(
        self,
        frozen_lm_head: nn.Module,
        shared_meitei_rows: nn.Parameter,
        start_id: int,
    ):
        super().__init__()

        self.frozen_lm_head = frozen_lm_head

        for p in self.frozen_lm_head.parameters():
            p.requires_grad_(False)

        self.meitei_rows = shared_meitei_rows

        self.start_id = start_id

    def forward(self, hidden_states):

        # Original full-vocabulary logits.
        #
        # The LM-head parameters are frozen, so this does NOT
        # create LM-head parameter gradients.
        base_logits = self.frozen_lm_head(
            hidden_states
        )

        # Trainable 44-token output logits.
        meitei_logits = torch.matmul(
            hidden_states,
            self.meitei_rows.transpose(0, 1),
        )

        left = base_logits[
            ...,
            :self.start_id,
        ]

        right = base_logits[
            ...,
            self.start_id
            + self.meitei_rows.shape[0]:,
        ]

        return torch.cat(
            [
                left,
                meitei_logits,
                right,
            ],
            dim=-1,
        )


# ============================================================================
# TOKENIZER
# ============================================================================

def load_tokenizer():

    print(
        "[TOKENIZER] Loading Epoch-6 tokenizer"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(EPOCH6_TOKENIZER),
        local_files_only=True,
        use_fast=True,
    )

    print(
        "[TOKENIZER] Length:",
        len(tokenizer),
    )

    if len(tokenizer) != FINAL_VOCAB_SIZE:
        raise RuntimeError(
            f"Tokenizer length {len(tokenizer)} "
            f"!= {FINAL_VOCAB_SIZE}"
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


# ============================================================================
# EXACT EPOCH-6 PROMPT
# ============================================================================

def build_prompt_prefix(tokenizer):

    prompt_text = (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{TASK_PREFIX}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids

    print(
        "[PROMPT] Prefix token count:",
        ids.shape[1],
    )

    print(
        "[PROMPT] Exact prefix:"
    )

    print(
        repr(prompt_text)
    )

    return ids


# ============================================================================
# PROJECTOR
# ============================================================================

def load_projector():

    print(
        "[PROJECTOR] Loading Epoch-6 projector"
    )

    projector = AudioProjector(
        audio_dim=AUDIO_DIM,
        llm_dim=HIDDEN_SIZE,
        stride=2,
    )

    state = torch.load(
        EPOCH6_PROJECTOR,
        map_location="cpu",
        weights_only=True,
    )

    projector.load_state_dict(
        state,
        strict=True,
    )

    projector.to(DEVICE)

    projector.train()

    for p in projector.parameters():
        p.requires_grad_(True)

    count = sum(
        p.numel()
        for p in projector.parameters()
    )

    print(
        "[PROJECTOR] Parameters:",
        count,
    )

    return projector


# ============================================================================
# LEGACY MEITEI ROW EXTRACTION
# ============================================================================

def load_legacy_meitei_rows():

    files = list(
        EPOCH6_ADAPTER.glob(
            "*.safetensors"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No Epoch-6 safetensors found."
        )

    # Prefer the actual adapter file.
    files = [
        f for f in files
        if "adapter_model" in f.name
    ]

    if not files:
        raise FileNotFoundError(
            "No Epoch-6 adapter_model*.safetensors found."
        )

    sf = files[0]

    print(
        "[LEGACY] Reading:",
        sf.name,
    )

    embed_key = (
        "base_model.model.model.embed_tokens.weight"
    )

    lm_key = (
        "base_model.model.lm_head.weight"
    )

    with safe_open(
        str(sf),
        framework="pt",
        device="cpu",
    ) as f:

        keys = set(f.keys())

        if embed_key not in keys:
            raise KeyError(
                f"Missing legacy embedding key:\n"
                f"{embed_key}"
            )

        if lm_key not in keys:
            raise KeyError(
                f"Missing legacy LM-head key:\n"
                f"{lm_key}"
            )

        # IMPORTANT: MEITEI_END = 151709 is already the exclusive
        # Python slice endpoint. Do NOT add +1 here.
        embed_rows = f.get_slice(
            embed_key
        )[
            MEITEI_START:
            MEITEI_END
        ].float()

        lm_rows = f.get_slice(
            lm_key
        )[
            MEITEI_START:
            MEITEI_END
        ].float()

    print(
        "[LEGACY] Embed rows:",
        tuple(embed_rows.shape),
    )

    print(
        "[LEGACY] LM-head rows:",
        tuple(lm_rows.shape),
    )

    expected = (
        MEITEI_COUNT,
        HIDDEN_SIZE,
    )

    if tuple(embed_rows.shape) != expected:
        raise RuntimeError(
            f"Unexpected embedding shape: "
            f"{tuple(embed_rows.shape)}"
        )

    if tuple(lm_rows.shape) != expected:
        raise RuntimeError(
            f"Unexpected LM-head shape: "
            f"{tuple(lm_rows.shape)}"
        )

    difference = (
        embed_rows - lm_rows
    ).abs().max().item()

    print(
        "[LEGACY] Max embed/lm-head "
        "row difference:",
        difference,
    )

    if difference > 1e-4:

        print(
            "[LEGACY] WARNING: "
            "embedding and LM-head rows differ."
        )

        print(
            "[LEGACY] Epoch-7 will use "
            "embedding rows as the shared "
            "starting point."
        )

    return embed_rows


# ============================================================================
# QWEN
# ============================================================================


def extract_legacy_lora_cpu():
    """
    Extract the 504 Epoch-6 LoRA tensors before CUDA/Qwen initialization.

    The extraction runs in a separate CPU-only Python process so the
    large 2.4 GB safetensors file is never opened by the CUDA process.
    """

    import subprocess
    import sys
    import tempfile
    import pickle

    sf_candidates = sorted(
        (EPOCH6 / "qwen_lora").glob("adapter_model*.safetensors")
    )

    if not sf_candidates:
        raise FileNotFoundError(
            f"No adapter_model*.safetensors found in "
            f"{EPOCH6 / 'qwen_lora'}"
        )

    sf = sf_candidates[-1]

    print("[PRELOAD] Epoch-6 LoRA:", sf.name)
    print("[PRELOAD] Starting CPU extraction...")

    extractor_code = r"""
import sys
import pickle
from safetensors import safe_open

sf_path = sys.argv[1]
out_path = sys.argv[2]

legacy_tensors = {}

with safe_open(
    sf_path,
    framework="pt",
    device="cpu",
) as f:

    keys = [
        k for k in f.keys()
        if ".lora_A.weight" in k
        or ".lora_B.weight" in k
    ]

    print(
        f"[SUBPROCESS] LoRA keys: {len(keys)}",
        flush=True,
    )

    for i, key in enumerate(keys, 1):

        tensor = f.get_tensor(key)

        if not tensor.isfinite().all():
            raise RuntimeError(
                f"Non-finite tensor: {key}"
            )

        legacy_tensors[key] = tensor.contiguous()

        if i % 50 == 0 or i == len(keys):
            print(
                f"[SUBPROCESS] Extracted "
                f"{i} / {len(keys)}",
                flush=True,
            )

if len(legacy_tensors) != 504:
    raise RuntimeError(
        f"Expected 504 LoRA tensors, "
        f"found {len(legacy_tensors)}"
    )

with open(out_path, "wb") as f:
    pickle.dump(
        legacy_tensors,
        f,
        protocol=pickle.HIGHEST_PROTOCOL,
    )

print(
    "[SUBPROCESS] Extraction complete.",
    flush=True,
)
"""

    with tempfile.TemporaryDirectory(
        prefix="epoch7_lora_preload_",
        dir=str(ROOT / "ASR" / "temp_lora_preload")
    ) as tmp:

        tmp_dir = Path(tmp)

        extractor_path = (
            tmp_dir / "extract_lora_cpu.py"
        )

        cache_path = (
            tmp_dir / "legacy_lora_cpu.pkl"
        )

        extractor_path.write_text(
            extractor_code,
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(extractor_path),
                str(sf),
                str(cache_path),
            ],
            cwd=str(ROOT),
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "CPU LoRA extraction failed with "
                f"exit code {result.returncode}"
            )

        if not cache_path.exists():
            raise RuntimeError(
                "CPU LoRA extraction did not "
                "produce the tensor cache."
            )

        with open(cache_path, "rb") as f:
            legacy_tensors = pickle.load(f)

    if len(legacy_tensors) != 504:
        raise RuntimeError(
            f"Expected 504 cached LoRA tensors, "
            f"found {len(legacy_tensors)}"
        )

    print(
        "[PRELOAD] CPU extraction successful:",
        len(legacy_tensors),
        "tensors",
    )

    return legacy_tensors


def build_qwen(tokenizer, legacy_lora_tensors):

    print(
        "[QWEN] Loading configuration"
    )

    config = AutoConfig.from_pretrained(
        BASE_MODEL,
    )

    if config.vocab_size != BASE_VOCAB_SIZE:
        raise RuntimeError(
            f"Base vocab mismatch: "
            f"{config.vocab_size}"
        )

    if config.hidden_size != HIDDEN_SIZE:
        raise RuntimeError(
            f"Hidden size mismatch: "
            f"{config.hidden_size}"
        )

    if config.tie_word_embeddings is not True:
        raise RuntimeError(
            "Expected Qwen tied word embeddings."
        )

    print(
        "[QWEN] Base vocab:",
        config.vocab_size,
    )

    print(
        "[QWEN] Hidden size:",
        config.hidden_size,
    )

    print(
        "[QWEN] Loading 4-bit NF4 model"
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    cuda_report(
        "After Qwen load"
    )

    # ------------------------------------------------------------
    # Verify original vocabulary
    # ------------------------------------------------------------

    input_embedding = (
        model.get_input_embeddings()
    )

    print(
        "[QWEN] Pre-resize input embedding:",
        tuple(input_embedding.weight.shape),
    )

    if (
        input_embedding.weight.shape[0]
        != BASE_VOCAB_SIZE
    ):
        raise RuntimeError(
            "Loaded Qwen model does not have "
            f"{BASE_VOCAB_SIZE} tokens."
        )

    # ------------------------------------------------------------
    # Resize exactly like Epoch-6
    # ------------------------------------------------------------

    print(
        "[QWEN] Resizing vocabulary:",
        BASE_VOCAB_SIZE,
        "->",
        FINAL_VOCAB_SIZE,
    )

    model.resize_token_embeddings(
        FINAL_VOCAB_SIZE
    )

    print(
        "[QWEN] Post-resize embedding:",
        tuple(
            model.get_input_embeddings()
            .weight.shape
        ),
    )

    # ------------------------------------------------------------
    # Freeze ALL Qwen parameters first
    # ------------------------------------------------------------

    for p in model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------
    # Load legacy Meitei rows
    # ------------------------------------------------------------

    legacy_rows = load_legacy_meitei_rows()

    original_embedding = (
        model.get_input_embeddings()
    )

    embedding_weight = (
        original_embedding.weight
    )

    legacy_rows = legacy_rows.to(
        device=embedding_weight.device,
        dtype=embedding_weight.dtype,
    )

    # ------------------------------------------------------------
    # Replace input embedding
    # ------------------------------------------------------------

    meitei_embedding = MeiteiEmbedding(
        frozen_embedding=original_embedding,
        trainable_rows=legacy_rows,
        start_id=MEITEI_START,
    )

    model.set_input_embeddings(
        meitei_embedding
    )

    # ------------------------------------------------------------
    # Replace LM head with a wrapper.
    #
    # IMPORTANT:
    # Do NOT clone the complete LM-head matrix.
    # ------------------------------------------------------------

    original_lm_head = model.lm_head

    shared_rows = (
        meitei_embedding.trainable_rows
    )

    meitei_lm_head = MeiteiLMHead(
        frozen_lm_head=original_lm_head,
        shared_meitei_rows=shared_rows,
        start_id=MEITEI_START,
    )

    model.lm_head = meitei_lm_head

    # ------------------------------------------------------------
    # Gradient checkpointing
    #
    # We deliberately do NOT call
    # prepare_model_for_kbit_training().
    # ------------------------------------------------------------

    model.config.use_cache = False

    if hasattr(
        model,
        "gradient_checkpointing_enable",
    ):
        model.gradient_checkpointing_enable()

    if hasattr(
        model,
        "enable_input_require_grads",
    ):
        model.enable_input_require_grads()

    # ------------------------------------------------------------
    # LoRA
    # ------------------------------------------------------------

    print(
        "[LORA] Creating Epoch-6-compatible LoRA"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    # ------------------------------------------------------------
    # Load ONLY legacy LoRA tensors.
    # Do NOT load legacy full embedding/lm-head
    # tensors as trainable parameters.
    # ------------------------------------------------------------

    load_cached_lora_weights(model, legacy_lora_tensors)

    return model


# ============================================================================
# LEGACY LORA WEIGHTS
# ============================================================================

def _build_lora_key_map(legacy_keys, current_keys):
    """
    Build a mapping from legacy Epoch-6 LoRA keys to current PEFT keys.

    PEFT 0.20.0 (Epoch-6) stored LoRA weights as:
        base_model.model.model.layers.N.mlp.down_proj.lora_A.weight

    PEFT 0.21.0 (current) uses a ModuleDict keyed by adapter name:
        base_model.model.model.layers.N.mlp.down_proj.lora_A.default.weight

    This function detects which naming scheme is in use for each
    legacy key and returns the mapped current key.
    """

    current_set = set(current_keys)
    key_map = {}  # legacy_key -> current_key
    unmapped = []

    for legacy_key in legacy_keys:

        # Direct match - already compatible (should not happen with
        # different PEFT versions, but handle gracefully).
        if legacy_key in current_set:
            key_map[legacy_key] = legacy_key
            continue

        # PEFT 0.20 -> 0.21 migration:
        # '.lora_A.weight' -> '.lora_A.default.weight'
        # '.lora_B.weight' -> '.lora_B.default.weight'
        mapped = None

        if legacy_key.endswith(".lora_A.weight"):
            candidate = (
                legacy_key[: -len(".weight")]
                + ".default.weight"
            )
            if candidate in current_set:
                mapped = candidate

        elif legacy_key.endswith(".lora_B.weight"):
            candidate = (
                legacy_key[: -len(".weight")]
                + ".default.weight"
            )
            if candidate in current_set:
                mapped = candidate

        if mapped is not None:
            key_map[legacy_key] = mapped
        else:
            unmapped.append(legacy_key)

    return key_map, unmapped


def load_cached_lora_weights(model, legacy_lora_tensors):
    """
    Warm-start Epoch-7 from the Epoch-6 LoRA checkpoint.

    The large safetensors file is extracted in a separate CPU-only
    subprocess. The CUDA-resident Qwen/PEFT model therefore never
    opens the 2.4 GB safetensors file itself.
    """
    import subprocess
    import sys
    import tempfile
    import pickle

    sf_candidates = sorted(
        (EPOCH6 / "qwen_lora").glob("adapter_model*.safetensors")
    )

    if not sf_candidates:
        raise FileNotFoundError(
            f"No adapter_model*.safetensors found in "
            f"{EPOCH6 / 'qwen_lora'}"
        )

    sf = sf_candidates[-1]

    print("[LEGACY] Reading:", sf.name)
    print("[LEGACY] CPU subprocess extraction starting...")

    extractor_code = r"""
import sys
import pickle
from safetensors import safe_open

sf_path = sys.argv[1]
out_path = sys.argv[2]

legacy_tensors = {}

with safe_open(sf_path, framework="pt", device="cpu") as f:
    keys = [
        k for k in f.keys()
        if ".lora_A.weight" in k or ".lora_B.weight" in k
    ]

    print(
        f"[SUBPROCESS] LoRA keys: {len(keys)}",
        flush=True
    )

    for i, key in enumerate(keys, 1):
        tensor = f.get_tensor(key)

        if not tensor.isfinite().all():
            raise RuntimeError(
                f"Non-finite tensor: {key}"
            )

        legacy_tensors[key] = tensor.contiguous()

        if i % 50 == 0 or i == len(keys):
            print(
                f"[SUBPROCESS] Extracted {i} / {len(keys)}",
                flush=True
            )

if len(legacy_tensors) != 504:
    raise RuntimeError(
        f"Expected 504 LoRA tensors, "
        f"found {len(legacy_tensors)}"
    )

with open(out_path, "wb") as f:
    pickle.dump(
        legacy_tensors,
        f,
        protocol=pickle.HIGHEST_PROTOCOL
    )

print(
    "[SUBPROCESS] Extraction complete.",
    flush=True
)
"""

    with tempfile.TemporaryDirectory(
        prefix="epoch7_lora_"
    ) as tmp:

        tmp_dir = Path(tmp)

        extractor_path = (
            tmp_dir / "extract_lora_cpu.py"
        )

        cache_path = (
            tmp_dir / "legacy_lora_cpu.pkl"
        )

        extractor_path.write_text(
            extractor_code,
            encoding="utf-8"
        )

        result = subprocess.run(
            [
                sys.executable,
                str(extractor_path),
                str(sf),
                str(cache_path),
            ],
            cwd=str(ROOT),
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "CPU LoRA extraction subprocess failed "
                f"with exit code {result.returncode}"
            )

        if not cache_path.exists():
            raise RuntimeError(
                "CPU extraction finished without creating "
                "the tensor cache."
            )

        print(
            "[LEGACY] CPU extraction successful."
        )

        with open(cache_path, "rb") as f:
            legacy_tensors = pickle.load(f)

        print(
            "[LEGACY] CPU tensors loaded:",
            len(legacy_tensors)
        )

    loaded = 0
    unmapped = []

    print(
        "[LEGACY] Beginning PEFT tensor copies..."
    )

    for legacy_key, tensor in legacy_tensors.items():

        match = re.match(
            r"base_model\.model\.model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$",
            legacy_key,
        )

        if match is None:
            unmapped.append(legacy_key)
            continue

        layer_idx = int(match.group(1))
        module_path = match.group(2)
        side = match.group(3)

        try:
            module = (
                model
                .base_model
                .model
                .model
                .layers[layer_idx]
            )

            for part in module_path.split("."):
                module = getattr(module, part)

            if side == "A":
                target = (
                    module
                    .lora_A
                    .default
                    .weight
                )
            else:
                target = (
                    module
                    .lora_B
                    .default
                    .weight
                )

        except Exception as exc:
            raise RuntimeError(
                f"Could not map legacy LoRA key: "
                f"{legacy_key}"
            ) from exc

        if tuple(target.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"Shape mismatch for {legacy_key}: "
                f"checkpoint={tuple(tensor.shape)} "
                f"target={tuple(target.shape)}"
            )

        tensor_device = tensor.to(
            device=target.device,
            dtype=target.dtype,
        )

        with torch.no_grad():
            target.copy_(tensor_device)

        loaded += 1

        if loaded % 50 == 0 or loaded == 504:
            print(
                "[LEGACY] Loaded:",
                loaded,
                "/ 504"
            )

    if unmapped:
        raise RuntimeError(
            f"Unmapped legacy LoRA keys: "
            f"{unmapped[:10]}"
        )

    if loaded != 504:
        raise RuntimeError(
            f"Expected 504 loaded LoRA tensors, "
            f"got {loaded}"
        )

    print(
        "[LEGACY] Loaded tensors:",
        loaded
    )

    print(
        "[LEGACY] Legacy LoRA warm-start successful."
    )


def report_trainable(
    model,
    projector,
):

    print()
    print("=" * 70)
    print("TRAINABLE PARAMETER REPORT")
    print("=" * 70)

    qwen_total = 0
    qwen_trainable = 0

    trainable_names = []

    for name, p in model.named_parameters():

        qwen_total += p.numel()

        if p.requires_grad:

            qwen_trainable += p.numel()

            trainable_names.append(
                (
                    name,
                    tuple(p.shape),
                    p.numel(),
                )
            )

    for name, shape, count in trainable_names:

        print(
            f"TRAINABLE QWEN: "
            f"{name:80s} "
            f"{shape} "
            f"{count:,}"
        )

    projector_trainable = sum(
        p.numel()
        for p in projector.parameters()
        if p.requires_grad
    )

    expected_meitei = (
        MEITEI_COUNT
        * HIDDEN_SIZE
    )

    expected_lora = 252 * 2 * 1

    print()
    print(
        "QWEN TOTAL PARAMS     =",
        f"{qwen_total:,}",
    )

    print(
        "QWEN TRAINABLE PARAMS =",
        f"{qwen_trainable:,}",
    )

    print(
        "PROJECTOR TRAINABLE    =",
        f"{projector_trainable:,}",
    )

    print(
        "TOTAL TRAINABLE       =",
        f"{qwen_trainable + projector_trainable:,}",
    )

    print()
    print(
        "EXPECTED MEITEI ROWS  =",
        f"{expected_meitei:,}",
    )

    print(
        "EXPECTED LoRA TENSORS  =",
        "504",
    )

    # qwen_trainable includes both Meitei rows AND LoRA parameters.
    # So the threshold is just the Meitei rows alone.
    if qwen_trainable < expected_meitei:
        raise RuntimeError(
            "Unexpectedly low Qwen trainable count."
        )


# ============================================================================
# VERIFY TRAINABLE PARAMETERS
# ============================================================================

def verify_trainable_structure(model):

    print()
    print(
        "[VERIFY] Checking trainable structure..."
    )

    bad = []

    meitei_params = []

    lora_params = []

    for name, p in model.named_parameters():

        if not p.requires_grad:
            continue

        if (
            "trainable_rows" in name
            or "meitei_rows" in name
        ):
            meitei_params.append(
                (name, p)
            )
            continue

        if (
            ".lora_A." in name
            or ".lora_B." in name
        ):
            lora_params.append(
                (name, p)
            )
            continue

        bad.append(
            (name, p)
        )

    print(
        "[VERIFY] Meitei trainable parameters:",
        len(meitei_params),
    )

    for name, p in meitei_params:

        print(
            "   MEITEI:",
            name,
            tuple(p.shape),
            f"{p.numel():,}",
        )

    print(
        "[VERIFY] LoRA trainable tensors:",
        len(lora_params),
    )

    print(
        "[VERIFY] Unexpected trainable tensors:",
        len(bad),
    )

    if bad:

        for name, p in bad[:30]:
            print(
                "   BAD:",
                name,
                tuple(p.shape),
                f"{p.numel():,}",
            )

        raise RuntimeError(
            "Unexpected trainable Qwen parameters "
            "detected."
        )

    # Verify exact Meitei parameter size.
    meitei_count = sum(
        p.numel()
        for _, p in meitei_params
    )

    expected_meitei = (
        MEITEI_COUNT
        * HIDDEN_SIZE
    )

    if meitei_count != expected_meitei:
        raise RuntimeError(
            "Unexpected Meitei trainable "
            f"parameter count: {meitei_count}"
        )

    # 252 LoRA A + 252 LoRA B.
    if len(lora_params) != 504:
        raise RuntimeError(
            "Expected 504 LoRA trainable tensors, "
            f"found {len(lora_params)}."
        )

    print(
        "[VERIFY] Full-vocabulary embedding/lm-head: PASS"
    )

    print(
        "[VERIFY] Meitei-only embedding rows: PASS"
    )

    print(
        "[VERIFY] LoRA structure: PASS"
    )


# ============================================================================
# CACHE INDEX
# ============================================================================

def build_cache_index():

    index = {}

    if not CACHE_DIR.exists():

        print(
            "[CACHE] Directory does not exist."
        )

        return index

    metadata_files = list(
        CACHE_DIR.glob("*.json")
    )

    print(
        "[CACHE] Metadata files:",
        len(metadata_files),
    )

    for meta_path in metadata_files:

        try:

            md = json.loads(
                meta_path.read_text(
                    encoding="utf-8"
                )
            )

            dataset = md.get(
                "dataset"
            )

            record_index = md.get(
                "record_index"
            )

            if dataset is None:
                continue

            if record_index is None:
                continue

            pt_path = (
                meta_path.with_suffix(".pt")
            )

            if not pt_path.exists():
                continue

            key = (
                str(dataset),
                int(record_index),
            )

            index[key] = pt_path

        except Exception:
            continue

    print(
        "[CACHE] Usable cached features:",
        len(index),
    )

    return index


# ============================================================================
# ENCODER
# ============================================================================

def build_encoder():

    from transformers import (
        Wav2Vec2BertProcessor,
        Wav2Vec2BertModel,
    )

    print(
        "[ENCODER] Loading W2V-BERT 2.0"
    )

    processor = (
        Wav2Vec2BertProcessor.from_pretrained(
            ENCODER_MODEL
        )
    )

    encoder = (
        Wav2Vec2BertModel.from_pretrained(
            ENCODER_MODEL,
            dtype=(
                torch.bfloat16
                if DEVICE == "cuda"
                else torch.float32
            ),
        )
    )

    encoder.to(DEVICE)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad_(False)

    print(
        "[ENCODER] Hidden size:",
        encoder.config.hidden_size,
    )

    return processor, encoder


# ============================================================================
# AUDIO
# ============================================================================

def load_audio_16k(
    path,
):

    wav_np, sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=False,
    )

    wav = torch.from_numpy(
        wav_np
    )

    if wav.ndim > 1:
        wav = wav.mean(
            dim=-1
        )

    if sr != 16000:

        wav = torchaudio.functional.resample(
            wav,
            sr,
            16000,
        )

    max_samples = int(
        MAX_AUDIO_SECONDS
        * 16000
    )

    wav = wav[:max_samples]

    return wav.numpy()


@torch.no_grad()
def encode_audio(
    audio_path,
    processor,
    encoder,
):

    audio = load_audio_16k(
        audio_path
    )

    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    outputs = encoder(
        **inputs
    )

    return outputs.last_hidden_state


# ============================================================================
# FEATURE LOADER
# ============================================================================

def load_cached_feature(
    cache_index,
    dataset_name,
    record_index,
):

    path = cache_index.get(
        (
            str(dataset_name),
            int(record_index),
        )
    )

    if path is None:
        return None

    try:

        feature = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        if isinstance(feature, dict):

            if "features" in feature:
                feature = feature["features"]

            elif "last_hidden_state" in feature:
                feature = feature[
                    "last_hidden_state"
                ]

        if not isinstance(
            feature,
            torch.Tensor,
        ):
            return None

        return feature

    except Exception:

        return None


# ============================================================================
# DATASET
# ============================================================================

def resolve_audio_path(
    path,
):
    """Resolve dataset audio paths across restored project locations."""

    p = Path(str(path).strip())

    # 1. Original absolute path still exists.
    if p.is_absolute() and p.exists():
        return p

    # 2. Recover everything after the ASR directory.
    #
    # Old:
    # C:\Users\...\ASR\manipuri_1k_project\sample_298.wav
    #
    # Current:
    # ROOT\ASR\manipuri_1k_project\sample_298.wav

    parts = list(p.parts)

    try:
        asr_index = next(
            i
            for i, part in enumerate(parts)
            if str(part).lower() == "asr"
        )

        relative_from_asr = Path(
            *parts[asr_index + 1:]
        )

        candidate = ASR / relative_from_asr

        if candidate.exists():
            return candidate

    except StopIteration:
        pass

    # 3. Try normal project-relative paths.
    candidate = ROOT / p

    if candidate.exists():
        return candidate

    candidate = ASR / p

    if candidate.exists():
        return candidate

    # 4. Final fallback: search by exact filename.
    filename = p.name

    if filename:
        matches = list(
            ASR.rglob(filename)
        )

        if matches:
            return matches[0]

    # Keep the unresolved path for diagnostics.
    return p

def load_datasets():

    train_df = pd.read_csv(
        TRAIN_CSV,
        encoding="utf-8-sig",
    )

    val_df = pd.read_csv(
        VAL_CSV,
        encoding="utf-8-sig",
    )

    test_df = pd.read_csv(
        TEST_CSV,
        encoding="utf-8-sig",
    )

    print(
        "[DATA] train:",
        len(train_df),
    )

    print(
        "[DATA] val:",
        len(val_df),
    )

    print(
        "[DATA] test:",
        len(test_df),
    )

    train_df = train_df[
        train_df["audio_path"].apply(
            lambda p:
            resolve_audio_path(p).exists()
        )
    ].reset_index(
        drop=True
    )

    val_df = val_df[
        val_df["audio_path"].apply(
            lambda p:
            resolve_audio_path(p).exists()
        )
    ].reset_index(
        drop=True
    )

    print(
        "[DATA] after audio existence check:",
        f"train={len(train_df)}",
        f"val={len(val_df)}",
    )

    return (
        train_df,
        val_df,
        test_df,
    )


# ============================================================================
# BUILD SINGLE EXAMPLE
# ============================================================================

def build_single_example(
    row,
    tokenizer,
    task_prefix_ids,
    qwen,
    projector,
    cache_index=None,
    encoder_processor=None,
    encoder=None,
):

    audio_path = resolve_audio_path(
        row["audio_path"]
    )

    # ------------------------------------------------------------
    # Audio features
    # ------------------------------------------------------------

    feature = None

    dataset_name = row.get(
        "source",
        None,
    )

    # The cache uses explicit dataset/index metadata.
    # We only use it when those fields are available.
    if (
        cache_index is not None
        and "record_index" in row.index
        and dataset_name is not None
    ):

        feature = load_cached_feature(
            cache_index,
            dataset_name,
            int(row.name),
        )

    if feature is not None:

        h = feature.to(
            DEVICE,
            dtype=(
                torch.bfloat16
                if DEVICE == "cuda"
                else torch.float32
            ),
        )

        print_once = False

    else:

        if encoder is None:
            raise RuntimeError(
                "No cached feature and encoder "
                "was not initialized."
            )

        h = encode_audio(
            audio_path,
            encoder_processor,
            encoder,
        )

    # ------------------------------------------------------------
    # Projector
    # ------------------------------------------------------------

    audio_emb = projector(
        h.float()
    )

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    target_text = (
        str(row["text"]).strip()
        + "<|im_end|>"
    )

    target_ids = tokenizer(
        target_text,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_TEXT_TOKENS,
        return_tensors="pt",
    ).input_ids.to(DEVICE)

    # ------------------------------------------------------------
    # Prompt + target embeddings
    # ------------------------------------------------------------

    prefix_emb = (
        qwen.get_input_embeddings()(
            task_prefix_ids
        )
    )

    target_emb = (
        qwen.get_input_embeddings()(
            target_ids
        )
    )

    inputs_embeds = torch.cat(
        [
            prefix_emb.to(
                target_emb.dtype
            ),

            audio_emb.to(
                target_emb.dtype
            ),

            target_emb,
        ],
        dim=1,
    )

    # ------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------

    ignore_len = (
        prefix_emb.shape[1]
        + audio_emb.shape[1]
    )

    labels = torch.cat(
        [
            torch.full(
                (
                    1,
                    ignore_len,
                ),
                -100,
                dtype=torch.long,
                device=DEVICE,
            ),

            target_ids,
        ],
        dim=1,
    )

    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=DEVICE,
    )

    return (
        inputs_embeds,
        attention_mask,
        labels,
    )


# ============================================================================
# TRAIN STEP
# ============================================================================

def train_one_row(
    row,
    tokenizer,
    task_prefix_ids,
    qwen,
    projector,
    cache_index,
    encoder_processor,
    encoder,
):

    qwen.train()
    projector.train()

    (
        inputs_embeds,
        attention_mask,
        labels,
    ) = build_single_example(
        row,
        tokenizer,
        task_prefix_ids,
        qwen,
        projector,
        cache_index,
        encoder_processor,
        encoder,
    )

    out = qwen(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
    )

    return out.loss


# ============================================================================
# SAFE STEP
# ============================================================================

def run_step_safely(
    fn,
    row,
    message,
):

    try:

        loss = fn(row)

        if (
            isinstance(loss, torch.Tensor)
            and not torch.isfinite(loss).all()
        ):

            print(
                "[SKIP] Non-finite loss:",
                message,
            )

            return None

        return loss

    except torch.cuda.OutOfMemoryError:

        print(
            "[CUDA OOM]",
            message,
            "Skipping example.",
        )

        clear_cuda()

        return None

    except Exception as e:

        print(
            "[ERROR]",
            message,
            ":",
            repr(e),
        )

        return None


# ============================================================================
# GREEDY INFERENCE
# ============================================================================

@torch.no_grad()
def greedy_transcribe(
    row,
    tokenizer,
    task_prefix_ids,
    qwen,
    projector,
    encoder_processor,
    encoder,
    max_new_tokens=MAX_TEXT_TOKENS,
):

    qwen.eval()
    projector.eval()

    audio_path = resolve_audio_path(
        row["audio_path"]
    )

    h = encode_audio(
        audio_path,
        encoder_processor,
        encoder,
    )

    audio_emb = projector(
        h.float()
    )

    prefix_emb = (
        qwen.get_input_embeddings()(
            task_prefix_ids
        )
    )

    prompt_emb = torch.cat(
        [
            prefix_emb.to(
                audio_emb.dtype
            ),
            audio_emb,
        ],
        dim=1,
    )

    mask = torch.ones(
        prompt_emb.shape[:2],
        dtype=torch.long,
        device=DEVICE,
    )

    im_end_id = tokenizer.convert_tokens_to_ids(
        "<|im_end|>"
    )

    stop_ids = [
        x
        for x in [
            tokenizer.eos_token_id,
            im_end_id,
        ]
        if x is not None
    ]

    stop_ids = list(
        sorted(
            set(stop_ids)
        )
    )

    out_ids = qwen.generate(
        inputs_embeds=prompt_emb,
        attention_mask=mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=stop_ids,
        pad_token_id=(
            tokenizer.pad_token_id
            or tokenizer.eos_token_id
        ),
    )

    return tokenizer.decode(
        out_ids[0],
        skip_special_tokens=True,
    ).strip()


# ============================================================================
# DRY RUN
# ============================================================================

def dry_run():

    print()
    print("=" * 70)
    print("EPOCH-7 RECOVERY DRY RUN")
    print("=" * 70)

    print(
        "[CHECK] ROOT:",
        ROOT,
    )

    print(
        "[CHECK] Epoch-6:",
        EPOCH6,
    )

    if not EPOCH6.exists():
        raise FileNotFoundError(
            EPOCH6
        )

    if not EPOCH6_ADAPTER.exists():
        raise FileNotFoundError(
            EPOCH6_ADAPTER
        )

    if not EPOCH6_PROJECTOR.exists():
        raise FileNotFoundError(
            EPOCH6_PROJECTOR
        )

    # ------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------

    tokenizer = load_tokenizer()

    # Epoch-6 LoRA CPU preload
    legacy_lora_tensors = (
        extract_legacy_lora_cpu()
    )


    # ------------------------------------------------------------
    # Projector
    # ------------------------------------------------------------

    task_prefix_ids = (
        build_prompt_prefix(
            tokenizer
        )
    )

    # ------------------------------------------------------------
    # Qwen
    # ------------------------------------------------------------

    qwen = build_qwen(
        tokenizer,
        legacy_lora_tensors,
    )

    # ------------------------------------------------------------
    # Parameter report
    # ------------------------------------------------------------

    report_trainable(
        qwen,
        projector,
    )

    verify_trainable_structure(
        qwen
    )

    # ------------------------------------------------------------
    # Verify shared parameter
    # ------------------------------------------------------------

    base_model = (
        qwen.get_base_model()
    )

    embedding_module = (
        base_model.get_input_embeddings()
    )

    lm_head = (
        base_model.lm_head
    )

    if not isinstance(
        embedding_module,
        MeiteiEmbedding,
    ):
        raise RuntimeError(
            "Input embedding is not "
            "MeiteiEmbedding."
        )

    if not isinstance(
        lm_head,
        MeiteiLMHead,
    ):
        raise RuntimeError(
            "LM head is not "
            "MeiteiLMHead."
        )

    if (
        lm_head.meitei_rows
        is not
        embedding_module.trainable_rows
    ):
        raise RuntimeError(
            "Embedding and LM head do not "
            "share the same Meitei Parameter."
        )

    shared = (
        embedding_module.trainable_rows
    )

    print(
        "[DRY RUN] Shared Meitei parameter: PASS"
    )

    print(
        "[DRY RUN] Shared shape:",
        tuple(shared.shape),
    )

    # ------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------

    if (
        qwen.config.vocab_size
        != FINAL_VOCAB_SIZE
    ):
        raise RuntimeError(
            f"Qwen vocab size "
            f"{qwen.config.vocab_size} "
            f"!= {FINAL_VOCAB_SIZE}"
        )

    print(
        "[DRY RUN] Vocabulary size: PASS"
    )

    # ------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------

    cache_index = (
        build_cache_index()
    )

    # ------------------------------------------------------------
    # Memory
    # ------------------------------------------------------------

    cuda_report(
        "Final dry-run memory"
    )

    print()
    print("=" * 70)
    print("DRY RUN RESULT: PASS")
    print("=" * 70)

    print(
        "No optimizer was created."
    )

    print(
        "No training step was executed."
    )

    print(
        "No Epoch-6 file was modified."
    )

    print(
        "Epoch-6 LoRA weights were loaded "
        "read-only into memory."
    )

    print(
        "Epoch-6 projector was loaded "
        "read-only into memory."
    )


# ============================================================================
# ONE-EXAMPLE FORWARD TEST
# ============================================================================

def one_example_test(
    train_df,
    tokenizer,
    task_prefix_ids,
    qwen,
    projector,
    cache_index,
    encoder_processor,
    encoder,
):

    print()
    print("=" * 70)
    print("STAGE 1: ONE-EXAMPLE FORWARD/BACKWARD")
    print("=" * 70)

    row = train_df.iloc[0]

    qwen.train()
    projector.train()

    optimizer = AdamW(
        [
            {
                "params": projector.parameters(),
                "lr": PROJECTOR_LR,
            },
            {
                "params": [
                    p
                    for p in qwen.parameters()
                    if p.requires_grad
                ],
                "lr": QWEN_LR,
            },
        ],
        weight_decay=0.01,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss = run_step_safely(
        lambda r:
        train_one_row(
            r,
            tokenizer,
            task_prefix_ids,
            qwen,
            projector,
            cache_index,
            encoder_processor,
            encoder,
        ),
        row,
        "one-example test",
    )

    if loss is None:
        raise RuntimeError(
            "One-example forward failed."
        )

    print(
        "[TEST] Loss:",
        float(loss.detach()),
    )

    loss.backward()

    print(
        "[TEST] Backward pass: PASS"
    )

    grad_count = 0

    for name, p in qwen.named_parameters():

        if p.requires_grad:

            if p.grad is not None:

                grad_count += 1

                if not torch.isfinite(
                    p.grad
                ).all():

                    raise RuntimeError(
                        f"Non-finite gradient: {name}"
                    )

    for name, p in projector.named_parameters():

        if p.grad is not None:

            grad_count += 1

            if not torch.isfinite(
                p.grad
            ).all():

                raise RuntimeError(
                    f"Non-finite projector gradient: {name}"
                )

    print(
        "[TEST] Parameters with gradients:",
        grad_count,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    clear_cuda()

    print(
        "[TEST] One-example test: PASS"
    )


# ============================================================================
# OVERFIT TEST
# ============================================================================

def overfit_test(
    train_df,
    tokenizer,
    task_prefix_ids,
    qwen,
    projector,
    cache_index,
    encoder_processor,
    encoder,
):

    print()
    print("=" * 70)
    print(
        f"STAGE 2: {OVERFIT_SIZE}-EXAMPLE OVERFIT TEST"
    )
    print("=" * 70)

    tiny = (
        train_df
        .head(OVERFIT_SIZE)
        .copy()
        .reset_index(drop=True)
    )

    optimizer = AdamW(
        [
            {
                "params": projector.parameters(),
                "lr": PROJECTOR_LR,
            },
            {
                "params": [
                    p
                    for p in qwen.parameters()
                    if p.requires_grad
                ],
                "lr": QWEN_LR,
            },
        ],
        weight_decay=0.01,
    )

    history = []

    for epoch in range(
        OVERFIT_EPOCHS
    ):

        qwen.train()
        projector.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running = 0.0
        steps = 0

        for i, row in tqdm(
            tiny.iterrows(),
            total=len(tiny),
            desc=f"Overfit epoch {epoch + 1}",
        ):

            loss = run_step_safely(
                lambda r:
                train_one_row(
                    r,
                    tokenizer,
                    task_prefix_ids,
                    qwen,
                    projector,
                    cache_index,
                    encoder_processor,
                    encoder,
                ),
                row,
                "overfit test",
            )

            if loss is None:
                continue

            loss.backward()

            if (
                (i + 1) % GRAD_ACCUM == 0
                or
                (i + 1) == len(tiny)
            ):

                parameters = (
                    list(
                        projector.parameters()
                    )
                    +
                    [
                        p
                        for p in qwen.parameters()
                        if p.requires_grad
                    ]
                )

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        parameters,
                        1.0,
                    )
                )

                if torch.isfinite(
                    grad_norm
                ):

                    optimizer.step()

                else:

                    print(
                        "[OVERFIT] "
                        "Non-finite gradient norm."
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

            running += float(
                loss.detach()
            )

            steps += 1

        average = (
            running
            /
            max(steps, 1)
        )

        history.append(
            average
        )

        print(
            f"Overfit epoch {epoch + 1}: "
            f"avg loss = {average:.4f}"
        )

        clear_cuda()

    print()
    print(
        "Overfit loss history:",
        history,
    )

    if (
        len(history) >= 2
        and history[-1] >= history[0]
    ):

        print()
        print(
            "WARNING: loss did not decrease."
        )

        return False

    print()
    print(
        "OVERFIT TEST: PASS"
    )

    return True


# ============================================================================
# EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate(
    df,
    n,
    tokenizer,
    task_prefix_ids,
    qwen,
    projector,
    encoder_processor,
    encoder,
):

    subset = df.head(n)

    refs = []
    hyps = []

    for idx, (_, row) in enumerate(
        tqdm(
            subset.iterrows(),
            total=len(subset),
            desc="Validation",
        )
    ):

        try:

            prediction = (
                greedy_transcribe(
                    row,
                    tokenizer,
                    task_prefix_ids,
                    qwen,
                    projector,
                    encoder_processor,
                    encoder,
                )
            )

        except Exception as e:

            print(
                "[EVAL ERROR]",
                repr(e),
            )

            continue

        reference = str(
            row["text"]
        )

        refs.append(
            reference
        )

        hyps.append(
            prediction
        )

        if idx < 3:

            print()
            print(
                "REF:",
                reference,
            )

            print(
                "HYP:",
                prediction,
            )

    if not refs:

        return (
            float("nan"),
            float("nan"),
        )

    return (
        cer(
            refs,
            hyps,
        ),
        wer(
            refs,
            hyps,
        ),
    )


# ============================================================================
# FULL TRAINING
# ============================================================================

def full_training(
    train_df,
    val_df,
    tokenizer,
    task_prefix_ids,
    qwen,
    projector,
    cache_index,
    encoder_processor,
    encoder,
    num_epochs,
):

    optimizer = AdamW(
        [
            {
                "params": projector.parameters(),
                "lr": PROJECTOR_LR,
            },
            {
                "params": [
                    p
                    for p in qwen.parameters()
                    if p.requires_grad
                ],
                "lr": QWEN_LR,
            },
        ],
        weight_decay=0.01,
    )

    best_wer = float("inf")
    patience = 0

    epoch_log = []

    for epoch in range(
        num_epochs
    ):

        print()
        print("=" * 70)
        print(
            f"EPOCH {epoch + 1}/{num_epochs}"
        )
        print("=" * 70)

        shuffled = (
            train_df
            .sample(
                frac=1.0,
                random_state=SEED + epoch,
            )
            .reset_index(drop=True)
        )

        qwen.train()
        projector.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running = 0.0
        steps = 0

        for i, row in tqdm(
            shuffled.iterrows(),
            total=len(shuffled),
            desc=f"Epoch {epoch + 1}",
        ):

            loss = run_step_safely(
                lambda r:
                train_one_row(
                    r,
                    tokenizer,
                    task_prefix_ids,
                    qwen,
                    projector,
                    cache_index,
                    encoder_processor,
                    encoder,
                ),
                row,
                "training",
            )

            if loss is None:
                continue

            loss.backward()

            if (
                (i + 1) % GRAD_ACCUM == 0
                or
                (i + 1) == len(shuffled)
            ):

                parameters = (
                    list(
                        projector.parameters()
                    )
                    +
                    [
                        p
                        for p in qwen.parameters()
                        if p.requires_grad
                    ]
                )

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        parameters,
                        1.0,
                    )
                )

                if torch.isfinite(
                    grad_norm
                ):

                    optimizer.step()

                else:

                    print(
                        "[TRAIN] "
                        "Non-finite gradient norm."
                    )

                optimizer.zero_grad(
                    set_to_none=True
                )

            running += float(
                loss.detach()
            )

            steps += 1

        train_loss = (
            running
            /
            max(steps, 1)
        )

        # --------------------------------------------------------
        # Validation
        # --------------------------------------------------------

        val_cer, val_wer = evaluate(
            val_df,
            min(
                VAL_SUBSET,
                len(val_df),
            ),
            tokenizer,
            task_prefix_ids,
            qwen,
            projector,
            encoder_processor,
            encoder,
        )

        print()
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f} "
            f"CER={val_cer:.4f} "
            f"WER={val_wer:.4f}"
        )

        epoch_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "CER": val_cer,
                "WER": val_wer,
            }
        )

        # --------------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------------

        epoch_dir = (
            OUTPUT_ROOT
            / f"epoch_{epoch + 1}"
        )

        epoch_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            projector.state_dict(),
            epoch_dir
            / "audio_projector.pt",
        )

        qwen.save_pretrained(
            str(
                epoch_dir
                / "qwen_lora"
            )
        )

        tokenizer.save_pretrained(
            str(
                epoch_dir
                / "tokenizer"
            )
        )

        torch.save(
            optimizer.state_dict(),
            epoch_dir
            / "optimizer.pt",
        )

        trainer_state = {
            "epoch": epoch + 1,
            "best_val_wer": best_wer,
            "patience_counter": patience,
            "train_loss": train_loss,
            "val_cer": val_cer,
            "val_wer": val_wer,
        }

        (
            epoch_dir
            / "trainer_state.json"
        ).write_text(
            json.dumps(
                trainer_state,
                indent=2,
            ),
            encoding="utf-8",
        )

        # --------------------------------------------------------
        # Best model
        # --------------------------------------------------------

        if (
            not math.isnan(val_wer)
            and val_wer < best_wer
        ):

            best_wer = val_wer
            patience = 0

            best_dir = (
                OUTPUT_ROOT
                / "best_model"
            )

            if best_dir.exists():
                shutil.rmtree(
                    best_dir
                )

            shutil.copytree(
                epoch_dir,
                best_dir,
            )

            print(
                "[CHECKPOINT] New best model."
            )

        else:

            patience += 1

            print(
                "[CHECKPOINT] "
                f"No improvement. "
                f"Patience "
                f"{patience}/{EARLY_STOPPING_PATIENCE}"
            )

            if (
                patience
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "[TRAIN] "
                    "Early stopping."
                )

                break

        clear_cuda()

    log_path = (
        OUTPUT_ROOT
        / "training_log.csv"
    )

    pd.DataFrame(
        epoch_log
    ).to_csv(
        log_path,
        index=False,
    )

    print()
    print(
        "Training complete."
    )

    print(
        "Training log:",
        log_path,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model construction only.",
    )

    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Run one-example test + 20-example overfit test.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run full Epoch-7 recovery training.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of Epoch-7 epochs.",
    )

    args = parser.parse_args()

    if not any(
        [
            args.dry_run,
            args.overfit,
            args.train,
        ]
    ):

        parser.print_help()

        return

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("EPOCH-7 RECOVERY")
    print("=" * 70)

    print(
        "[CHECK] Device:",
        DEVICE,
    )

    print(
        "[CHECK] ROOT:",
        ROOT,
    )

    print(
        "[CHECK] Epoch-6:",
        EPOCH6,
    )

    print(
        "[CHECK] HF_HOME:",
        os.environ.get(
            "HF_HOME",
            "<default>",
        ),
    )

    print(
        "[CHECK] HF_HUB_CACHE:",
        os.environ.get(
            "HUGGINGFACE_HUB_CACHE",
            "<default>",
        ),
    )

    if not EPOCH6.exists():
        raise FileNotFoundError(
            EPOCH6
        )

    if DEVICE != "cuda":
        raise RuntimeError(
            "CUDA GPU is required."
        )

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    (
        train_df,
        val_df,
        test_df,
    ) = load_datasets()

    # ------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------

    tokenizer = load_tokenizer()

    # ------------------------------------------------------------
    # Projector
    # ------------------------------------------------------------

    # Epoch-6 LoRA CPU preload
    legacy_lora_tensors = (
        extract_legacy_lora_cpu()
    )

    projector = load_projector()

    # ------------------------------------------------------------
    # Prompt
    # ------------------------------------------------------------

    task_prefix_ids = (
        build_prompt_prefix(
            tokenizer
        )
    )

    # ------------------------------------------------------------
    # Qwen
    # ------------------------------------------------------------

    qwen = build_qwen(
        tokenizer,
        legacy_lora_tensors,
    )

    # ------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------

    report_trainable(
        qwen,
        projector,
    )

    verify_trainable_structure(
        qwen
    )

    # ------------------------------------------------------------
    # Dry run ends here
    # ------------------------------------------------------------

    if args.dry_run:

        cuda_report(
            "Dry-run final"
        )

        print()
        print("=" * 70)
        print("DRY RUN RESULT: PASS")
        print("=" * 70)

        print(
            "No training step executed."
        )

        print(
            "No optimizer created."
        )

        print(
            "Epoch-6 files were not modified."
        )

        return

    # ------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------

    cache_index = (
        build_cache_index()
    )

    # ------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------

    encoder_processor, encoder = (
        build_encoder()
    )

    cuda_report(
        "After encoder load"
    )

    # ------------------------------------------------------------
    # Overfit / sanity
    # ------------------------------------------------------------

    one_example_test(
        train_df,
        tokenizer,
        task_prefix_ids,
        qwen,
        projector,
        cache_index,
        encoder_processor,
        encoder,
    )

    if args.overfit:

        passed = overfit_test(
            train_df,
            tokenizer,
            task_prefix_ids,
            qwen,
            projector,
            cache_index,
            encoder_processor,
            encoder,
        )

        if not passed:

            print()
            print(
                "OVERFIT TEST DID NOT PASS."
            )

            print(
                "Full training was NOT started."
            )

            return

        print()
        print(
            "Overfit test passed."
        )

        print(
            "No full training requested."
        )

        return

    # ------------------------------------------------------------
    # Full training
    # ------------------------------------------------------------

    if args.train:

        print()
        print("=" * 70)
        print("FULL EPOCH-7 RECOVERY TRAINING")
        print("=" * 70)

        print(
            "Epochs:",
            args.epochs,
        )

        print(
            "Epoch-6 optimizer state: NOT loaded"
        )

        print(
            "Epoch-6 LoRA: warm-started"
        )

        print(
            "Epoch-6 projector: warm-started"
        )

        response = input(
            "\nStart full training? [y/N]: "
        ).strip().lower()

        if response != "y":

            print(
                "Training cancelled."
            )

            return

        full_training(
            train_df,
            val_df,
            tokenizer,
            task_prefix_ids,
            qwen,
            projector,
            cache_index,
            encoder_processor,
            encoder,
            args.epochs,
        )


if __name__ == "__main__":
    main()




