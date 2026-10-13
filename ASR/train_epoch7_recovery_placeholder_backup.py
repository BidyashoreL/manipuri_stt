"""
Epoch-7 Recovery Trainer
========================

Purpose:
    Continue from Epoch-6 learned weights while fixing the
    catastrophic full-vocabulary embedding/lm_head training issue.

IMPORTANT:
    - train.py is untouched.
    - Epoch-6 checkpoint is read-only.
    - Legacy optimizer is NOT resumed.
    - Only the 44 Meitei Mayek rows are trainable.
    - Qwen word embeddings remain tied.
    - Epoch-6 projector architecture is preserved.
    - Actual Epoch-6 objective is preserved:
          ChatML prefix -> audio embeddings -> target embeddings

Usage:

    DRY RUN:
        python .\ASR\train_epoch7_recovery.py --dry-run

    TRAIN:
        python .\ASR\train_epoch7_recovery.py --train

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from safetensors import safe_open


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"E:\manipuri_stt1\manipuri_stt4"
    r"\manipuri_stt-20260816T162024Z-1-001"
    r"\manipuri_stt"
)

ASR = ROOT / "ASR"

EPOCH6 = ASR / "checkpoints" / "epoch_6_broken_objective"
EPOCH6_ADAPTER = EPOCH6 / "qwen_lora"
EPOCH6_PROJECTOR = EPOCH6 / "audio_projector.pt"
EPOCH6_TOKENIZER = EPOCH6 / "tokenizer"

TRAIN_CSV = ASR / "train.csv"
VAL_CSV = ASR / "val.csv"
TEST_CSV = ASR / "test.csv"

CACHE_DIR = ASR / "v2" / "cache" / "features"

OUTPUT_ROOT = ASR / "checkpoints" / "epoch_7_recovery"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONSTANTS VERIFIED BY PREFLIGHT
# ============================================================

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"

ENCODER_MODEL = "facebook/w2v-bert-2.0"

BASE_VOCAB_SIZE = 151936
FINAL_VOCAB_SIZE = 151709
HIDDEN_SIZE = 2048

MEITEI_START = 151665
MEITEI_END = 151708
MEITEI_COUNT = 44

AUDIO_DIM = 1024

SYSTEM_PROMPT = (
    "You are a helpful speech recognition assistant."
)

TASK_PREFIX = (
    "<|im_start|>user\n"
)

# This will be verified against the saved tokenizer/template.
# The original train.py builds the complete prefix:
#
# <|im_start|>system
# You are a helpful speech recognition assistant.<|im_end|>
# <|im_start|>user
# <TASK_PREFIX><|im_end|>
# <|im_start|>assistant
#
# TASK_PREFIX is loaded from the original training configuration
# if present; otherwise the known task prefix is used.


# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    COMPUTE_DTYPE = torch.bfloat16
else:
    COMPUTE_DTYPE = torch.float32


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# AUDIO PROJECTOR
# EXACT EPOCH-6 ARCHITECTURE
# ============================================================

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
            llm_dim
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


# ============================================================
# TOKEN-ONLY TRAINABLE EMBEDDING
# ============================================================

class MeiteiEmbedding(nn.Module):
    """
    Frozen full Qwen embedding + trainable 44-row overlay.

    No full-vocabulary gradient tensor is created.
    """

    def __init__(
        self,
        frozen_embedding: nn.Embedding,
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
            & (
                input_ids
                < self.start_id + self.trainable_rows.shape[0]
            )
        )

        if not mask.any():
            return base

        local_ids = (
            input_ids[mask] - self.start_id
        )

        replacement = self.trainable_rows[
            local_ids
        ]

        output = base.clone()

        output[mask] = replacement

        return output


# ============================================================
# TOKEN-ONLY TRAINABLE LM HEAD
# ============================================================

class MeiteiLMHead(nn.Module):
    """
    Frozen Qwen output projection with a trainable 44-column
    Meitei overlay.

    The trainable rows are shared with MeiteiEmbedding.
    """

    def __init__(
        self,
        frozen_weight: torch.Tensor,
        shared_meitei_rows: nn.Parameter,
        start_id: int,
    ):
        super().__init__()

        self.register_buffer(
            "frozen_weight",
            frozen_weight.detach(),
            persistent=False,
        )

        self.meitei_rows = shared_meitei_rows

        self.start_id = start_id

    def forward(self, hidden_states):

        # Base vocabulary logits.
        base_logits = F.linear(
            hidden_states,
            self.frozen_weight,
        )

        # Only compute the 44 trainable output columns.
        meitei_logits = torch.matmul(
            hidden_states,
            self.meitei_rows.transpose(0, 1),
        )

        left = base_logits[
            ..., :self.start_id
        ]

        right = base_logits[
            ...,
            self.start_id + self.meitei_rows.shape[0]:,
        ]

        return torch.cat(
            [
                left,
                meitei_logits,
                right,
            ],
            dim=-1,
        )


# ============================================================
# SAFE LEGACY ROW EXTRACTION
# ============================================================

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

    sf = files[0]

    print(
        f"[LEGACY] Reading {sf.name}"
    )

    with safe_open(
        str(sf),
        framework="pt",
        device="cpu",
    ) as f:

        embed_key = (
            "base_model.model.model.embed_tokens.weight"
        )

        lm_key = (
            "base_model.model.lm_head.weight"
        )

        if embed_key not in f.keys():
            raise KeyError(
                f"Missing {embed_key}"
            )

        if lm_key not in f.keys():
            raise KeyError(
                f"Missing {lm_key}"
            )

        embed_rows = f.get_slice(
            embed_key
        )[
            MEITEI_START:MEITEI_END + 1
        ].float()

        lm_rows = f.get_slice(
            lm_key
        )[
            MEITEI_START:MEITEI_END + 1
        ].float()

    print(
        "[LEGACY] Embed rows:",
        tuple(embed_rows.shape)
    )

    print(
        "[LEGACY] LM-head rows:",
        tuple(lm_rows.shape)
    )

    if tuple(embed_rows.shape) != (
        MEITEI_COUNT,
        HIDDEN_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected embedding row shape: "
            f"{embed_rows.shape}"
        )

    if tuple(lm_rows.shape) != (
        MEITEI_COUNT,
        HIDDEN_SIZE,
    ):
        raise RuntimeError(
            f"Unexpected LM-head row shape: "
            f"{lm_rows.shape}"
        )

    difference = (
        embed_rows - lm_rows
    ).abs().max().item()

    print(
        "[LEGACY] Max embed/lm-head row difference:",
        difference
    )

    if difference > 1e-4:

        print(
            "[WARNING] Legacy tied rows are not identical."
        )

        print(
            "[WARNING] Using embedding rows as the shared "
            "Epoch-7 starting point."
        )

    return embed_rows


# ============================================================
# LOAD PROJECTOR
# ============================================================

def load_projector():

    print(
        "[PROJECTOR] Loading Epoch-6 projector"
    )

    projector = AudioProjector()

    state = torch.load(
        EPOCH6_PROJECTOR,
        map_location="cpu",
        weights_only=True,
    )

    projector.load_state_dict(
        state,
        strict=True,
    )

    projector.eval()

    for p in projector.parameters():
        p.requires_grad_(True)

    print(
        "[PROJECTOR] Parameters:",
        sum(
            p.numel()
            for p in projector.parameters()
        ),
    )

    return projector


# ============================================================
# LOAD TOKENIZER
# ============================================================

def load_tokenizer():

    print(
        "[TOKENIZER] Loading Epoch-6 tokenizer"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(EPOCH6_TOKENIZER),
        local_files_only=True,
    )

    print(
        "[TOKENIZER] Length:",
        len(tokenizer)
    )

    if len(tokenizer) != FINAL_VOCAB_SIZE:
        raise RuntimeError(
            f"Tokenizer length {len(tokenizer)} "
            f"!= {FINAL_VOCAB_SIZE}"
        )

    return tokenizer


# ============================================================
# BUILD EXACT CHATML PREFIX
# ============================================================

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
        ids.shape[1]
    )

    print(
        "[PROMPT] Exact prefix:"
    )

    print(
        repr(prompt_text)
    )

    return ids


# ============================================================
# BUILD QWEN
# ============================================================

def build_qwen(tokenizer):

    print(
        "[QWEN] Loading configuration"
    )

    config = AutoConfig.from_pretrained(
        BASE_MODEL
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
            "Expected tied Qwen word embeddings."
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
        torch_dtype=torch.bfloat16,
    )

    print(
        "[QWEN] Pre-resize input embedding:",
        tuple(
            model.get_input_embeddings()
            .weight.shape
        )
    )

    if (
        model.get_input_embeddings()
        .weight.shape[0]
        != BASE_VOCAB_SIZE
    ):
        raise RuntimeError(
            "Loaded model does not have the "
            "expected 151936-token base vocabulary."
        )

    # --------------------------------------------------------
    # Resize exactly as Epoch-6 did.
    # --------------------------------------------------------

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
        )
    )

    # --------------------------------------------------------
    # Freeze every base Qwen parameter.
    # --------------------------------------------------------

    for p in model.parameters():
        p.requires_grad_(False)

    # --------------------------------------------------------
    # Load only the 44 learned Epoch-6 Meitei rows.
    # --------------------------------------------------------

    meitei_rows = load_legacy_meitei_rows()

    # Keep trainable parameter on the same device/dtype
    # used by Qwen computation.
    embed_weight = (
        model.get_input_embeddings()
        .weight
    )

    meitei_rows = meitei_rows.to(
        device=embed_weight.device,
        dtype=embed_weight.dtype,
    )

    # --------------------------------------------------------
    # Replace embedding with frozen-base + 44-row overlay.
    # --------------------------------------------------------

    original_embedding = (
        model.get_input_embeddings()
    )

    meitei_embedding = MeiteiEmbedding(
        frozen_embedding=original_embedding,
        trainable_rows=meitei_rows,
        start_id=MEITEI_START,
    )

    model.set_input_embeddings(
        meitei_embedding
    )

    # --------------------------------------------------------
    # Replace output head.
    #
    # Use the SAME Parameter as the embedding overlay.
    # --------------------------------------------------------

    original_lm_head = model.lm_head

    frozen_output_weight = (
        original_lm_head.weight.detach()
        .clone()
        .to(
            device=embed_weight.device,
            dtype=embed_weight.dtype,
        )
    )

    shared_rows = (
        meitei_embedding.trainable_rows
    )

    model.lm_head = MeiteiLMHead(
        frozen_weight=frozen_output_weight,
        shared_meitei_rows=shared_rows,
        start_id=MEITEI_START,
    )

    # --------------------------------------------------------
    # Enable gradient checkpointing without
    # prepare_model_for_kbit_training().
    #
    # That function caused the known FP32 memory spike.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Attach fresh LoRA configuration.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load only legacy LoRA tensors.
    # Do NOT load legacy full embedding/lm_head tensors.
    # --------------------------------------------------------

    load_legacy_lora_weights(
        model
    )

    return model


# ============================================================
# LEGACY LORA WEIGHTS
# ============================================================

def load_legacy_lora_weights(model):

    files = list(
        EPOCH6_ADAPTER.glob(
            "*.safetensors"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No Epoch-6 LoRA safetensors found."
        )

    sf = files[0]

    print(
        "[LORA] Loading legacy LoRA weights:",
        sf.name
    )

    current = model.state_dict()

    loaded = 0
    missing = []

    with safe_open(
        str(sf),
        framework="pt",
        device="cpu",
    ) as f:

        for key in f.keys():

            if (
                ".lora_A." not in key
                and ".lora_B." not in key
            ):
                continue

            if key not in current:

                missing.append(key)
                continue

            tensor = f.get_tensor(
                key
            )

            target = current[key]

            if tuple(tensor.shape) != tuple(
                target.shape
            ):
                raise RuntimeError(
                    f"LoRA shape mismatch:\n"
                    f"{key}\n"
                    f"legacy={tuple(tensor.shape)}\n"
                    f"current={tuple(target.shape)}"
                )

            target.copy_(
                tensor.to(
                    device=target.device,
                    dtype=target.dtype,
                )
            )

            loaded += 1

    print(
        "[LORA] Loaded tensors:",
        loaded
    )

    if missing:

        print(
            "[LORA] Missing legacy keys:",
            len(missing)
        )

        for key in missing[:20]:
            print(
                "  ",
                key
            )

        raise RuntimeError(
            "Legacy LoRA keys did not match the "
            "new PEFT model exactly."
        )

    print(
        "[LORA] Legacy LoRA warm-start successful."
    )


# ============================================================
# TRAINABLE PARAMETER REPORT
# ============================================================

def report_trainable(model, projector):

    print("\n" + "=" * 70)
    print("TRAINABLE PARAMETER REPORT")
    print("=" * 70)

    qwen_trainable = 0
    qwen_total = 0

    for name, p in model.named_parameters():

        qwen_total += p.numel()

        if p.requires_grad:

            qwen_trainable += p.numel()

            print(
                f"TRAINABLE QWEN: "
                f"{name:90s} "
                f"{tuple(p.shape)} "
                f"{p.numel():,}"
            )

    projector_trainable = sum(
        p.numel()
        for p in projector.parameters()
        if p.requires_grad
    )

    print()
    print(
        "QWEN TOTAL PARAMS       =",
        f"{qwen_total:,}"
    )

    print(
        "QWEN TRAINABLE PARAMS   =",
        f"{qwen_trainable:,}"
    )

    print(
        "PROJECTOR TRAINABLE     =",
        f"{projector_trainable:,}"
    )

    print(
        "TOTAL TRAINABLE         =",
        f"{qwen_trainable + projector_trainable:,}"
    )

    expected_meitei = (
        MEITEI_COUNT * HIDDEN_SIZE
    )

    print()
    print(
        "EXPECTED MEITEI ROWS    =",
        f"{expected_meitei:,}"
    )

    if qwen_trainable < expected_meitei:
        raise RuntimeError(
            "Unexpectedly low Qwen trainable count."
        )


# ============================================================
# CACHE LOADER
# ============================================================

def find_cached_feature(
    dataset_name,
    record_index,
):

    # Existing V2 cache filenames are hashes, so metadata
    # must be inspected to locate dataset/index.
    #
    # To avoid expensive scanning during every training step,
    # the index is built once.

    return None


def build_cache_index():

    index = {}

    if not CACHE_DIR.exists():

        print(
            "[CACHE] Directory missing."
        )

        return index

    metadata_files = list(
        CACHE_DIR.glob("*.json")
    )

    print(
        "[CACHE] Building cache index from",
        len(metadata_files),
        "metadata files"
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

            pt_path = meta_path.with_suffix(
                ".pt"
            )

            if not pt_path.exists():
                continue

            index[
                (dataset, int(record_index))
            ] = pt_path

        except Exception:
            continue

    print(
        "[CACHE] Indexed feature files:",
        len(index)
    )

    return index


# ============================================================
# AUDIO FEATURE EXTRACTION
# ============================================================

def build_encoder():

    from transformers import (
        Wav2Vec2BertFeatureExtractor,
        Wav2Vec2BertModel,
    )

    print(
        "[ENCODER] Loading W2V-BERT 2.0"
    )

    extractor = (
        Wav2Vec2BertFeatureExtractor
        .from_pretrained(
            ENCODER_MODEL
        )
    )

    encoder = Wav2Vec2BertModel.from_pretrained(
        ENCODER_MODEL,
        torch_dtype=(
            torch.bfloat16
            if DEVICE == "cuda"
            else torch.float32
        ),
    )

    encoder.to(DEVICE)
    encoder.eval()

    for p in encoder.parameters():
        p.requires_grad_(False)

    return extractor, encoder

@torch.no_grad()
def encode_audio(
    audio_path,
    extractor,
    encoder,
):

    audio, sr = sf.read(
        str(audio_path)
    )

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != 16000:

        import torchaudio

        wav = torch.tensor(
            audio,
            dtype=torch.float32,
        )

        wav = torchaudio.functional.resample(
            wav,
            sr,
            16000,
        )

        audio = wav.numpy()

    inputs = extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
    )

    input_values = (
        inputs.input_values
        if hasattr(
            inputs,
            "input_values"
        )
        else inputs["input_values"]
    )

    input_values = input_values.to(
        DEVICE
    )

    outputs = encoder(
        input_values
    )

    hidden = outputs.last_hidden_state

    return hidden


# ============================================================
# DRY RUN
# ============================================================

def dry_run():

    print("=" * 70)
    print("EPOCH-7 RECOVERY DRY RUN")
    print("=" * 70)

    print(
        "\n[CHECK] ROOT:",
        ROOT
    )

    print(
        "[CHECK] Epoch-6:",
        EPOCH6
    )

    if not EPOCH6.exists():
        raise FileNotFoundError(
            EPOCH6
        )

    tokenizer = load_tokenizer()

    projector = load_projector()

    prefix_ids = build_prompt_prefix(
        tokenizer
    )

    model = build_qwen(
        tokenizer
    )

    report_trainable(
        model,
        projector
    )

    print("\n[DRY RUN] Model construction successful.")

    print(
        "[DRY RUN] Prefix shape:",
        tuple(prefix_ids.shape)
    )

    # --------------------------------------------------------
    # Verify shared parameter
    # --------------------------------------------------------

    base_model = model.get_base_model()

    embedding_module = (
        base_model.get_input_embeddings()
    )

    lm_head = base_model.lm_head

    shared_parameter = (
        embedding_module.trainable_rows
    )

    if lm_head.meitei_rows is not shared_parameter:

        raise RuntimeError(
            "Embedding and LM-head are NOT sharing "
            "the same Meitei Parameter."
        )

    print(
        "[DRY RUN] Shared Meitei parameter: PASS"
    )

    print(
        "[DRY RUN] Shape:",
        tuple(shared_parameter.shape)
    )

    # --------------------------------------------------------
    # Verify vocabulary dimensions
    # --------------------------------------------------------

    if model.config.vocab_size != FINAL_VOCAB_SIZE:

        raise RuntimeError(
            f"Model config vocab size "
            f"{model.config.vocab_size} "
            f"!= {FINAL_VOCAB_SIZE}"
        )

    print(
        "[DRY RUN] Vocabulary size: PASS"
    )

    # --------------------------------------------------------
    # Verify no legacy full embedding parameter is trainable.
    # --------------------------------------------------------

    bad = []

    for name, p in model.named_parameters():

        if (
            (
                "embed_tokens" in name
                or "lm_head" in name
            )
            and p.requires_grad
            and "trainable_rows" not in name
            and "meitei_rows" not in name
        ):
            bad.append(name)

    if bad:

        print(
            "[DRY RUN] BAD TRAINABLE EMBEDDING PARAMETERS:"
        )

        for name in bad:
            print(
                "   ",
                name
            )

        raise RuntimeError(
            "Full embedding/lm_head parameter "
            "is unexpectedly trainable."
        )

    print(
        "[DRY RUN] Full-vocabulary embedding freeze: PASS"
    )

    # --------------------------------------------------------
    # Cache report
    # --------------------------------------------------------

    cache_index = build_cache_index()

    print(
        "[DRY RUN] Cache entries available:",
        len(cache_index)
    )

    # --------------------------------------------------------
    # GPU memory report
    # --------------------------------------------------------

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / 1024**3
        )

        reserved = (
            torch.cuda.memory_reserved()
            / 1024**3
        )

        print(
            f"[GPU] Allocated: {allocated:.2f} GB"
        )

        print(
            f"[GPU] Reserved : {reserved:.2f} GB"
        )

    print("\n" + "=" * 70)
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


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate recovery model only.",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Training mode. Not enabled yet.",
    )

    args = parser.parse_args()

    if args.train:

        raise RuntimeError(
            "Full training is intentionally disabled in "
            "this first recovery build. Run --dry-run first."
        )

    if not args.dry_run:

        print(
            "Specify --dry-run."
        )

        print(
            "Example:"
        )

        print(
            "python .\\ASR\\train_epoch7_recovery.py --dry-run"
        )

        return

    dry_run()


if __name__ == "__main__":
    main()
