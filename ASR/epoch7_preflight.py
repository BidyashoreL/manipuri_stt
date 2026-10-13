from pathlib import Path
import json

ROOT = Path(r"E:\manipuri_stt1\manipuri_stt4\manipuri_stt-20260816T162024Z-1-001\manipuri_stt")

CKPT = ROOT / "ASR" / "checkpoints" / "epoch_6_broken_objective"
TOK = CKPT / "tokenizer"
ADAPTER = CKPT / "qwen_lora"
PROJECTOR = CKPT / "audio_projector.pt"

print("=" * 70)
print("EPOCH-7 RECOVERY — READ-ONLY COMPATIBILITY PREFLIGHT")
print("=" * 70)

errors = []

# ============================================================
# 1. QWEN CONFIG
# ============================================================
print("\n[1] QWEN BASE CONFIG")

try:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

    print("CONFIG_VOCAB_SIZE      =", cfg.vocab_size)
    print("HIDDEN_SIZE            =", cfg.hidden_size)
    print("TIED_WORD_EMBEDDINGS   =", cfg.tie_word_embeddings)

    if cfg.vocab_size != 151936:
        errors.append(f"Unexpected Qwen vocab size: {cfg.vocab_size}")

    if cfg.hidden_size != 2048:
        errors.append(f"Unexpected hidden size: {cfg.hidden_size}")

    if cfg.tie_word_embeddings is not True:
        errors.append("Qwen is not reporting tied word embeddings")

except Exception as e:
    errors.append(f"Qwen config check failed: {repr(e)}")


# ============================================================
# 2. TOKENIZER
# ============================================================
print("\n[2] EPOCH-6 TOKENIZER")

tokenizer_json = TOK / "tokenizer.json"

if not tokenizer_json.exists():
    errors.append(f"Missing tokenizer: {tokenizer_json}")
else:
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))

    vocab = data["model"]["vocab"]
    added = data.get("added_tokens", [])

    all_tokens = {}

    for token, idx in vocab.items():
        all_tokens[token] = idx

    for item in added:
        all_tokens[item["content"]] = item["id"]

    tokenizer_length = max(all_tokens.values()) + 1

    print("MODEL_VOCAB_SIZE       =", len(vocab))
    print("ADDED_TOKEN_COUNT      =", len(added))
    print("TOTAL_TOKENIZER_LENGTH =", tokenizer_length)

    meitei = [
        (token, idx)
        for token, idx in all_tokens.items()
        if 151665 <= idx <= 151708
    ]

    meitei.sort(key=lambda x: x[1])

    print("MEITEI_TOKEN_COUNT     =", len(meitei))

    print("\nMEITEI TOKEN RANGE:")
    for token, idx in meitei:
        print(f"  {idx}: {token}")

    if tokenizer_length != 151709:
        errors.append(
            f"Epoch-6 tokenizer length is {tokenizer_length}, expected 151709"
        )

    if len(meitei) != 44:
        errors.append(
            f"Expected 44 Meitei tokens, found {len(meitei)}"
        )


# ============================================================
# 3. LORA CONFIG
# ============================================================
print("\n[3] EPOCH-6 LORA CONFIG")

adapter_config = ADAPTER / "adapter_config.json"

if not adapter_config.exists():
    errors.append(f"Missing adapter config: {adapter_config}")
else:
    acfg = json.loads(adapter_config.read_text(encoding="utf-8"))

    print("BASE_MODEL            =", acfg.get("base_model_name_or_path"))
    print("R                     =", acfg.get("r"))
    print("LORA_ALPHA            =", acfg.get("lora_alpha"))
    print("LORA_DROPOUT          =", acfg.get("lora_dropout"))
    print("TARGET_MODULES        =", acfg.get("target_modules"))
    print("MODULES_TO_SAVE       =", acfg.get("modules_to_save"))
    print("PEFT_TYPE             =", acfg.get("peft_type"))

    expected_targets = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    }

    actual_targets = set(acfg.get("target_modules") or [])

    if actual_targets != expected_targets:
        errors.append(
            f"Unexpected LoRA targets: {sorted(actual_targets)}"
        )

    if acfg.get("r") != 16:
        errors.append("Expected LoRA rank 16")

    if acfg.get("lora_alpha") != 32:
        errors.append("Expected LoRA alpha 32")


# ============================================================
# 4. EPOCH-6 SAFETENSORS
# ============================================================
print("\n[4] EPOCH-6 SAFETENSORS")

try:
    from safetensors import safe_open

    safes = list(ADAPTER.glob("*.safetensors"))

    if not safes:
        errors.append("No Epoch-6 safetensors file found")
    else:
        for sf in safes:

            print("\nFILE:", sf.name)

            with safe_open(
                str(sf),
                framework="pt",
                device="cpu"
            ) as f:

                keys = list(f.keys())

                print("TENSOR_COUNT =", len(keys))

                embed_keys = [
                    k for k in keys
                    if "embed_tokens" in k
                ]

                lm_keys = [
                    k for k in keys
                    if "lm_head" in k
                ]

                lora_a = [
                    k for k in keys
                    if ".lora_A." in k
                ]

                lora_b = [
                    k for k in keys
                    if ".lora_B." in k
                ]

                print("LORA_A_COUNT =", len(lora_a))
                print("LORA_B_COUNT =", len(lora_b))
                print("EMBED_KEYS   =", embed_keys)
                print("LM_HEAD_KEYS =", lm_keys)

                for key in embed_keys + lm_keys:

                    shape = f.get_slice(key).get_shape()

                    print(
                        f"{key} -> {shape}"
                    )

                    if tuple(shape) != (151709, 2048):
                        errors.append(
                            f"{key} has unexpected shape {shape}"
                        )

                if len(lora_a) != 252:
                    errors.append(
                        f"Expected 252 LoRA-A tensors, found {len(lora_a)}"
                    )

                if len(lora_b) != 252:
                    errors.append(
                        f"Expected 252 LoRA-B tensors, found {len(lora_b)}"
                    )

except Exception as e:
    errors.append(
        f"Safetensors inspection failed: {repr(e)}"
    )


# ============================================================
# 5. AUDIO PROJECTOR
# ============================================================
print("\n[5] EPOCH-6 AUDIO PROJECTOR")

if not PROJECTOR.exists():

    errors.append(
        f"Missing projector: {PROJECTOR}"
    )

else:

    import torch

    state = torch.load(
        PROJECTOR,
        map_location="cpu",
        weights_only=True
    )

    for key, value in state.items():
        print(
            f"{key:20s} {tuple(value.shape)} {value.dtype}"
        )

    expected = {
        "downsample.weight": (1024, 1024, 2),
        "downsample.bias": (1024,),
        "proj.weight": (2048, 1024),
        "proj.bias": (2048,),
        "norm.weight": (2048,),
        "norm.bias": (2048,),
    }

    for key, shape in expected.items():

        if key not in state:

            errors.append(
                f"Projector missing {key}"
            )

        elif tuple(state[key].shape) != shape:

            errors.append(
                f"Projector {key}: expected {shape}, "
                f"found {tuple(state[key].shape)}"
            )


# ============================================================
# 6. FEATURE CACHE
# ============================================================
print("\n[6] FEATURE CACHE")

CACHE = ROOT / "ASR" / "v2" / "cache" / "features"

if not CACHE.exists():

    errors.append(
        f"Cache directory missing: {CACHE}"
    )

else:

    pt_files = list(CACHE.glob("*.pt"))
    json_files = list(CACHE.glob("*.json"))

    print("CACHE_DIRECTORY =", CACHE)
    print("PT_FILES        =", len(pt_files))
    print("JSON_FILES      =", len(json_files))

    if pt_files:

        sample = pt_files[0]

        print("SAMPLE_PT       =", sample.name)

        meta = sample.with_suffix(".json")

        if meta.exists():

            try:

                md = json.loads(
                    meta.read_text(
                        encoding="utf-8"
                    )
                )

                print(
                    "SAMPLE_DATASET  =",
                    md.get("dataset")
                )

                print(
                    "SAMPLE_INDEX    =",
                    md.get("record_index")
                )

                print(
                    "SAMPLE_SHAPE    =",
                    md.get("shape")
                )

                print(
                    "SAMPLE_DTYPE    =",
                    md.get("dtype")
                )

            except Exception as e:

                errors.append(
                    f"Could not read cache metadata "
                    f"{meta.name}: {repr(e)}"
                )


# ============================================================
# FINAL
# ============================================================
print("\n" + "=" * 70)

if errors:

    print("PREFLIGHT RESULT: FAIL")
    print("=" * 70)

    for i, error in enumerate(errors, 1):
        print(f"{i}. {error}")

    print("\nNO TRAINING SHOULD BE STARTED.")

    raise SystemExit(1)

else:

    print("PREFLIGHT RESULT: PASS")
    print("=" * 70)
    print(
        "Epoch-6 artifacts are structurally compatible "
        "with the planned Epoch-7 recovery architecture."
    )
    print("\nNO FILES WERE MODIFIED.")
