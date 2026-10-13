from pathlib import Path
import json
import re
import unicodedata
import warnings

import numpy as np
from pyctcdecode import build_ctcdecoder
from jiwer import cer, wer

# ============================================================
# FRESH 4-GRAM VERIFICATION — B3-NARROW EPOCH 11
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

CACHE_DIR = (
    ASR / "checkpoints" / "v3_ctc" / "b3_narrow" / "val_logit_cache"
)

VOCAB_PATH = (
    ASR / "checkpoints" / "v2_ctc" / "b2_partial_encoder" / "vocab.json"
)

LM_DIR = ASR / "manipuri_asr_v2" / "kenlm_experiment"
LM_PATH = LM_DIR / "manipuri_4gram.bin"
UNIGRAM_PATH = LM_DIR / "manipuri_unigrams.txt"

OUTPUT_PATH = (
    ASR / "checkpoints" / "v3_ctc" / "b3_narrow"
    / "kenlm_4gram_sweep" / "verification_alpha05_beta20.json"
)

ALPHA = 0.5
BETA = 2.0
BEAM_WIDTH = 500


def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    return re.sub(r"\s+", " ", text).strip()


def require_file(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def main():
    print("=" * 65)
    print("B3-NARROW 4-GRAM INDEPENDENT VERIFICATION")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. Verify input files
    # --------------------------------------------------------
    required = [
        CACHE_DIR / "logits.npy",
        CACHE_DIR / "offsets.npy",
        CACHE_DIR / "texts.json",
        CACHE_DIR / "metadata.json",
        VOCAB_PATH,
        LM_PATH,
        UNIGRAM_PATH,
    ]

    for path in required:
        require_file(path)
        print(f"[OK] {path.name}")

    # --------------------------------------------------------
    # 2. Load and verify cached logits
    # --------------------------------------------------------
    print("\n[LOAD] Reading cached validation logits...")

    logits = np.load(
        CACHE_DIR / "logits.npy",
        mmap_mode="r",
    )
    offsets = np.load(CACHE_DIR / "offsets.npy")

    references = json.loads(
        (CACHE_DIR / "texts.json").read_text(encoding="utf-8")
    )

    metadata = json.loads(
        (CACHE_DIR / "metadata.json").read_text(encoding="utf-8")
    )

    if isinstance(references, dict):
        raise ValueError(
            "texts.json contains a dictionary, not a list. "
            "Inspect its structure before continuing."
        )

    if len(references) != 864:
        raise ValueError(
            f"Expected 864 validation references; found {len(references)}."
        )

    if offsets.ndim != 1 or len(offsets) != len(references) + 1:
        raise ValueError("Offsets do not match the number of references.")

    if int(offsets[0]) != 0 or int(offsets[-1]) != logits.shape[0]:
        raise ValueError("Offsets do not cover the cached logits correctly.")

    if np.any(np.diff(offsets) <= 0):
        raise ValueError("One or more utterances have invalid frame offsets.")

    if logits.ndim != 2 or logits.shape[1] != 60:
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    print(f"[OK] Logits shape: {logits.shape}")
    print(f"[OK] Validation samples: {len(references)}")
    print(f"[OK] Cache model: {metadata.get('model')}")
    print(f"[OK] Experiment: {metadata.get('experiment')}")
    print(f"[OK] Epoch: {metadata.get('epoch')}")

    # --------------------------------------------------------
    # 3. Reconstruct vocabulary in numeric ID order
    # --------------------------------------------------------
    print("\n[VOCAB] Loading token-to-ID vocabulary...")

    vocab_data = json.loads(
        VOCAB_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(vocab_data, dict):
        raise ValueError("Expected vocab.json to be a token-to-ID dictionary.")

    if not all(
        isinstance(token_id, int)
        for token_id in vocab_data.values()
    ):
        raise ValueError("Vocabulary IDs must all be integers.")

    vocab_size = len(vocab_data)
    ids = list(vocab_data.values())

    if sorted(ids) != list(range(vocab_size)):
        raise ValueError(
            "Vocabulary IDs are not contiguous from 0. "
            "Do not guess or remove tokens."
        )

    labels = [
        token
        for token, token_id in sorted(
            vocab_data.items(),
            key=lambda item: item[1],
        )
    ]

    # IMPORTANT:
    # pyctcdecode represents the CTC blank as an empty string.
    # Passing the literal "<blank>" makes it treat that as an
    # ordinary label and can cause it to add another blank.
    blank_id = vocab_data.get("<blank>")

    if blank_id is None:
        raise ValueError("The vocabulary has no '<blank>' token.")

    if blank_id != 0:
        raise ValueError(
            f"Blank ID is {blank_id}, not 0. Verify the training convention."
        )

    labels[blank_id] = ""

    if len(labels) != logits.shape[1]:
        raise ValueError(
            f"Vocabulary/logits mismatch: {len(labels)} labels "
            f"versus {logits.shape[1]} logit classes."
        )

    if labels[blank_id] != "":
        raise ValueError("Failed to convert the blank label correctly.")

    print(f"[OK] Vocabulary entries: {len(labels)}")
    print(f"[OK] Blank ID: {blank_id}, represented as an empty string")
    print(f"[OK] Logit classes: {logits.shape[1]}")

    # --------------------------------------------------------
    # 4. Load 4-gram model and unigram list
    # --------------------------------------------------------
    print("\n[LM] Loading 4-gram language model...")

    unigrams = [
        line.strip()
        for line in UNIGRAM_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    print(f"[OK] Unigrams loaded: {len(unigrams)}")
    print(f"[OK] Language model: {LM_PATH.name}")
    print(f"[CONFIG] alpha={ALPHA}, beta={BETA}")
    print(f"[CONFIG] beam width={BEAM_WIDTH}")

    # --------------------------------------------------------
    # 5. Build decoder
    # --------------------------------------------------------
    print("\n[DECODER] Building pyctcdecode decoder...")

    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(LM_PATH),
        unigrams=unigrams,
        alpha=ALPHA,
        beta=BETA,
    )

    print("[OK] Decoder constructed.")

    # --------------------------------------------------------
    # 6. Decode every validation utterance
    # --------------------------------------------------------
    print("\n[DECODE] Processing 864 validation utterances...")

    predictions = []

    for i in range(len(references)):
        start = int(offsets[i])
        end = int(offsets[i + 1])

        # Cached arrays are log_softmax values, as recorded
        # in metadata.json.
        lp = np.asarray(logits[start:end])

        if lp.shape[1] != len(labels):
            raise ValueError(
                f"Sample {i}: {lp.shape[1]} classes, "
                f"but decoder vocabulary has {len(labels)} labels."
            )

        prediction = decoder.decode(
            lp,
            beam_width=BEAM_WIDTH,
        )

        predictions.append(normalize_text(prediction))

        if (i + 1) % 100 == 0 or i + 1 == len(references):
            print(f"[PROGRESS] {i + 1}/{len(references)}")

    # --------------------------------------------------------
    # 7. Calculate metrics
    # --------------------------------------------------------
    refs = [normalize_text(x) for x in references]

    score_cer = float(cer(refs, predictions))
    score_wer = float(wer(refs, predictions))

    # --------------------------------------------------------
    # 8. Save to a NEW file; preserve historical results
    # --------------------------------------------------------
    result = {
        "experiment": "fresh_4gram_verification",
        "model": metadata.get("model"),
        "checkpoint": metadata.get("checkpoint"),
        "epoch": metadata.get("epoch"),
        "validation_samples": len(refs),
        "official_test_accessed": False,
        "normalization": "NFC + whitespace collapse",
        "logits_type": metadata.get("logits_type"),
        "decoder_method": "decoder.decode",
        "beam_width": BEAM_WIDTH,
        "alpha": ALPHA,
        "beta": BETA,
        "cer": score_cer,
        "wer": score_wer,
        "cer_percent": score_cer * 100,
        "wer_percent": score_wer * 100,
        "logits_shape": list(logits.shape),
        "vocab_size": len(labels),
        "blank_id": blank_id,
        "language_model": str(LM_PATH),
        "unigram_file": str(UNIGRAM_PATH),
        "predictions": predictions,
        "references": refs,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("VERIFICATION RESULTS")
    print("=" * 65)
    print(f"Samples: {len(refs)}")
    print(f"CER: {score_cer:.6%}")
    print(f"WER: {score_wer:.6%}")
    print(f"CER: {score_cer * 100:.4f}%")
    print(f"WER: {score_wer * 100:.4f}%")
    print(f"\nSaved results to:\n{OUTPUT_PATH}")
    print("\nExisting sweep files were not overwritten.")


if __name__ == "__main__":
    main()
