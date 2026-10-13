
"""
B3-NARROW CTC Head Bias Forensics

Purpose:
    Inspect the trained B3 CTC head's weights and biases.
    This is a read-only diagnostic: it does not modify the checkpoint.

Run from:
    ASR/manipuri_asr_v2/

Output:
    diagnostics/ctc_head_forensics/ctc_head_bias_summary.json
"""

import json
from pathlib import Path

import torch


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()
ASR = ROOT.parent

CHECKPOINT = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "best.pt"
)

VOCAB_PATH = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

OUT_DIR = ROOT / "diagnostics" / "ctc_head_forensics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stop(message):
    raise RuntimeError(message)


def main():
    print("=" * 72)
    print("B3-NARROW CTC HEAD BIAS FORENSICS")
    print("=" * 72)

    print(f"[CHECKPOINT] {CHECKPOINT}")
    print(f"[VOCAB]      {VOCAB_PATH}")

    if not CHECKPOINT.is_file():
        stop(
            "B3 checkpoint not found. Expected:\n"
            f"{CHECKPOINT}"
        )

    if not VOCAB_PATH.is_file():
        stop(
            "Vocabulary file not found. Expected:\n"
            f"{VOCAB_PATH}"
        )

    # --------------------------------------------------------
    # Load vocabulary and identify the blank token.
    # --------------------------------------------------------

    with open(VOCAB_PATH, "r", encoding="utf-8-sig") as f:
        vocab = json.load(f)

    if not isinstance(vocab, dict):
        stop("Expected vocab.json to contain a token-to-ID dictionary.")

    print("\n[VOCABULARY]")
    print(f"Vocabulary entries: {len(vocab)}")

    blank_matches = []

    if "<blank>" in vocab:
        blank_matches = [("<blank>", int(vocab["<blank>"]))]
    else:
        stop(
            "The expected <blank> token was not found in vocab.json. "
            "Check the vocabulary before continuing."
        )

    for token, token_id in blank_matches:
        print(f"Blank-like token: {token!r} -> {token_id}")

    # --------------------------------------------------------
    # Load checkpoint on CPU. No model weights are changed.
    # --------------------------------------------------------

    print("\n[LOAD] Reading checkpoint on CPU...")

    try:
        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        stop(
            "Could not safely load checkpoint with weights_only=True.\n"
            "Do not change checkpoint-loading settings blindly.\n"
            f"Error: {exc}"
        )

    if not isinstance(checkpoint, dict):
        stop("Unexpected checkpoint format: expected a dictionary.")

    print("[CHECKPOINT KEYS]")
    for key in checkpoint:
        print(f"  {key}")

    # --------------------------------------------------------
    # Extract the model state dictionary.
    # --------------------------------------------------------

    state_dict = None
    state_source = None

    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)

        if isinstance(value, dict) and any(
            isinstance(item, torch.Tensor) for item in value.values()
        ):
            state_dict = value
            state_source = key
            break

    if state_dict is None:
        if any(
            isinstance(value, torch.Tensor)
            for value in checkpoint.values()
        ):
            state_dict = checkpoint
            state_source = "checkpoint root"

    if state_dict is None:
        stop(
            "Could not identify a tensor state dictionary in the "
            "checkpoint. Inspect the checkpoint keys before proceeding."
        )

    print(f"\n[STATE DICT] Source: {state_source}")

    # --------------------------------------------------------
    # Find CTC head parameters.
    # --------------------------------------------------------

    head_parameters = []

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        if "ctc_head" in key.lower():
            head_parameters.append((key, value.detach().float()))

    print("\n[CTC HEAD CANDIDATES]")

    for key, tensor in head_parameters:
        print(f"{key:65s} shape={tuple(tensor.shape)}")

    if not head_parameters:
        print("\n[DIAGNOSTIC] No parameter key contains 'ctc_head'.")
        print("Possible linear-layer parameters for inspection:")

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.ndim in (1, 2):
                print(f"{key:65s} shape={tuple(value.shape)}")

        stop(
            "CTC head parameters were not identified. "
            "Do not guess the blank index or layer name."
        )

    weight_candidates = [
        (key, tensor)
        for key, tensor in head_parameters
        if key.lower().endswith(".weight") and tensor.ndim == 2
    ]

    bias_candidates = [
        (key, tensor)
        for key, tensor in head_parameters
        if key.lower().endswith(".bias") and tensor.ndim == 1
    ]

    if len(weight_candidates) != 1:
        stop(
            "Expected exactly one 2D CTC head weight, but found "
            f"{len(weight_candidates)}. Review the candidates above."
        )

    weight_key, weight = weight_candidates[0]

    bias_key = None
    bias = None

    if len(bias_candidates) == 1:
        bias_key, bias = bias_candidates[0]
    elif len(bias_candidates) > 1:
        stop(
            "Found multiple possible CTC head biases. "
            "Inspect the printed keys before continuing."
        )

    # PyTorch linear layers normally use [output_classes, input_features].
    num_classes, feature_dim = weight.shape

    print("\n" + "=" * 72)
    print("CTC HEAD DIMENSIONS")
    print("=" * 72)
    print(f"Weight key:   {weight_key}")
    print(f"Weight shape: {tuple(weight.shape)}")
    print(f"Number of classes: {num_classes}")
    print(f"Input feature dimension: {feature_dim}")

    if bias is not None:
        print(f"Bias key: {bias_key}")
        print(f"Bias shape: {tuple(bias.shape)}")
    else:
        print("Explicit bias parameter: absent")

    # --------------------------------------------------------
    # Identify blank ID only when the vocabulary convention
    # can be verified. Do not assume class 0 is blank.
    # --------------------------------------------------------

    blank_id = None
    blank_token = None

    if len(blank_matches) == 1:
        blank_token, blank_id = blank_matches[0]

    # If the trainer uses a special blank convention not
    # represented by a token containing "blank", stop safely.
    if blank_id is None:
        stop(
            "Could not determine the CTC blank ID from vocab.json. "
            "Check SPECIAL_BLANK and the blank-ID convention in "
            "train_b3_narrow_E11_E12.py before running further analysis."
        )

    if not 0 <= blank_id < num_classes:
        stop(
            f"Blank ID {blank_id} is outside the head's "
            f"{num_classes} output classes."
        )

    if bias is not None and bias.numel() != num_classes:
        stop("Bias dimension does not match the number of output classes.")

    print(f"Verified blank token: {blank_token!r}")
    print(f"Verified blank ID: {blank_id}")

    # --------------------------------------------------------
    # Weight analysis.
    # --------------------------------------------------------

    weight_norms = torch.linalg.vector_norm(weight, dim=1)

    nonblank_mask = torch.ones(num_classes, dtype=torch.bool)
    nonblank_mask[blank_id] = False

    blank_weight_norm = float(weight_norms[blank_id].item())
    nonblank_norms = weight_norms[nonblank_mask]

    mean_nonblank_norm = float(nonblank_norms.mean().item())
    median_nonblank_norm = float(nonblank_norms.median().item())
    std_nonblank_norm = float(
        nonblank_norms.std(unbiased=False).item()
    )

    mean_nonblank_weight = weight[nonblank_mask].mean(dim=0)
    blank_weight = weight[blank_id]

    blank_mean_distance = float(
        torch.linalg.vector_norm(
            blank_weight - mean_nonblank_weight
        ).item()
    )

    cosine_similarity = float(
        torch.nn.functional.cosine_similarity(
            blank_weight.unsqueeze(0),
            mean_nonblank_weight.unsqueeze(0),
            dim=1,
        )[0].item()
    )

    weight_norm_ratio = (
        blank_weight_norm / max(mean_nonblank_norm, 1e-12)
    )

    print("\n" + "=" * 72)
    print("WEIGHT ANALYSIS")
    print("=" * 72)
    print(f"Blank weight norm:                {blank_weight_norm:.8f}")
    print(f"Mean nonblank weight norm:        {mean_nonblank_norm:.8f}")
    print(f"Median nonblank weight norm:      {median_nonblank_norm:.8f}")
    print(f"Std nonblank weight norm:         {std_nonblank_norm:.8f}")
    print(f"Blank/mean-nonblank norm ratio:   {weight_norm_ratio:.8f}")
    print(f"Blank-to-mean weight distance:    {blank_mean_distance:.8f}")
    print(f"Cosine similarity to mean:        {cosine_similarity:.8f}")

    # --------------------------------------------------------
    # Bias analysis.
    # --------------------------------------------------------

    if bias is not None:
        blank_bias = float(bias[blank_id].item())
        nonblank_bias = bias[nonblank_mask]

        mean_nonblank_bias = float(nonblank_bias.mean().item())
        median_nonblank_bias = float(nonblank_bias.median().item())
        std_nonblank_bias = float(
            nonblank_bias.std(unbiased=False).item()
        )

        bias_gap = blank_bias - mean_nonblank_bias

        print("\n" + "=" * 72)
        print("BIAS ANALYSIS")
        print("=" * 72)
        print(f"Blank bias:                       {blank_bias:.8f}")
        print(f"Mean nonblank bias:               {mean_nonblank_bias:.8f}")
        print(f"Median nonblank bias:             {median_nonblank_bias:.8f}")
        print(f"Std nonblank bias:                {std_nonblank_bias:.8f}")
        print(f"Blank minus mean nonblank bias:   {bias_gap:.8f}")

        sorted_bias = torch.argsort(bias, descending=True)

        print("\nTOP 10 CLASSES BY BIAS")
        for rank, index in enumerate(sorted_bias[:10], start=1):
            index = int(index.item())
            marker = "  <-- BLANK" if index == blank_id else ""
            print(
                f"{rank:2d}. class={index:3d} "
                f"bias={bias[index].item(): .8f}{marker}"
            )

        print("\nBOTTOM 10 CLASSES BY BIAS")
        for rank, index in enumerate(sorted_bias[-10:], start=1):
            index = int(index.item())
            marker = "  <-- BLANK" if index == blank_id else ""
            print(
                f"{rank:2d}. class={index:3d} "
                f"bias={bias[index].item(): .8f}{marker}"
            )

    else:
        blank_bias = None
        mean_nonblank_bias = None
        median_nonblank_bias = None
        std_nonblank_bias = None
        bias_gap = None

        print("\nNo explicit bias parameter found in the CTC head.")

    # --------------------------------------------------------
    # Weight norm ranking.
    # --------------------------------------------------------

    sorted_norms = torch.argsort(weight_norms, descending=True)

    print("\nTOP 10 CLASSES BY WEIGHT NORM")

    for rank, index in enumerate(sorted_norms[:10], start=1):
        index = int(index.item())
        marker = "  <-- BLANK" if index == blank_id else ""
        print(
            f"{rank:2d}. class={index:3d} "
            f"norm={weight_norms[index].item(): .8f}{marker}"
        )

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    summary = {
        "checkpoint": str(CHECKPOINT),
        "weight_key": weight_key,
        "bias_key": bias_key,
        "num_classes": int(num_classes),
        "feature_dim": int(feature_dim),
        "blank_token": blank_token,
        "blank_id": int(blank_id),
        "blank_weight_norm": blank_weight_norm,
        "mean_nonblank_weight_norm": mean_nonblank_norm,
        "median_nonblank_weight_norm": median_nonblank_norm,
        "std_nonblank_weight_norm": std_nonblank_norm,
        "blank_weight_norm_ratio": weight_norm_ratio,
        "blank_to_mean_nonblank_weight_distance": blank_mean_distance,
        "cosine_similarity_blank_to_mean_nonblank": cosine_similarity,
        "blank_bias": blank_bias,
        "mean_nonblank_bias": mean_nonblank_bias,
        "median_nonblank_bias": median_nonblank_bias,
        "std_nonblank_bias": std_nonblank_bias,
        "blank_bias_gap": bias_gap,
    }

    output_path = OUT_DIR / "ctc_head_bias_summary.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print("CTC HEAD BIAS FORENSICS COMPLETE")
    print("=" * 72)
    print(f"[OUTPUT] {output_path}")
    print("\nThis test inspects parameters only.")
    print("It does not prove that the head causes blank collapse.")


if __name__ == "__main__":
    main()
