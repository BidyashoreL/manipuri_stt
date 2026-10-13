import sys
import subprocess
import torch
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

CHECKPOINT_DIR = Path(
    "../checkpoints/v2_ctc/b2_partial_encoder/regularized"
)

VALIDATOR = Path("validate_b2_regularized.py")


# ============================================================
# CANDIDATES
# ============================================================

CANDIDATES = {
    "E6_E7": [
        "model_E6_best.pt",
        "model_E7_best.pt",
    ],

    "E5_E6": [
        "model_E5_best.pt",
        "model_E6_best.pt",
    ],
}


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_state_dict(path):

    print(f"[LOAD] {path.name}")

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(state, dict):
        raise TypeError(
            f"Unexpected checkpoint format: {type(state)}"
        )

    print(f"[LOAD] Parameters: {len(state)}")

    return state


# ============================================================
# AVERAGE STATE DICTS
# ============================================================

def average_state_dicts(state_dicts):

    if not state_dicts:
        raise ValueError("No state dictionaries supplied.")

    keys = list(state_dicts[0].keys())

    # Make sure all checkpoints have identical structure
    for i, state_dict in enumerate(state_dicts[1:], start=1):

        if list(state_dict.keys()) != keys:

            raise RuntimeError(
                f"Checkpoint {i} has a different "
                f"state_dict structure."
            )

    averaged = {}

    for key in keys:

        tensors = [
            state_dict[key]
            for state_dict in state_dicts
        ]

        # Average floating-point parameters
        if torch.is_floating_point(tensors[0]):

            averaged[key] = (
                torch.stack(
                    [
                        tensor.float()
                        for tensor in tensors
                    ],
                    dim=0,
                )
                .mean(dim=0)
                .to(tensors[0].dtype)
            )

        # Keep integer/buffer tensors from first checkpoint
        else:

            averaged[key] = tensors[0].clone()

    return averaged


# ============================================================
# CREATE SOUP
# ============================================================

def create_soup(name, checkpoint_names):

    print()
    print("=" * 72)
    print(f"CREATING SOUP: {name}")
    print("=" * 72)

    paths = [
        CHECKPOINT_DIR / filename
        for filename in checkpoint_names
    ]

    # Check all files exist
    for path in paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Checkpoint not found:\n{path}"
            )

        print(f"  ✓ {path.name}")

    # Load checkpoints
    state_dicts = [
        load_state_dict(path)
        for path in paths
    ]

    # Average parameters
    print()
    print("[AVERAGING]")

    if len(paths) == 2:

        print(
            f"  {paths[0].name} + {paths[1].name}"
        )

        print(
            "  Uniform weights: 1/2 + 1/2"
        )

    else:

        weight = 1.0 / len(paths)

        print(
            "  + ".join(path.name for path in paths)
        )

        print(
            "  Uniform weights: "
            + " + ".join(
                [f"{weight:.4f}"] * len(paths)
            )
        )

    averaged = average_state_dicts(state_dicts)

    # Output path
    output = (
        CHECKPOINT_DIR
        / f"model_soup_{name}.pt"
    )

    # Save
    torch.save(
        averaged,
        output,
    )

    print()
    print("[SAVED]")
    print(f"  {output}")
    print(f"  Parameters: {len(averaged)}")

    return output


# ============================================================
# VALIDATE SOUP
# ============================================================

def validate(checkpoint):

    print()
    print("=" * 72)
    print(f"VALIDATING: {checkpoint.name}")
    print("=" * 72)

    if not VALIDATOR.exists():

        raise FileNotFoundError(
            f"Validator not found:\n{VALIDATOR}"
        )

    # IMPORTANT:
    # Use the exact Python interpreter running this script.
    #
    # This prevents Windows from accidentally launching
    # another Python installation without pandas/torch/etc.
    command = [
        sys.executable,
        str(VALIDATOR),
        "--checkpoint",
        checkpoint.name,
    ]

    print("[PYTHON]")
    print(f"  {sys.executable}")

    print("[COMMAND]")
    print(
        "  "
        + " ".join(
            f'"{x}"' if " " in x else x
            for x in command
        )
    )

    print()

    result = subprocess.run(
        command,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"\nValidation failed for:\n"
            f"{checkpoint.name}\n"
            f"Exit code: {result.returncode}"
        )

    print()
    print(
        f"[VALIDATION COMPLETE] "
        f"{checkpoint.name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("MANIPURI ASR V2 — GREEDY SOUP VALIDATION")
    print("=" * 72)

    print()
    print("[ENVIRONMENT]")
    print(f"Python: {sys.executable}")

    print()
    print("[CHECKPOINT DIRECTORY]")
    print(f"{CHECKPOINT_DIR.resolve()}")

    print()
    print("[EXPERIMENTS]")
    print("  1. E6 + E7")
    print("  2. E5 + E6")
    print()
    print("Existing E5 + E6 + E7 validation will NOT be rerun.")
    print("Official 410-sample test set will NOT be touched.")

    # --------------------------------------------------------
    # Run candidates
    # --------------------------------------------------------

    for name, checkpoints in CANDIDATES.items():

        soup_path = create_soup(
            name,
            checkpoints,
        )

        validate(soup_path)

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("GREEDY SOUP VALIDATION COMPLETE")
    print("=" * 72)

    print()
    print("Candidates evaluated:")

    print(
        "  ✓ E6 + E7"
    )

    print(
        "  ✓ E5 + E6"
    )

    print()
    print("Previously validated:")

    print(
        "  ✓ E5 + E6 + E7"
    )

    print()
    print("Next step:")
    print(
        "Compare validation CERs and determine "
        "the final checkpoint."
    )

    print()
    print("DO NOT evaluate the 410-sample test set yet.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()