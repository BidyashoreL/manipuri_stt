import torch
from pathlib import Path

CHECKPOINT_DIR = Path(
    "../checkpoints/v2_ctc/b2_partial_encoder/regularized"
)

CHECKPOINTS = [
    CHECKPOINT_DIR / "model_E5_best.pt",
    CHECKPOINT_DIR / "model_E6_best.pt",
    CHECKPOINT_DIR / "model_E7_best.pt",
]

OUTPUT = CHECKPOINT_DIR / "model_soup_E5_E6_E7.pt"


def load_state_dict(path):
    print(f"[LOAD] {path}")
    state = torch.load(path, map_location="cpu")

    # Make sure we received a plain state_dict.
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint format: {type(state)}")

    print(f"[LOAD] Parameters: {len(state)}")
    return state


def average_state_dicts(state_dicts):
    if not state_dicts:
        raise ValueError("No state dictionaries supplied.")

    keys = list(state_dicts[0].keys())

    # Make sure every checkpoint has exactly the same parameter structure.
    for i, sd in enumerate(state_dicts[1:], start=1):
        if list(sd.keys()) != keys:
            raise RuntimeError(
                f"Checkpoint {i} has a different state_dict structure."
            )

    averaged = {}

    for key in keys:
        tensors = [sd[key] for sd in state_dicts]

        # Floating-point parameters: arithmetic mean.
        if torch.is_floating_point(tensors[0]):
            averaged[key] = torch.stack(
                [t.float() for t in tensors], dim=0
            ).mean(dim=0).to(tensors[0].dtype)

        # Integer / boolean buffers: preserve the first checkpoint.
        else:
            averaged[key] = tensors[0].clone()

    return averaged


def main():
    print("=" * 72)
    print("MANIPURI ASR V2 — UNIFORM CHECKPOINT AVERAGING")
    print("=" * 72)

    print("\n[CHECKPOINTS]")
    for path in CHECKPOINTS:
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        print(f"  ✓ {path.name}")

    state_dicts = [
        load_state_dict(path)
        for path in CHECKPOINTS
    ]

    print("\n[AVERAGING]")
    print("  E5 + E6 + E7")
    print("  Uniform weights: 1/3 + 1/3 + 1/3")

    averaged = average_state_dicts(state_dicts)

    torch.save(averaged, OUTPUT)

    print("\n[SAVED]")
    print(f"  {OUTPUT}")
    print(f"  Parameters: {len(averaged)}")

    print("\n" + "=" * 72)
    print("UNIFORM SOUP COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()