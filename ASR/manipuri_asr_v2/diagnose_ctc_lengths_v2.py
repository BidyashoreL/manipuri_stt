from pathlib import Path
import statistics
import torch
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
ASR = ROOT / "ASR"

ALL_CSV = ASR / "combined_dataset" / "all.csv"
TRAIN_CSV = ASR / "combined_dataset" / "train.csv"
VAL_CSV = ASR / "combined_dataset" / "val.csv"
CACHE_ROOT = ASR / "feature_cache"


def load_all():
    df = pd.read_csv(ALL_CSV)

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


def load_split(path):
    df = pd.read_csv(path)

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return df


def make_manifest_lookup(all_df):
    lookup = {}

    for manifest_index, row in all_df.iterrows():

        source = str(row["source"])

        lookup_key = (
            source,
            str(row["audio_file"]),
        )

        lookup[lookup_key] = manifest_index

        # Also support audio_path where available.
        if "audio_path" in row and pd.notna(row["audio_path"]):
            lookup[
                (
                    source,
                    str(row["audio_path"]),
                )
            ] = manifest_index

    return lookup


def resolve_manifest_index(row, lookup):

    source = str(row["source"])

    candidates = []

    if "audio_file" in row and pd.notna(row["audio_file"]):
        candidates.append(
            (
                source,
                str(row["audio_file"]),
            )
        )

    if "audio_path" in row and pd.notna(row["audio_path"]):
        candidates.append(
            (
                source,
                str(row["audio_path"]),
            )
        )

    for key in candidates:
        if key in lookup:
            return lookup[key]

    return None


def repeated_adjacent_chars(text):
    count = 0

    for i in range(1, len(text)):
        if text[i] == text[i - 1]:
            count += 1

    return count


def analyze(name, df, lookup):

    rows = []

    missing = []
    unresolved = []

    for _, row in df.iterrows():

        manifest_index = resolve_manifest_index(
            row,
            lookup
        )

        if manifest_index is None:
            unresolved.append(row)
            continue

        source = str(row["source"])

        cache_path = (
            CACHE_ROOT
            / source
            / f"{manifest_index:06d}.pt"
        )

        if not cache_path.exists():
            missing.append(
                (
                    manifest_index,
                    source,
                    str(row["audio_file"]),
                )
            )
            continue

        try:

            obj = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )

            features = obj["features"]

            frames = int(features.shape[0])

            text = str(row["text"]).strip()

            target_len = len(text)

            repeats = repeated_adjacent_chars(text)

            required_frames = target_len + repeats

            rows.append(
                {
                    "manifest_index": manifest_index,
                    "source": source,
                    "frames": frames,
                    "target_len": target_len,
                    "repeats": repeats,
                    "required_frames": required_frames,
                    "ratio": (
                        required_frames / frames
                        if frames > 0
                        else float("inf")
                    ),
                    "text": text,
                }
            )

        except Exception as e:

            print(
                f"[ERROR] {cache_path}: {e}"
            )

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"CSV samples              : {len(df):,}"
    )

    print(
        f"Resolved to manifest     : "
        f"{len(rows) + len(missing):,}"
    )

    print(
        f"Unresolved               : "
        f"{len(unresolved):,}"
    )

    print(
        f"Missing cache            : "
        f"{len(missing):,}"
    )

    if not rows:
        print("No usable cache samples.")
        return

    frames = [r["frames"] for r in rows]
    targets = [r["target_len"] for r in rows]
    required = [r["required_frames"] for r in rows]
    ratios = [r["ratio"] for r in rows]

    n = len(rows)

    def percentage(condition):
        return (
            100.0 * sum(condition) / n
        )

    print()
    print("FEATURE FRAMES")

    print(
        f"  min                    : "
        f"{min(frames)}"
    )

    print(
        f"  max                    : "
        f"{max(frames)}"
    )

    print(
        f"  mean                   : "
        f"{statistics.mean(frames):.2f}"
    )

    print(
        f"  median                 : "
        f"{statistics.median(frames):.2f}"
    )

    print()
    print("TARGET CHARACTERS")

    print(
        f"  min                    : "
        f"{min(targets)}"
    )

    print(
        f"  max                    : "
        f"{max(targets)}"
    )

    print(
        f"  mean                   : "
        f"{statistics.mean(targets):.2f}"
    )

    print(
        f"  median                 : "
        f"{statistics.median(targets):.2f}"
    )

    print()
    print("CTC REQUIRED FRAMES")
    print(
        "required = target length + "
        "adjacent repeated characters"
    )

    print(
        f"  min                    : "
        f"{min(required)}"
    )

    print(
        f"  max                    : "
        f"{max(required)}"
    )

    print(
        f"  mean                   : "
        f"{statistics.mean(required):.2f}"
    )

    print(
        f"  median                 : "
        f"{statistics.median(required):.2f}"
    )

    print()
    print("CTC FEASIBILITY")

    impossible = [
        r["required_frames"] > r["frames"]
        for r in rows
    ]

    exact_or_tight = [
        r["required_frames"] >= r["frames"]
        for r in rows
    ]

    pressure_80 = [
        r["required_frames"] > 0.80 * r["frames"]
        for r in rows
    ]

    pressure_60 = [
        r["required_frames"] > 0.60 * r["frames"]
        for r in rows
    ]

    pressure_50 = [
        r["required_frames"] > 0.50 * r["frames"]
        for r in rows
    ]

    print(
        f"  IMPOSSIBLE (> frames) : "
        f"{sum(impossible):,} "
        f"({percentage(impossible):.2f}%)"
    )

    print(
        f"  >= frames             : "
        f"{sum(exact_or_tight):,} "
        f"({percentage(exact_or_tight):.2f}%)"
    )

    print(
        f"  >80% frames           : "
        f"{sum(pressure_80):,} "
        f"({percentage(pressure_80):.2f}%)"
    )

    print(
        f"  >60% frames           : "
        f"{sum(pressure_60):,} "
        f"({percentage(pressure_60):.2f}%)"
    )

    print(
        f"  >50% frames           : "
        f"{sum(pressure_50):,} "
        f"({percentage(pressure_50):.2f}%)"
    )

    print()
    print("WORST 15 SAMPLES")

    worst = sorted(
        rows,
        key=lambda x: x["ratio"],
        reverse=True,
    )[:15]

    for r in worst:

        preview = (
            r["text"]
            .replace("\n", " ")
        )

        if len(preview) > 80:
            preview = preview[:80] + "..."

        print(
            f"\n"
            f"  manifest={r['manifest_index']:4d} "
            f"source={r['source']:<15} "
            f"frames={r['frames']:4d} "
            f"target={r['target_len']:4d} "
            f"repeats={r['repeats']:3d} "
            f"required={r['required_frames']:4d} "
            f"ratio={r['ratio']:.3f}"
        )

        print(
            f"    text: {preview}"
        )

    print()
    print("SAMPLE CACHE PATHS")

    for r in rows[:5]:

        path = (
            CACHE_ROOT
            / r["source"]
            / f"{r['manifest_index']:06d}.pt"
        )

        print(
            f"  {path}"
        )


def main():

    print("=" * 80)
    print("MANIPURI ASR — CTC TEMPORAL DIAGNOSTIC V2")
    print("=" * 80)

    all_df = load_all()

    train_df = load_split(TRAIN_CSV)
    val_df = load_split(VAL_CSV)

    print(
        f"[DATA] all   = {len(all_df):,}"
    )

    print(
        f"[DATA] train = {len(train_df):,}"
    )

    print(
        f"[DATA] val   = {len(val_df):,}"
    )

    lookup = make_manifest_lookup(all_df)

    print(
        f"[DATA] manifest lookup entries = "
        f"{len(lookup):,}"
    )

    analyze(
        "TRAIN",
        train_df,
        lookup
    )

    analyze(
        "VALIDATION",
        val_df,
        lookup
    )

    print()
    print("=" * 80)
    print("DIAGNOSTIC V2 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()