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


def load_csv(path):
    df = pd.read_csv(path)

    if "text" in df.columns:
        df["text"] = (
            df["text"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    return df


def build_exact_lookup(all_df):

    lookup = {}

    for i, row in all_df.reset_index(drop=True).iterrows():

        source = str(row.get("source", ""))

        if source == "indicvoices":

            key = (
                "indicvoices",
                str(row.get("parquet_file", "")),
                int(row.get("parquet_row", -1)),
            )

        else:

            key = (
                source,
                str(row.get("audio_path", "")),
            )

        lookup[key] = i

    return lookup


def get_manifest_index(row, lookup):

    source = str(row.get("source", ""))

    if source == "indicvoices":

        key = (
            "indicvoices",
            str(row.get("parquet_file", "")),
            int(row.get("parquet_row", -1)),
        )

    else:

        key = (
            source,
            str(row.get("audio_path", "")),
        )

    return lookup.get(key, -1)


def build_cache_index():

    index = {}

    count = 0

    for source_dir in CACHE_ROOT.iterdir():

        if not source_dir.is_dir():
            continue

        source = source_dir.name

        for p in source_dir.glob("*.pt"):

            try:
                manifest_index = int(p.stem)
            except ValueError:
                continue

            index[(source, manifest_index)] = p
            count += 1

    print(f"[CACHE] Indexed feature files: {count:,}")

    return index


def repeated_adjacent_tokens(text):

    count = 0

    for i in range(1, len(text)):

        if text[i] == text[i - 1]:
            count += 1

    return count


def analyze(name, df, lookup, cache_index):

    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(f"CSV samples                  : {len(df):,}")

    resolved = 0
    unresolved = 0
    missing_cache = 0
    bad_features = 0

    rows = []

    for split_index, row in df.iterrows():

        manifest_index = get_manifest_index(
            row,
            lookup
        )

        if manifest_index < 0:

            unresolved += 1

            continue

        resolved += 1

        source = str(row["source"])

        cache_key = (
            source,
            manifest_index,
        )

        cache_path = cache_index.get(cache_key)

        if cache_path is None:

            missing_cache += 1

            continue

        try:

            payload = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )

            features = payload["features"]

            if not isinstance(features, torch.Tensor):
                bad_features += 1
                continue

            if features.ndim != 2:
                bad_features += 1
                continue

            frames = int(features.shape[0])
            feature_dim = int(features.shape[1])

            text = str(row["text"]).strip()

            target_len = len(text)

            repeats = repeated_adjacent_tokens(text)

            required_frames = target_len + repeats

            rows.append(
                {
                    "split_index": split_index,
                    "manifest_index": manifest_index,
                    "source": source,
                    "frames": frames,
                    "feature_dim": feature_dim,
                    "target_len": target_len,
                    "repeats": repeats,
                    "required_frames": required_frames,
                    "ratio": (
                        required_frames / frames
                        if frames > 0
                        else float("inf")
                    ),
                    "text": text,
                    "cache_path": str(cache_path),
                }
            )

        except Exception as e:

            print(
                f"[ERROR] {cache_path}: {e}"
            )

    print()
    print("MAPPING")
    print(f"  resolved                    : {resolved:,}")
    print(f"  unresolved                  : {unresolved:,}")
    print(f"  missing cache               : {missing_cache:,}")
    print(f"  bad feature tensors         : {bad_features:,}")
    print(f"  usable samples              : {len(rows):,}")

    if not rows:
        return

    frames = [r["frames"] for r in rows]
    dims = [r["feature_dim"] for r in rows]
    targets = [r["target_len"] for r in rows]
    required = [r["required_frames"] for r in rows]
    ratios = [r["ratio"] for r in rows]

    n = len(rows)

    def pct(values):

        return (
            100.0 * sum(values) / n
        )

    print()
    print("FEATURE SHAPES")

    print(
        f"  dimensions                 : "
        f"{sorted(set(dims))}"
    )

    print(
        f"  frames min                 : "
        f"{min(frames)}"
    )

    print(
        f"  frames max                 : "
        f"{max(frames)}"
    )

    print(
        f"  frames mean                : "
        f"{statistics.mean(frames):.2f}"
    )

    print(
        f"  frames median              : "
        f"{statistics.median(frames):.2f}"
    )

    print()
    print("TARGET LENGTH")

    print(
        f"  min                       : "
        f"{min(targets)}"
    )

    print(
        f"  max                       : "
        f"{max(targets)}"
    )

    print(
        f"  mean                      : "
        f"{statistics.mean(targets):.2f}"
    )

    print(
        f"  median                    : "
        f"{statistics.median(targets):.2f}"
    )

    print()
    print("CTC REQUIRED FRAMES")

    print(
        f"  min                       : "
        f"{min(required)}"
    )

    print(
        f"  max                       : "
        f"{max(required)}"
    )

    print(
        f"  mean                      : "
        f"{statistics.mean(required):.2f}"
    )

    print(
        f"  median                    : "
        f"{statistics.median(required):.2f}"
    )

    impossible = [
        r["required_frames"] > r["frames"]
        for r in rows
    ]

    tight = [
        r["required_frames"] >= r["frames"]
        for r in rows
    ]

    over80 = [
        r["required_frames"] > 0.80 * r["frames"]
        for r in rows
    ]

    over60 = [
        r["required_frames"] > 0.60 * r["frames"]
        for r in rows
    ]

    over50 = [
        r["required_frames"] > 0.50 * r["frames"]
        for r in rows
    ]

    print()
    print("CTC FEASIBILITY")

    print(
        f"  IMPOSSIBLE                 : "
        f"{sum(impossible):,} "
        f"({pct(impossible):.2f}%)"
    )

    print(
        f"  >= frames                  : "
        f"{sum(tight):,} "
        f"({pct(tight):.2f}%)"
    )

    print(
        f"  >80% frames                : "
        f"{sum(over80):,} "
        f"({pct(over80):.2f}%)"
    )

    print(
        f"  >60% frames                : "
        f"{sum(over60):,} "
        f"({pct(over60):.2f}%)"
    )

    print(
        f"  >50% frames                : "
        f"{sum(over50):,} "
        f"({pct(over50):.2f}%)"
    )

    print()
    print("SAMPLE MAPPING CHECK")

    for r in rows[:10]:

        print(
            f"\n"
            f"  split={r['split_index']:4d} "
            f"manifest={r['manifest_index']:4d} "
            f"source={r['source']}"
        )

        print(
            f"  frames={r['frames']} "
            f"dim={r['feature_dim']} "
            f"target={r['target_len']} "
            f"required={r['required_frames']}"
        )

        print(
            f"  cache={r['cache_path']}"
        )

        preview = r["text"].replace("\n", " ")

        if len(preview) > 120:
            preview = preview[:120] + "..."

        print(
            f"  text={preview}"
        )

    print()
    print("WORST CTC SAMPLES")

    worst = sorted(
        rows,
        key=lambda x: x["ratio"],
        reverse=True
    )[:15]

    for r in worst:

        preview = r["text"].replace("\n", " ")

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
            f"    {preview}"
        )


def main():

    print("=" * 80)
    print("MANIPURI ASR — FINAL CTC DATA INTEGRITY DIAGNOSTIC")
    print("=" * 80)

    print(f"[PATH] all   : {ALL_CSV}")
    print(f"[PATH] train : {TRAIN_CSV}")
    print(f"[PATH] val   : {VAL_CSV}")
    print(f"[PATH] cache : {CACHE_ROOT}")

    all_df = load_csv(ALL_CSV)
    train_df = load_csv(TRAIN_CSV)
    val_df = load_csv(VAL_CSV)

    print()
    print(f"[DATA] all   = {len(all_df):,}")
    print(f"[DATA] train = {len(train_df):,}")
    print(f"[DATA] val   = {len(val_df):,}")

    lookup = build_exact_lookup(all_df)

    print(
        f"[DATA] exact manifest lookup = "
        f"{len(lookup):,}"
    )

    cache_index = build_cache_index()

    analyze(
        "TRAIN",
        train_df,
        lookup,
        cache_index
    )

    analyze(
        "VALIDATION",
        val_df,
        lookup,
        cache_index
    )

    print()
    print("=" * 80)
    print("FINAL DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()