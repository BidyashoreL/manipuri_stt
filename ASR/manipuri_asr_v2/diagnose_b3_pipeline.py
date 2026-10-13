from pathlib import Path
import json
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ASR = SCRIPT_DIR.parent

VAL_CSV = ASR / "combined_dataset" / "val.csv"
TRAIN_CSV = ASR / "combined_dataset" / "train.csv"

VOCAB_PATH = (
    ASR
    / "checkpoints"
    / "v2_ctc"
    / "b2_partial_encoder"
    / "vocab.json"
)

CACHE_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "val_logit_cache"
)

NBEST_PATH = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "nbest"
    / "b3_4gram_pooled100_val.json"
)

OUTPUT_DIR = (
    ASR
    / "checkpoints"
    / "v3_ctc"
    / "b3_narrow"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# EDIT DISTANCE
# ============================================================

def edit_ops(ref, hyp):
    """
    Character-level Levenshtein alignment.

    Returns:
        substitutions
        deletions
        insertions
    """

    ref = list(ref)
    hyp = list(hyp)

    n = len(ref)
    m = len(hyp)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "D"

    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                back[i][j] = "C"
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]

                best_cost, best_op = min(choices, key=lambda x: x[0])

                dp[i][j] = best_cost
                back[i][j] = best_op

    i = n
    j = m

    substitutions = 0
    deletions = 0
    insertions = 0

    confusion = Counter()
    deleted_chars = Counter()
    inserted_chars = Counter()

    while i > 0 or j > 0:

        op = back[i][j]

        if op == "C":
            i -= 1
            j -= 1

        elif op == "S":
            substitutions += 1
            confusion[(ref[i - 1], hyp[j - 1])] += 1
            i -= 1
            j -= 1

        elif op == "D":
            deletions += 1
            deleted_chars[ref[i - 1]] += 1
            i -= 1

        elif op == "I":
            insertions += 1
            inserted_chars[hyp[j - 1]] += 1
            j -= 1

        else:
            raise RuntimeError(f"Unknown alignment op: {op}")

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "confusion": confusion,
        "deleted_chars": deleted_chars,
        "inserted_chars": inserted_chars,
        "distance": substitutions + deletions + insertions,
    }


def cer(ref, hyp):
    ref = normalize_text(ref)
    hyp = normalize_text(hyp)

    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0

    return edit_ops(ref, hyp)["distance"] / len(ref)


def wer(ref, hyp):
    ref_words = normalize_text(ref).split()
    hyp_words = normalize_text(hyp).split()

    n = len(ref_words)
    m = len(hyp_words)

    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):

            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                )

    return dp[n][m] / n


# ============================================================
# LOAD VOCAB
# ============================================================

print("=" * 80)
print("B3 CTC PIPELINE DIAGNOSTIC")
print("=" * 80)

print("\n[1/9] Loading vocabulary...")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

id_to_token = {int(v): k for k, v in vocab.items()}

print("Vocabulary size:", len(vocab))

latin_tokens = []
angle_tokens = []
non_meitei_tokens = []

for token in vocab.keys():

    if re.search(r"[A-Za-z]", token):
        latin_tokens.append(token)

    if "<" in token or ">" in token:
        angle_tokens.append(token)

    if token not in ("<blank>", "<unk>", " "):
        if not any(
            "\U0000D100" <= ch <= "\U0000D1FF"
            for ch in token
        ):
            non_meitei_tokens.append(token)

print("\nVocabulary audit:")
print("Latin tokens :", latin_tokens)
print("Angle tokens :", angle_tokens)
print("Other tokens :", non_meitei_tokens[:100])


# ============================================================
# LOAD DATA
# ============================================================

print("\n[2/9] Loading train/validation manifests...")

train_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
val_df = pd.read_csv(VAL_CSV, encoding="utf-8-sig")

train_df["text_norm"] = train_df["text"].map(normalize_text)
val_df["text_norm"] = val_df["text"].map(normalize_text)

print("Train rows:", len(train_df))
print("Val rows  :", len(val_df))


# ============================================================
# LABEL AUDIT
# ============================================================

print("\n[3/9] Auditing labels...")

all_train_text = "\n".join(train_df["text_norm"].tolist())

tag_matches = re.findall(r"<[^>]*>", all_train_text)

latin_rows = train_df["text_norm"].str.contains(
    r"[A-Za-z]",
    regex=True,
    na=False
).sum()

angle_rows = train_df["text_norm"].str.contains(
    r"[<>]",
    regex=True,
    na=False
).sum()

replacement_rows = train_df["text_norm"].str.contains(
    "\ufffd",
    regex=False,
    na=False
).sum()

print("Training rows containing Latin :", latin_rows)
print("Training rows containing < >   :", angle_rows)
print("Training rows containing       :", replacement_rows)

print("\nMost common tags:")
print(Counter(tag_matches).most_common(20))


# ============================================================
# LOAD LOGIT CACHE
# ============================================================

print("\n[4/9] Loading validation logits...")

logits_path = CACHE_DIR / "logits.npy"
offsets_path = CACHE_DIR / "offsets.npy"
texts_path = CACHE_DIR / "texts.json"
metadata_path = CACHE_DIR / "metadata.json"

logits = np.load(logits_path, mmap_mode="r")
offsets = np.load(offsets_path)

with open(texts_path, "r", encoding="utf-8") as f:
    cache_texts = json.load(f)

with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

print("Logit shape:", logits.shape)
print("Cache samples:", len(cache_texts))
print("Cache metadata:", metadata)


# ============================================================
# LOAD N-BEST
# ============================================================

print("\n[5/9] Loading pooled N-best...")

with open(NBEST_PATH, "r", encoding="utf-8") as f:
    nbest = json.load(f)

samples = nbest["samples"]

print("N-best samples:", len(samples))


# ============================================================
# VALIDATE ORDER
# ============================================================

print("\n[6/9] Validating sample alignment...")

assert len(val_df) == len(samples)
assert len(cache_texts) == len(samples)

for i in range(len(samples)):
    a = normalize_text(val_df.iloc[i]["text"])
    b = normalize_text(cache_texts[i])
    c = normalize_text(samples[i]["reference"])

    if not (a == b == c):
        raise RuntimeError(
            f"Reference mismatch at sample {i}\n"
            f"VAL  : {a}\n"
            f"CACHE: {b}\n"
            f"NBEST: {c}"
        )

print("✓ Validation / cache / N-best order matches.")


# ============================================================
# GLOBAL DIAGNOSTICS
# ============================================================

print("\n[7/9] Computing sample-level diagnostics...")

rows = []

global_confusion = Counter()
global_deleted = Counter()
global_inserted = Counter()

aggregate = {
    "greedy": {
        "sub": 0,
        "del": 0,
        "ins": 0,
        "ref_chars": 0,
    },
    "beam": {
        "sub": 0,
        "del": 0,
        "ins": 0,
        "ref_chars": 0,
    },
    "oracle100": {
        "sub": 0,
        "del": 0,
        "ins": 0,
        "ref_chars": 0,
    },
}


def oracle_candidate(hypotheses, ref, k):
    candidates = hypotheses[:k]

    best = None

    for rank, h in enumerate(candidates, start=1):

        text = normalize_text(h["text"])
        score = cer(ref, text)

        if best is None or score < best[0]:
            best = (score, rank, text)

    return best


for idx, sample in enumerate(samples):

    if idx % 100 == 0:
        print(f"  Processed {idx}/{len(samples)}")

    ref = normalize_text(sample["reference"])
    hypotheses = sample["hypotheses"]

    # --------------------------------------------------------
    # 4G / original top-1
    # --------------------------------------------------------

    beam_hyp = normalize_text(hypotheses[0]["text"])

    # --------------------------------------------------------
    # Greedy from logits
    # --------------------------------------------------------

    start = int(offsets[idx])
    end = int(offsets[idx + 1])

    sample_logits = np.asarray(logits[start:end])

    greedy_ids = np.argmax(sample_logits, axis=-1)

    # CTC collapse
    collapsed = []

    previous = None

    for token_id in greedy_ids:

        token_id = int(token_id)

        if token_id == 0:
            previous = token_id
            continue

        if token_id == previous:
            continue

        token = id_to_token.get(token_id, "")

        if token not in ("<blank>", "<unk>"):
            collapsed.append(token)

        previous = token_id

    greedy_hyp = normalize_text("".join(collapsed))

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    oracle5 = oracle_candidate(hypotheses, ref, 5)
    oracle10 = oracle_candidate(hypotheses, ref, 10)
    oracle20 = oracle_candidate(hypotheses, ref, 20)
    oracle50 = oracle_candidate(hypotheses, ref, 50)
    oracle100 = oracle_candidate(hypotheses, ref, 100)

    # --------------------------------------------------------
    # Edit operations
    # --------------------------------------------------------

    greedy_ops = edit_ops(ref, greedy_hyp)
    beam_ops = edit_ops(ref, beam_hyp)
    oracle_ops = edit_ops(ref, oracle100[2])

    global_confusion.update(greedy_ops["confusion"])
    global_confusion.update(beam_ops["confusion"])

    global_deleted.update(greedy_ops["deleted_chars"])
    global_deleted.update(beam_ops["deleted_chars"])

    global_inserted.update(greedy_ops["inserted_chars"])
    global_inserted.update(beam_ops["inserted_chars"])

    for name, ops in [
        ("greedy", greedy_ops),
        ("beam", beam_ops),
        ("oracle100", oracle_ops),
    ]:
        aggregate[name]["sub"] += ops["substitutions"]
        aggregate[name]["del"] += ops["deletions"]
        aggregate[name]["ins"] += ops["insertions"]
        aggregate[name]["ref_chars"] += len(ref)

    # --------------------------------------------------------
    # Duration / length
    # --------------------------------------------------------

    source_row = val_df.iloc[idx]

    duration = float(source_row["duration"]) if "duration" in val_df.columns else np.nan

    ref_chars = len(ref)
    hyp_chars = len(beam_hyp)

    ref_words = len(ref.split())
    hyp_words = len(beam_hyp.split())

    ratio = hyp_chars / max(ref_chars, 1)

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    exact_rank = None

    for rank, h in enumerate(hypotheses[:100], start=1):
        if normalize_text(h["text"]) == ref:
            exact_rank = rank
            break

    if exact_rank is None:
        category = "ABSENT_TOP100"
    elif exact_rank <= 20:
        category = "EXACT_TOP20"
    elif exact_rank <= 50:
        category = "EXACT_21_50"
    else:
        category = "EXACT_51_100"

    # --------------------------------------------------------
    # Suspicious conditions
    # --------------------------------------------------------

    has_latin_ref = bool(re.search(r"[A-Za-z]", ref))
    has_latin_hyp = bool(re.search(r"[A-Za-z]", beam_hyp))
    has_angle_ref = bool(re.search(r"[<>]", ref))
    has_angle_hyp = bool(re.search(r"[<>]", beam_hyp))

    # --------------------------------------------------------
    # Row
    # --------------------------------------------------------

    rows.append({
        "sample_index": idx,
        "source": source_row.get("source", ""),
        "speaker_id": source_row.get("speaker_id", ""),

        "duration_sec": duration,

        "ref_chars": ref_chars,
        "beam_chars": hyp_chars,
        "char_length_ratio": ratio,

        "ref_words": ref_words,
        "beam_words": hyp_words,

        "greedy": greedy_hyp,
        "beam4g": beam_hyp,

        "greedy_cer": cer(ref, greedy_hyp),
        "beam4g_cer": cer(ref, beam_hyp),

        "greedy_wer": wer(ref, greedy_hyp),
        "beam4g_wer": wer(ref, beam_hyp),

        "oracle5_cer": oracle5[0],
        "oracle10_cer": oracle10[0],
        "oracle20_cer": oracle20[0],
        "oracle50_cer": oracle50[0],
        "oracle100_cer": oracle100[0],

        "oracle5_rank": oracle5[1],
        "oracle10_rank": oracle10[1],
        "oracle20_rank": oracle20[1],
        "oracle50_rank": oracle50[1],
        "oracle100_rank": oracle100[1],

        "exact_rank": exact_rank,
        "category": category,

        "greedy_sub": greedy_ops["substitutions"],
        "greedy_del": greedy_ops["deletions"],
        "greedy_ins": greedy_ops["insertions"],

        "beam_sub": beam_ops["substitutions"],
        "beam_del": beam_ops["deletions"],
        "beam_ins": beam_ops["insertions"],

        "has_latin_ref": has_latin_ref,
        "has_latin_hyp": has_latin_hyp,
        "has_angle_ref": has_angle_ref,
        "has_angle_hyp": has_angle_hyp,

        "reference": ref,
        "hypothesis": beam_hyp,
        "oracle100_hypothesis": oracle100[2],
    })


df = pd.DataFrame(rows)


# ============================================================
# LENGTH BINS
# ============================================================

def length_bin(n):
    if n <= 5:
        return "01_1-5"
    elif n <= 10:
        return "02_6-10"
    elif n <= 20:
        return "03_11-20"
    elif n <= 40:
        return "04_21-40"
    elif n <= 80:
        return "05_41-80"
    elif n <= 120:
        return "06_81-120"
    else:
        return "07_121+"


def duration_bin(x):
    if x < 1:
        return "01_<1s"
    elif x < 2:
        return "02_1-2s"
    elif x < 4:
        return "03_2-4s"
    elif x < 8:
        return "04_4-8s"
    elif x < 12:
        return "05_8-12s"
    else:
        return "06_12+s"


df["length_bin"] = df["ref_chars"].map(length_bin)
df["duration_bin"] = df["duration_sec"].map(duration_bin)


# ============================================================
# SUMMARY TABLES
# ============================================================

def aggregate_group(group):

    ref_chars = group["ref_chars"].sum()

    return pd.Series({
        "samples": len(group),

        "micro_CER_4G":
            (
                group["beam_sub"].sum()
                + group["beam_del"].sum()
                + group["beam_ins"].sum()
            )
            / max(ref_chars, 1),

        "micro_CER_oracle100":
            (
                group.apply(
                    lambda r: edit_ops(
                        r["reference"],
                        r["oracle100_hypothesis"]
                    )["distance"],
                    axis=1
                ).sum()
            )
            / max(ref_chars, 1),

        "mean_CER_4G": group["beam4g_cer"].mean(),
        "mean_CER_oracle100": group["oracle100_cer"].mean(),

        "mean_ref_chars": group["ref_chars"].mean(),
        "mean_duration_sec": group["duration_sec"].mean(),

        "exact_top100_pct":
            (group["exact_rank"].notna().mean() * 100),
    })


length_summary = (
    df.groupby("length_bin", sort=True)
      .apply(aggregate_group, include_groups=False)
      .reset_index()
)

duration_summary = (
    df.groupby("duration_bin", sort=True)
      .apply(aggregate_group, include_groups=False)
      .reset_index()
)


# ============================================================
# SOURCE SUMMARY
# ============================================================

if "source" in df.columns:

    source_summary = (
        df.groupby("source", dropna=False)
          .apply(aggregate_group, include_groups=False)
          .reset_index()
    )

else:
    source_summary = pd.DataFrame()


# ============================================================
# ERROR TYPE TOTALS
# ============================================================

error_totals = pd.DataFrame([
    {
        "decoder": "greedy",
        "substitutions": aggregate["greedy"]["sub"],
        "deletions": aggregate["greedy"]["del"],
        "insertions": aggregate["greedy"]["ins"],
        "total_errors":
            aggregate["greedy"]["sub"]
            + aggregate["greedy"]["del"]
            + aggregate["greedy"]["ins"],
        "reference_characters": aggregate["greedy"]["ref_chars"],
    },
    {
        "decoder": "4gram",
        "substitutions": aggregate["beam"]["sub"],
        "deletions": aggregate["beam"]["del"],
        "insertions": aggregate["beam"]["ins"],
        "total_errors":
            aggregate["beam"]["sub"]
            + aggregate["beam"]["del"]
            + aggregate["beam"]["ins"],
        "reference_characters": aggregate["beam"]["ref_chars"],
    },
])


error_totals["CER"] = (
    error_totals["total_errors"]
    / error_totals["reference_characters"]
)


# ============================================================
# SAVE
# ============================================================

print("\n[8/9] Saving diagnostic files...")

df.to_csv(
    OUTPUT_DIR / "sample_diagnostics.csv",
    index=False,
    encoding="utf-8-sig"
)

length_summary.to_csv(
    OUTPUT_DIR / "length_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

duration_summary.to_csv(
    OUTPUT_DIR / "duration_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

source_summary.to_csv(
    OUTPUT_DIR / "source_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

error_totals.to_csv(
    OUTPUT_DIR / "error_operations.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    OUTPUT_DIR / "character_confusions.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "substitutions": [
                {
                    "reference": a,
                    "hypothesis": b,
                    "count": c,
                }
                for (a, b), c
                in global_confusion.most_common(200)
            ],
            "deletions": [
                {
                    "character": a,
                    "count": c,
                }
                for a, c
                in global_deleted.most_common(100)
            ],
            "insertions": [
                {
                    "character": a,
                    "count": c,
                }
                for a, c
                in global_inserted.most_common(100)
            ],
        },
        f,
        ensure_ascii=False,
        indent=2,
    )


# ============================================================
# PRINT REPORT
# ============================================================

print("\n[9/9] FINAL DIAGNOSTIC REPORT")
print("=" * 80)

print("\nDATASET")
print("-" * 80)
print("Train samples:", len(train_df))
print("Validation samples:", len(val_df))

print("\nVOCABULARY")
print("-" * 80)
print("Vocabulary size:", len(vocab))
print("Latin tokens:", latin_tokens)
print("Angle tokens:", angle_tokens)
print("Other non-Meitei tokens:", non_meitei_tokens[:50])

print("\nLABEL AUDIT")
print("-" * 80)
print("Train rows with Latin:", latin_rows)
print("Train rows with < >:", angle_rows)
print("Train rows with replacement char:", replacement_rows)

print("\nERROR OPERATIONS")
print("-" * 80)
print(error_totals.to_string(index=False))

print("\nLENGTH SUMMARY")
print("-" * 80)
print(length_summary.to_string(index=False))

print("\nDURATION SUMMARY")
print("-" * 80)
print(duration_summary.to_string(index=False))

if not source_summary.empty:
    print("\nSOURCE SUMMARY")
    print("-" * 80)
    print(source_summary.to_string(index=False))

print("\nTOP CHARACTER CONFUSIONS")
print("-" * 80)

for (a, b), count in global_confusion.most_common(30):
    print(f"{a!r} -> {b!r} : {count}")

print("\nTOP DELETIONS")
print("-" * 80)

for char, count in global_deleted.most_common(20):
    print(f"{char!r} : {count}")

print("\nTOP INSERTIONS")
print("-" * 80)

for char, count in global_inserted.most_common(20):
    print(f"{char!r} : {count}")

print("\nWORST 20 BY 4-GRAM CER")
print("-" * 80)

worst = df.sort_values(
    "beam4g_cer",
    ascending=False
).head(20)

for _, r in worst.iterrows():

    print(
        f"\n[{int(r['sample_index'])}] "
        f"duration={r['duration_sec']:.2f}s "
        f"chars={int(r['ref_chars'])} "
        f"category={r['category']}"
    )

    print("REF :", r["reference"])
    print("4G  :", r["hypothesis"])
    print("ORCL:", r["oracle100_hypothesis"])

    print(
        f"CER : {r['beam4g_cer']*100:.2f}% "
        f"-> oracle100 "
        f"{r['oracle100_cer']*100:.2f}%"
    )


print("\nFILES")
print("-" * 80)

print("Sample diagnostics :",
      OUTPUT_DIR / "sample_diagnostics.csv")

print("Length summary     :",
      OUTPUT_DIR / "length_summary.csv")

print("Duration summary   :",
      OUTPUT_DIR / "duration_summary.csv")

print("Source summary     :",
      OUTPUT_DIR / "source_summary.csv")

print("Error operations   :",
      OUTPUT_DIR / "error_operations.csv")

print("Character confusion:",
      OUTPUT_DIR / "character_confusions.json")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)