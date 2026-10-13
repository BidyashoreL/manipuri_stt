from pathlib import Path
import pandas as pd
import unicodedata
import re
import json

# ============================================================
# PATHS
# ============================================================

ASR_DIR = Path(__file__).resolve().parent

TRAIN_CSV = ASR_DIR.parent / "combined_dataset" / "train.csv"
OUTPUT_DIR = ASR_DIR / "kenlm_experiment"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

UNIGRAM_FILE = OUTPUT_DIR / "manipuri_unigrams.txt"
CORPUS_FILE = OUTPUT_DIR / "manipuri_corpus.txt"


# ============================================================
# NORMALIZATION
# Same basic normalization used by B2
# ============================================================

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 70)
print("PREPARING MANIPURI UNIGRAM DICTIONARY")
print("=" * 70)

print(f"[INFO] Training CSV: {TRAIN_CSV}")

df = pd.read_csv(TRAIN_CSV)

print(f"[INFO] Training samples: {len(df)}")
print(f"[INFO] Columns: {list(df.columns)}")

if "text" not in df.columns:
    raise RuntimeError(
        f"'text' column not found.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# EXTRACT SENTENCES
# ============================================================

sentences = []

for value in df["text"].dropna():

    text = normalize_text(value)

    if text:
        sentences.append(text)


print(f"[INFO] Non-empty transcripts: {len(sentences)}")


# ============================================================
# BUILD WORD UNIGRAM DICTIONARY
# ============================================================

unigrams = set()

for sentence in sentences:

    words = sentence.split()

    for word in words:

        word = word.strip()

        if word:
            unigrams.add(word)


unigram_list = sorted(unigrams)


print(f"[INFO] Unique Manipuri words: {len(unigram_list)}")


# ============================================================
# SAVE UNIGRAM DICTIONARY
# ============================================================

with open(UNIGRAM_FILE, "w", encoding="utf-8") as f:

    for word in unigram_list:
        f.write(word + "\n")


# ============================================================
# SAVE NORMALIZED CORPUS
# ============================================================

with open(CORPUS_FILE, "w", encoding="utf-8") as f:

    for sentence in sentences:
        f.write(sentence + "\n")


# ============================================================
# STATISTICS
# ============================================================

word_counts = {}

for sentence in sentences:

    for word in sentence.split():

        word_counts[word] = word_counts.get(word, 0) + 1


print()
print("=" * 70)
print("RESULT")
print("=" * 70)

print(f"Training sentences : {len(sentences):,}")
print(f"Unique words       : {len(unigram_list):,}")
print(f"Corpus file        : {CORPUS_FILE}")
print(f"Unigram file       : {UNIGRAM_FILE}")

print()
print("Top 20 most frequent words:")

for word, count in sorted(
    word_counts.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:

    print(f"  {word} : {count:,}")

print()
print("[DONE]")