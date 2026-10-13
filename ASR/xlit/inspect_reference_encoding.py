from pathlib import Path

files = [
    Path("data/transcribed.txt"),
    Path("data/corrected.txt"),
    Path("data/corpus/literature_corpus/words.txt"),
]

for path in files:
    print(f"\n--- {path} ---")
    raw = path.read_bytes()

    print("File size:", len(raw), "bytes")
    print("UTF-8 BOM:", raw.startswith(b"\xef\xbb\xbf"))
    print("First 100 raw bytes:", repr(raw[:100]))

    for encoding in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            text = raw.decode(encoding)
            print(f"\nDecoded as {encoding}:")
            print(repr(text.splitlines()[:3]))
        except UnicodeDecodeError as exc:
            print(f"{encoding}: decoding failed: {exc}")

