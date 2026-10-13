import re


class Step1Normalizer:
    # Retains ONLY Latin characters, digits, spaces, and the Meitei Mayek script block.
    # Acts as a strict firewall against Thai, Devanagari, or Cyrillic acoustic hallucinations.
    VALID_CHARS_PATTERN = re.compile(r"[^a-zA-Z0-9\s\uABC0-\uABFF]")
    BRACKET_PATTERN = re.compile(r"\[.*?\]")

    def __init__(self, max_allowed_repeats: int = 2):
        self.max_allowed_repeats = max_allowed_repeats

    def clean_characters(self, text: str) -> str:
        """Lowercases text, eliminates bracketed noise, and enforces strict character filtering."""
        if not text:
            return ""

        text = text.lower()
        text = self.BRACKET_PATTERN.sub(" ", text)
        text = self.VALID_CHARS_PATTERN.sub(" ", text)

        return " ".join(text.split())

    def deduplicate_loops(self, text: str) -> str:
        """Truncates runaway acoustic repetition loops from Whisper decoder stalls."""
        words = text.split()
        if not words:
            return ""

        cleaned_words = []
        repeat_count = 1

        for i, word in enumerate(words):
            if i == 0:
                cleaned_words.append(word)
                continue

            if word == words[i - 1]:
                repeat_count += 1
            else:
                repeat_count = 1

            if repeat_count <= self.max_allowed_repeats:
                cleaned_words.append(word)

        return " ".join(cleaned_words)

    def process(self, raw_text: str) -> str:
        """Executes the complete normalization layer pipeline."""
        if not raw_text or not raw_text.strip():
            return ""
        text_cleaned = self.clean_characters(raw_text)
        return self.deduplicate_loops(text_cleaned)
