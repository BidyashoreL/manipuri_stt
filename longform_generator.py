import random
from typing import List, Dict


class LowResourceLongFormGenerator:
    """
    Converts sentence-level low-resource data into structured, pseudo-continuous
    long-form corpora to maintain robust time-bounding alignments.
    """

    def __init__(
        self,
        target_chunk_duration_sec: float = 30.0,
        average_words_per_sec: float = 2.5,
    ):
        self.target_duration = target_chunk_duration_sec
        self.words_per_sec = average_words_per_sec

    def estimate_duration(self, text: str) -> float:
        word_count = len(text.split())
        return max(1.0, word_count / self.words_per_sec)

    def generate_synthetic_longform(
        self, sentence_dataset: List[Dict[str, str]]
    ) -> List[Dict]:
        """
        Takes sentence records, stitches them sequentially, and injects precise
        synthetic temporal markers to force Whisper's multi-head attention to respect time bounds.
        """
        longform_corpus = []
        current_chunk_text = []
        current_duration = 0.0
        start_time = 0.0

        dataset_pool = list(sentence_dataset)
        random.shuffle(dataset_pool)

        for record in dataset_pool:
            text = record.get("text", "").strip()
            if not text:
                continue

            sentence_duration = self.estimate_duration(text)
            end_time = start_time + sentence_duration

            # Format matching Whisper's long-form tokenization constraints
            timestamped_segment = f"<|{start_time:.2f}|> {text} <|{end_time:.2f}|>"
            current_chunk_text.append(timestamped_segment)

            current_duration += sentence_duration
            start_time = end_time

            if current_duration >= self.target_duration:
                longform_corpus.append(
                    {
                        "text": " ".join(current_chunk_text),
                        "total_duration": current_duration,
                    }
                )
                current_chunk_text = []
                current_duration = 0.0
                start_time = 0.0

        if current_chunk_text:
            longform_corpus.append(
                {
                    "text": " ".join(current_chunk_text),
                    "total_duration": current_duration,
                }
            )

        return longform_corpus
