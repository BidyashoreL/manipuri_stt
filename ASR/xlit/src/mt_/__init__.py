from typing import List


from ..lon_ import Cleaner
from .conversion import PhonemeConvertor
from .syllabification import Syllabification
from .spelling import Spelling

__all__ = [
    "MTransliteration",
    "PhonemeConvertor",
    "Syllabification",
    "Spelling",
]


class MTransliteration:
    def __init__(self) -> None:
        self.pc = PhonemeConvertor()
        self.syllabification = Syllabification()
        self.spelling = Spelling()

    def transliterate_words(
        self,
        text: str,
        show_steps: bool = False,
    ) -> str:
        words = []
        for i, word in enumerate(text.split()):
            # print(f"{i=} | {word=}")
            words.append(self.transliterate(word=word, show_steps=show_steps))
        return "\n".join(words)

    def transliterate(
        self,
        word: str,
        show_steps: bool = False,
        sep: str = "/",
    ) -> str:
        # 0. Clean input text
        word = Cleaner.deepclean_bn_utf(word)
        res = ""

        # 0.1 Skip other process if string is empty
        if not word.strip():
            return ""

        # 1.1. Phoneme Conversion
        # Prepare phoneme sequence and character sequence
        phoneme_seq, char_seq = self.pc.extract_seq(word)

        # 1.2. Syllabification
        # Get split points
        split_tags = self.syllabification.get_split_tags(char_seq, phoneme_seq)

        syl_chars = self.pc.split_seq_by_bool(char_seq, split_tags)

        if show_steps:
            syl_phonemes = self.pc.split_seq_by_bool(phoneme_seq, split_tags, sep=".")
            res += f"{sep.join(syl_chars)}\t{sep.join(syl_phonemes)}\t"

        # 2.1. Phoneme Conversion
        # Prepare phoneme list and characters list (includes diphthongs)
        sup_phonemes: List[List[str]] = self.pc.split_more(
            [self.pc.prepare_syllable_phoneme(syllable) for syllable in syl_chars]
        )

        if show_steps:
            syl_phonemes = [".".join(phonemes) for phonemes in sup_phonemes]
            res += f"{sep.join(syl_phonemes)}\t"

        # 2.2. Spelling
        # Get words in mm from the syllables
        chars_mm = self.spelling.spell(sup_phonemes)

        res += "".join(chars_mm)

        return res
