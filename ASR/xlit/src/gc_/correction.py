from typing import Dict, List, Set, Tuple

from .u2b import U2B
from ..lon_ import BN, Cleaner


class Correction:

    def correct(
        self,
        text: str,
        include_steps: bool = False,
    ) -> str:
        steps = []
        # Step 0: Adjusting s550 characters
        text = self.__adjust_glyph(text, charmap=U2B.premap)
        steps.append(text)

        # Step 1: Mapping Bengali Alphabet
        text = self.__map_unicode(text, charmap=U2B.charmap)
        steps.append(text)

        # Step 2: Fix suffix position of r and then mapping
        text = self.__fix_r_glyph(
            text, chars=U2B.s550_extra_chars, charmap=U2B.R_char_r
        )
        steps.append(text)

        # Step 3: Fix prefix position of vowels and fix vowels
        text = self.__fix_vowels(
            text,
            chars=BN.L_vowels,
        )
        text = Cleaner.clean_bn_utf(text)

        steps.append(text)

        # Returns final content
        return text if not include_steps else "\t".join(steps)

    # Private methods
    def __adjust_glyph(self, text: str, charmap: Dict[str, str]) -> str:
        for key, value in charmap.items():
            text = text.replace(key, value)
        return text

    def __map_unicode(self, text: str, charmap: Dict[str, str]) -> str:
        # 1. Mapping to correct unicode values (in order of decreasing size of key length)
        sorted_keys = sorted(charmap.keys(), key=len, reverse=True)
        for key in sorted_keys:
            text = text.replace(key, charmap[key])
        # 2. Fix redundant virama
        num_mistypes: int = 2
        for num in range(num_mistypes + 1, 1, -1):
            text = text.replace(BN.virama * num, BN.virama)
        return text

    def __fix_r_glyph(self, text: str, chars: Set[str], charmap: Dict[str, str]) -> str:
        # 0. Remove extra chars
        for char in chars:
            text = text.replace(char, "")
        # 1. Fixing position of r glyph
        char_list = []
        for i, char in enumerate(text):
            if char in charmap:
                start = (
                    i - 7
                    if i > 6
                    else (i - 5 if i > 4 else (i - 3 if i > 2 else i - 1))
                )
                substring, offset = self.__jump(text[start : i + 1][::-1])
                char_list = char_list[: i - offset] + substring[::-1]
            else:
                char_list.append(char)
        # 2. Post mapping r
        text = "".join(char_list)
        for char, replacement in charmap.items():
            text = text.replace(char, replacement)

        # 3. Fix redundant virama
        for num in range(3, 1, -1):
            text = text.replace(BN.virama * num, BN.virama)
        return text

    def __fix_vowels(self, text: str, chars: Set[str]):
        # 1. Fixing Left Vowels' position
        char_list = []
        skip_index = -1
        for i, char in enumerate(text):
            if i == skip_index:
                skip_index = -1
            elif char in chars:
                stop = (
                    i + 8
                    if i <= len(text) - 7
                    else (i + 6 if i <= len(text) - 5 else i + 4)
                )
                substring, offset = self.__jump(text[i:stop])
                char_list += substring
                skip_index = i + offset
            elif i > skip_index:
                char_list.append(char)

        text = "".join(char_list)
        return text

    def __jump(self, chars: str) -> Tuple[List[str], int]:
        char, *right = chars
        i = 0
        while i < len(right) - 1 and right[i + 1] == BN.virama:
            i += 2
        if i >= len(right):
            return list(chars), 1
        return (right[: i + 1] + [char], i + 1)
