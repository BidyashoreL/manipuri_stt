from typing import Dict, List, Set

from .bn import BN, BNHelper
from .mm import MM, MMHelper


class Cleaner:

    # MM
    @staticmethod
    def deepclean_mm_utf(word_mm: str, allow_digits: bool = False):
        digit_mapping_dict = (
            {char: "" for char in MM.digit_set} if not allow_digits else {}
        )
        word_mm = Cleaner.clean_text(word_mm, mapping_dict={**digit_mapping_dict})
        word_mm = Cleaner.clean_mm_utf(word_mm)
        return word_mm

    @staticmethod
    def clean_mm_utf(word_mm: str) -> str:
        # Fix repeated cheitaps
        first_repitition: Dict[str, str] = {char * 2: char for char in MM.cheitap_set}
        word_mm = Cleaner.clean_text(word_mm, mapping_dict={**first_repitition})
        # Fix /AI/, /OI/, /UI/, /AU/
        word_mm = Cleaner.fix_diphthong(word_mm)
        # Fix /NG/
        word_mm = Cleaner.fix_nung(word_mm)
        word_mm = Cleaner.fix_ngou_lonsum(word_mm)

        return word_mm

    @staticmethod
    def fix_diphthong(word_mm: str):
        if MM.i not in word_mm and MM.atiya + MM.onap not in word_mm:
            return word_mm
        return Cleaner.clean_text(
            word_mm,
            mapping_dict={
                MM.anap + MM.i: MM.anap + MM.i_lonsum,
                MM.onap + MM.i: MM.onap + MM.i_lonsum,
                MM.unap + MM.i: MM.unap + MM.i_lonsum,
                MM.anap + MM.atiya + MM.onap: MM.anap + MM.un,
            },
        )

    @staticmethod
    def fix_nung(word_mm: str):
        if MM.nung not in word_mm:
            return word_mm
        return Cleaner.clean_text(
            word_mm,
            mapping_dict={v + MM.nung: v + MM.ngou_lonsum for v in MM.cheitap_set_V},
        )

    @staticmethod
    def fix_ngou_lonsum(word_mm: str):
        if MM.ngou_lonsum not in word_mm:
            return word_mm
        fixed_word_mm = ""
        for i, char in enumerate(word_mm):
            if char == MM.ngou_lonsum and word_mm[i - 1] not in MM.cheitap_set_V:
                fixed_word_mm += MM.nung
            else:
                fixed_word_mm += char

        fixed_word_mm = fixed_word_mm.replace(MM.nung + MM.apun_iyek, MM.nung)

        return fixed_word_mm

    @staticmethod
    def filter_mm_utf(word_mm: str) -> str:
        return "".join([char for char in word_mm if MMHelper.has_char(char)])

    # BN
    @staticmethod
    def deepclean_bn_utf(word_bn: str, allow_digits: bool = False) -> str:
        word_bn = Cleaner.clean_bn_utf(word_bn)
        # 1. Remove all digits
        digit_mapping_dict = (
            {char: "" for char in BN.digit_set} if not allow_digits else {}
        )
        # 2. Remove virama before dependent chars
        invalid_virama_pre = {
            f"{BN.virama}{char}": char
            for char in BN.fi_set_V | BN.main_set_V | BN.fi_set_C
        }
        # 3. Remove virama after dependent chars
        invalid_virama_suff = {
            f"{char}{BN.virama}": char
            for char in BN.fi_set_V | BN.main_set_V | BN.fi_set_C
        }
        # 4. Keep only one repitition for diacritic
        diacritic_repitition = {
            err * 2: err
            for err in {
                BN.virama + BN.ra,  # rophola
                BN.virama + BN.ya,  # jophola
            }
        }

        rare_error_mapping_dict = {
            **digit_mapping_dict,
            **invalid_virama_pre,
            **invalid_virama_suff,
            **diacritic_repitition,
        }

        word_bn = word_bn[1:] if word_bn[0] == BN.virama else word_bn
        word_bn = Cleaner.clean_text(word_bn, rare_error_mapping_dict)

        return Cleaner.filter_bn_utf(word_bn)

    @staticmethod
    def clean_bn_utf(word_bn: str) -> str:
        first_repitition: Dict[str, str] = {
            char * 2: char for char in {BN.nukta, BN.virama} | BN.fi_set_V
        }
        mapping_dict_vowel: Dict[str, str] = {
            # Order is important
            BN.v_i + BN.v_ii: BN.v_ii,
            BN.v_ii + BN.v_i: BN.v_ii,
            BN.v_u + BN.v_uu: BN.v_uu,
            BN.v_uu + BN.v_u: BN.v_uu,
            # Mistypes
            BN.a + BN.v_aa: BN.aa,
            BN.aa + BN.v_aa: BN.aa,
            # Always true
            BN.v_e + BN.v_aa: BN.v_o,  # ে +  া :  ো
            BN.v_e + BN.mark_au: BN.v_au,  # ে +  ৗ :  ৌ
            # Mistypes
            BN.v_aa + BN.v_e: BN.v_o,  # া +  ে :  ো
            BN.mark_au + BN.v_aa: BN.v_au,  # ৗ +  ে :  ৌ
            # Double vowel mistype
            BN.v_o + BN.v_aa: BN.v_o,
            BN.v_o + BN.mark_au: BN.v_au,
        }
        mapping_dict_consonant: Dict[str, str] = {
            # Combination with nukta
            BN.dda + BN.nukta: BN.rra,
            BN.ddha + BN.nukta: BN.rha,
            BN.ya + BN.nukta: BN.yya,
            # Remove mark characters
            BN.nukta: "",
            BN.mark_au: "",
        }
        word_bn = Cleaner.clean_text_ordered(
            word_bn,
            mapping_dicts=[
                first_repitition,
                mapping_dict_vowel,
                mapping_dict_consonant,
            ],
        )
        return Cleaner.filter_bn_utf(word_bn)

    @staticmethod
    def filter_bn_utf(word_bn: str) -> str:
        return "".join([char for char in word_bn if BNHelper.has_char(char)])

    ## Transcription/Transliteration Error
    @staticmethod
    def clean_transcribed_utf(word_mm: str) -> str:
        mapping_dict: Dict[str, str] = {
            # BN + MM error
            BN.nukta + BN.nukta: BN.nukta,
            MM.inap + BN.nukta: BN.nukta + MM.inap,
            MM.yenap + BN.nukta: BN.nukta + MM.yenap,
            MM.jil + BN.nukta: MM.yang,
            MM.dil + BN.nukta: MM.rai,
            # MM + MM error
            MM.yenap + MM.anap: MM.onap,
        }
        word_mm = Cleaner.clean_text(word_mm, mapping_dict)
        return Cleaner.filter_mm_utf(word_mm)

    # Common
    @staticmethod
    def clean_text_ordered(word: str, mapping_dicts: List[Dict[str, str]]) -> str:
        for mapping_dict in mapping_dicts:
            for key, value in mapping_dict.items():
                word = word.replace(key, value)
        return "".join(word)

    @staticmethod
    def clean_text(word: str, mapping_dict: Dict[str, str]) -> str:
        for key, value in mapping_dict.items():
            word = word.replace(key, value)
        return "".join(word)
