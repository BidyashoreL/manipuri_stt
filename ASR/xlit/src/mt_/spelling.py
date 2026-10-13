from typing import Dict, List, Tuple

from ..lon_ import Phoneme, PhonemeInventory, BN, MM, Cleaner


class Spelling:

    def __init__(self) -> None:
        self.pi = PhonemeInventory()

    def spell(
        self,
        sup_phonemes: List[List[str]],
    ) -> List[str]:
        chars_mm: List[str] = []
        for phoneme_seq in sup_phonemes:
            mm_syllable = (
                Cleaner.deepclean_mm_utf(self.spell_syllable(phoneme_seq))
                if phoneme_seq
                else ""
            )
            chars_mm.append(mm_syllable)
        return chars_mm

    def spell_syllable(self, phoneme_seq: List[str]) -> str:
        """
        Apun always inserted for cluster.
        Format Syllable-final cluster: lonsum+apun+mapum

        Args:
            phoneme_list (List[str]): Phoneme sequence in Syllable

        Returns:
            str: spelt syllable in MM
        """
        S = P2M.mm_begin[phoneme_seq[0]]
        # To check whether nucleus is met
        flag = True if phoneme_seq[0] in self.pi.phoneme_set_V else False
        for i, phoneme in enumerate(phoneme_seq[1:]):
            if phoneme == BN.virama:
                if i != len(phoneme_seq) - 2:  # Exclude virama at last position
                    S += MM.apun_iyek
            elif flag:  # Next phoneme after Nucleus
                S += P2M.mm_end[phoneme]
                flag = False
            elif phoneme in self.pi.phoneme_set_C:  # all C except after nucleus
                S += P2M.mm_begin[phoneme]
            else:  # V
                flag = True
                S += P2M.mm_end[phoneme]
        return S


# Resource: Phoneme -> Meetei Mayek
class P2M:
    """Phoneme to Meetei Mayek"""

    original_map: Dict[Phoneme, Tuple[str, str, str]] = {
        Phoneme.k: (MM.kok, MM.kok_lonsum, MM.kok_lonsum),
        Phoneme.kh: (MM.khou, MM.khou, MM.kok_lonsum),
        Phoneme.g: (MM.gok, MM.gok, MM.kok_lonsum),
        Phoneme.gh: (MM.ghou, MM.ghou, MM.kok_lonsum),
        Phoneme.ng: (
            MM.ngou,
            MM.ngou_lonsum,
            MM.ngou_lonsum,
        ),
        Phoneme.c: (MM.chil, MM.chil, MM.chil),
        Phoneme.z: (MM.jil, MM.jil, MM.jil),
        Phoneme.zh: (MM.jham, MM.jham, MM.jil),
        Phoneme.t: (MM.til, MM.til_lonsum, MM.til_lonsum),
        Phoneme.th: (MM.thou, MM.thou, MM.til_lonsum),
        Phoneme.d: (MM.dil, MM.dil, MM.til_lonsum),
        Phoneme.dh: (MM.dhou, MM.dhou, MM.til_lonsum),
        Phoneme.n: (MM.na, MM.na_lonsum, MM.na_lonsum),
        Phoneme.p: (MM.pa, MM.pa_lonsum, MM.pa_lonsum),
        Phoneme.ph: (MM.pham, MM.pham, MM.pa_lonsum),
        Phoneme.b: (MM.ba, MM.ba, MM.pa_lonsum),
        Phoneme.bh: (MM.bham, MM.bham, MM.pa_lonsum),
        Phoneme.m: (MM.mit, MM.mit_lonsum, MM.mit_lonsum),
        Phoneme.j: (MM.yang, MM.yang, MM.yang),
        Phoneme.r: (MM.rai, MM.rai, MM.rai),
        Phoneme.w: (MM.wai, MM.wai, MM.wai),
        Phoneme.l: (MM.lai, MM.lai_lonsum, MM.lai_lonsum),
        Phoneme.s: (MM.sam, MM.sam, MM.sam),
        Phoneme.h: (MM.huk, MM.huk, ""),
        # vowels - monophthongs
        Phoneme.i: (MM.i, MM.inap, MM.inap),
        Phoneme.e: (
            MM.atiya + MM.yenap,
            MM.yenap,
            MM.yenap,
        ),
        Phoneme.x: (MM.atiya, "", ""),
        Phoneme.u: (MM.un, MM.unap, MM.unap),
        Phoneme.o: (
            MM.atiya + MM.onap,
            MM.onap,
            MM.onap,
        ),
        Phoneme.a: (
            MM.atiya + MM.anap,
            MM.anap,
            MM.anap,
        ),
        # vowel - diphthongs
        Phoneme.ai: (
            MM.atiya + MM.anap + MM.i_lonsum,
            MM.anap + MM.i_lonsum,
            MM.anap + MM.i_lonsum,
        ),
        Phoneme.xi: (
            MM.atiya + MM.cheinap,
            MM.cheinap,
            MM.cheinap,
        ),
        Phoneme.ui: (
            MM.un + MM.i_lonsum,
            MM.unap + MM.i_lonsum,
            MM.unap + MM.i_lonsum,
        ),
        Phoneme.oi: (
            MM.atiya + MM.onap + MM.i_lonsum,
            MM.onap + MM.i_lonsum,
            MM.onap + MM.i_lonsum,
        ),
        Phoneme.au: (
            MM.atiya + MM.anap + MM.un,
            MM.anap + MM.un,
            MM.anap + MM.un,
        ),
        Phoneme.xu: (
            MM.atiya + MM.sounap,
            MM.sounap,
            MM.sounap,
        ),
    }

    mm_begin, mm_end, mm_end_2 = {}, {}, {}
    for phoneme, chars_mm in original_map.items():
        mm_begin[phoneme.value] = chars_mm[0]
        mm_end[phoneme.value] = chars_mm[1]
        mm_end_2[phoneme.value] = chars_mm[2]
