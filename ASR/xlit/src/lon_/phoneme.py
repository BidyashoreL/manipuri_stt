from enum import Enum
from typing import Dict, List, Set, Tuple


# Enums
class Phoneme(Enum):
    """Define the 36 phonemes of Meetei Mayek.

    Args:
        Enum (str): phoneme values
    """

    k: str = "K"
    kh: str = "KH"
    g: str = "G"
    gh: str = "GH"
    ng: str = "NG"
    c: str = "C"
    z: str = "Z"
    zh: str = "ZH"
    t: str = "T"
    th: str = "TH"
    d: str = "D"
    dh: str = "DH"
    n: str = "N"
    p: str = "P"
    ph: str = "PH"
    b: str = "B"
    bh: str = "BH"
    m: str = "M"
    j: str = "J"
    r: str = "R"
    w: str = "W"
    l: str = "L"
    s: str = "S"
    h: str = "H"
    i: str = "I"
    e: str = "E"
    a: str = "A"
    x: str = "X"
    u: str = "U"
    o: str = "O"
    ai: str = "AI"
    xi: str = "XI"
    ui: str = "UI"
    oi: str = "OI"
    au: str = "AU"
    xu: str = "XU"


class PoA(Enum):
    """Place of Articulation Enum"""

    BILABIAL = "bilabial"
    ALVEOLAR = "alveolar"
    PALATAL = "palatal"
    VELAR = "velar"
    GLOTTAL = "glottal"
    UNDEFINED = ""


class MoA(Enum):
    """Manner of Articulation"""

    NASAL = "nasal"
    PLOSIVE = "plosive"
    FRICATIVE = "fricative"
    APPROXIMANT = "approximant"
    TAP_FLAP = "tap/flap"
    TRILL = "trill"
    LATERAL_FRICATIVE = "lateral fricative"
    LATERAL_APPROXIMANT = "lateral approximant"
    LATEAL_TAP_FLAP = "lateral tap/flap"
    UNDEFINED = ""


class Sievers(Enum):
    """Sonority classes for SSP (Sievers 1876) based on relative loudness.

    Args:
        Enum (int): ssp value
    """

    PHONE_S: int = 0
    OBSTRUENT: int = 1
    NASAL: int = 2
    LIQUID: int = 3
    GLIDE: int = 4


class Parker(Enum):
    """Sonority classes for SSP (Parker 2002) based on acoustic intensity.

    Args:
        Enum (int): ssp value
    """

    VCL_STOP: int = 0
    VCL_FRICATIVE: int = 1
    VCD_STOP: int = 2
    VCD_FRICATIVE: int = 3
    PHONE_H: int = 4
    NASAL: int = 5
    TRILL: int = 6
    FLAP: int = 7
    LATERAL: int = 8
    RHOTIC: int = 9
    GLIDE: int = 10


class VowelTongue(Enum):
    """Tongue position of Articulation

    Args:
        enum (int): value
    """

    FRONT: int = 0
    CENTRAL: int = 1
    BACK: int = 2


class VowelMouth(Enum):
    """Articulation from mouth

    Args:
        enum (int): value
    """

    CLOSE: int = 0
    MID: int = 1
    OPEN: int = 2


class VowelLips(Enum):
    """Whether lips is rounded

    Args:
        enum (bool): value
    """

    ROUNDED: bool = True
    NOT_ROUNDED: bool = False


# 1. Meetei Mayek Phoneme Inventory
class PhonemeInventory:
    """
    Class representing the phoneme inventory of Meetei Mayek.

    Attributes:
        to_ipa_map (Dict[str, str]): Mapping of phonemes to their IPA representations.
        to_phoneme_map (Dict[str, str]): Mapping of IPA symbols to phonemes.
        feats_consonant (Dict[str, Tuple[Tuple[int, int], Tuple[int, int, bool, bool]]]): Features for consonant phonemes.
        feats_vowels (Dict[str, Tuple[int, int, bool]]): Features for vowel phonemes.
    """

    def __init__(self) -> None:
        """Initialize the Meetei Mayek phoneme inventory."""
        consonant_ipa: Dict[Phoneme, List[str]] = {
            Phoneme.k: ["k"],
            Phoneme.kh: ["kʰ"],
            Phoneme.g: ["g"],
            Phoneme.gh: ["gʰ"],
            Phoneme.ng: ["ŋ"],
            Phoneme.c: ["c", "tʃ"],
            Phoneme.z: ["ɟ", "ʒ", "dʒ"],
            Phoneme.zh: ["ɟʰ", "dʒʰ"],
            Phoneme.t: ["t"],
            Phoneme.th: ["tʰ", "θ"],
            Phoneme.d: ["d"],
            Phoneme.dh: ["dʰ", "ð"],
            Phoneme.n: ["n"],
            Phoneme.p: ["p"],
            Phoneme.ph: ["pʰ", "f"],
            Phoneme.b: ["b"],
            Phoneme.bh: ["bʰ", "v"],
            Phoneme.m: ["m"],
            Phoneme.j: ["j"],
            Phoneme.r: ["r"],
            Phoneme.w: ["w"],
            Phoneme.l: ["l"],
            Phoneme.s: ["s", "ʃ"],
            Phoneme.h: ["h"],
        }
        monophthongs_ipa: Dict[str, List[str]] = {
            Phoneme.i: ["i"],
            Phoneme.e: ["e", "æ", "ɛ"],
            Phoneme.x: ["ɘ", "ʌ"],
            Phoneme.u: ["u", "ʊ"],
            Phoneme.o: ["o", "ɔ"],
            Phoneme.a: ["a"],
        }
        diphthongs_ipa: Dict[str, List[str]] = {
            Phoneme.ai: ["ai"],
            Phoneme.xi: ["ɘi"],
            Phoneme.ui: ["ui"],
            Phoneme.oi: ["oi"],
            Phoneme.au: ["au"],
            Phoneme.xu: ["ɘu"],
        }

        corresponding_ipa_list = {
            **consonant_ipa,
            **monophthongs_ipa,
            **diphthongs_ipa,
        }

        # Phoneme Sets
        self.phoneme_set_C: Set[str] = {
            phoneme.value for phoneme in consonant_ipa.keys()
        }
        self.phoneme_set_M: Set[str] = {
            phoneme.value for phoneme in monophthongs_ipa.keys()
        }
        self.phoneme_set_D: Set[str] = {
            phoneme.value for phoneme in diphthongs_ipa.keys()
        }
        self.phoneme_set_V: Set[str] = self.phoneme_set_M.union(self.phoneme_set_D)
        self.phoneme_set_all: Set[str] = self.phoneme_set_C.union(self.phoneme_set_V)

        # Phoneme -> IPA
        self.to_ipa_map: Dict[str, str] = {
            phoneme.value: ipa_list[0]
            for phoneme, ipa_list in corresponding_ipa_list.items()
        }
        # IPA -> Phoneme
        self.to_phoneme_map: Dict[str, str] = {
            ipa: phoneme.value
            for phoneme, ipa_list in corresponding_ipa_list.items()
            for ipa in ipa_list
        }

        # consonant_features
        # consonant_phoneme:(ssp_tuple, articulation_tuple)
        # ssp_tuple -> (siever_ssp, parker_ssp)
        # articulation_tuple -> (manner, place, voiced)
        feats_consonants: Dict[
            Phoneme, Tuple[Tuple[Sievers, Parker], Tuple[str, str, bool]]
        ] = {
            Phoneme.k: (
                (Sievers.OBSTRUENT, Parker.VCL_STOP),
                (PoA.VELAR, MoA.PLOSIVE, False),
            ),
            Phoneme.kh: (
                (Sievers.OBSTRUENT, Parker.VCL_STOP),
                (PoA.VELAR, MoA.PLOSIVE, False),
            ),
            Phoneme.g: (
                (Sievers.OBSTRUENT, Parker.VCD_STOP),
                (PoA.VELAR, MoA.PLOSIVE, True),
            ),
            Phoneme.gh: (
                (Sievers.OBSTRUENT, Parker.VCD_STOP),
                (PoA.VELAR, MoA.PLOSIVE, True),
            ),
            Phoneme.ng: (
                (Sievers.NASAL, Parker.NASAL),
                (PoA.VELAR, MoA.NASAL, True),
            ),
            Phoneme.c: (
                (Sievers.OBSTRUENT, Parker.VCL_STOP),
                (PoA.PALATAL, MoA.PLOSIVE, False),
            ),
            Phoneme.z: (
                (Sievers.OBSTRUENT, Parker.VCD_STOP),
                (PoA.PALATAL, MoA.PLOSIVE, True),
            ),
            Phoneme.zh: (
                (Sievers.OBSTRUENT, Parker.VCD_STOP),
                (PoA.PALATAL, MoA.PLOSIVE, True),
            ),
            Phoneme.t: (
                (Sievers.OBSTRUENT, Parker.VCL_STOP),
                (PoA.ALVEOLAR, MoA.PLOSIVE, False),
            ),
            Phoneme.th: (
                (Sievers.OBSTRUENT, Parker.VCL_STOP),
                (PoA.ALVEOLAR, MoA.PLOSIVE, False),
            ),
            Phoneme.d: (
                (Sievers.OBSTRUENT, Parker.VCD_STOP),
                (PoA.ALVEOLAR, MoA.PLOSIVE, True),
            ),
            Phoneme.dh: (
                (Sievers.OBSTRUENT, Parker.VCD_STOP),
                (PoA.ALVEOLAR, MoA.PLOSIVE, True),
            ),
            Phoneme.n: (
                (Sievers.NASAL, Parker.NASAL),
                (PoA.ALVEOLAR, MoA.NASAL, True),
            ),
            Phoneme.p: (
                (Sievers.OBSTRUENT, Parker.VCL_STOP),
                (PoA.BILABIAL, MoA.PLOSIVE, False),
            ),
            Phoneme.ph: (
                (Sievers.OBSTRUENT, Parker.VCL_STOP),
                (PoA.BILABIAL, MoA.PLOSIVE, False),
            ),
            Phoneme.b: (
                (Sievers.OBSTRUENT, Parker.VCD_STOP),
                (PoA.BILABIAL, MoA.PLOSIVE, True),
            ),
            Phoneme.bh: (
                (Sievers.OBSTRUENT, Parker.VCD_STOP),
                (PoA.BILABIAL, MoA.PLOSIVE, True),
            ),
            Phoneme.m: (
                (Sievers.NASAL, Parker.NASAL),
                (PoA.BILABIAL, MoA.NASAL, True),
            ),
            Phoneme.j: (
                (Sievers.GLIDE, Parker.GLIDE),
                (PoA.PALATAL, MoA.APPROXIMANT, True),
            ),
            Phoneme.r: (
                (Sievers.LIQUID, Parker.RHOTIC),
                (PoA.ALVEOLAR, MoA.APPROXIMANT, True),
            ),
            Phoneme.w: (
                (Sievers.GLIDE, Parker.GLIDE),
                (PoA.BILABIAL, MoA.APPROXIMANT, True),
            ),
            Phoneme.l: (
                (Sievers.LIQUID, Parker.LATERAL),
                (PoA.ALVEOLAR, MoA.LATERAL_APPROXIMANT, True),
            ),
            Phoneme.s: (
                (Sievers.PHONE_S, Parker.VCL_STOP),
                (PoA.ALVEOLAR, MoA.FRICATIVE, False),
            ),
            Phoneme.h: (
                (Sievers.OBSTRUENT, Parker.PHONE_H),
                (PoA.GLOTTAL, MoA.FRICATIVE, False),
            ),
        }
        # vowel_features -> (tongue, mouth, is_lips_rounded)
        feats_vowels: Dict[str, Tuple[VowelTongue, VowelMouth, VowelLips]] = {
            Phoneme.i: (VowelTongue.FRONT, VowelMouth.CLOSE, VowelLips.NOT_ROUNDED),
            Phoneme.e: (VowelTongue.FRONT, VowelMouth.MID, VowelLips.NOT_ROUNDED),
            Phoneme.x: (VowelTongue.CENTRAL, VowelMouth.MID, VowelLips.NOT_ROUNDED),
            Phoneme.u: (VowelTongue.CENTRAL, VowelMouth.CLOSE, VowelLips.ROUNDED),
            Phoneme.o: (VowelTongue.BACK, VowelMouth.MID, VowelLips.ROUNDED),
            Phoneme.a: (VowelTongue.BACK, VowelMouth.OPEN, VowelLips.NOT_ROUNDED),
        }

        self.feats_consonants = {
            phoneme.value: value for phoneme, value in feats_consonants.items()
        }
        self.feats_vowels = {
            phoneme.value: value for phoneme, value in feats_vowels.items()
        }

    def _get_ssp(
        self, chars: str, tuple_element: int, ssp_map: Dict[int, str]
    ) -> Tuple[int, str]:
        """Helper method to get SSP value and name."""
        if chars in self.to_ipa_map or chars in self.to_phoneme_map:
            phoneme = self.to_phoneme_map.get(chars, chars)
            if phoneme not in self.feats_consonants:
                ssp_value = 5 if tuple_element == 0 else 11
                return ssp_value, "vowel"
            else:
                ssp_value = self.feats_consonants[phoneme][0][tuple_element]
                return ssp_value, ssp_map.get(ssp_value, "Unknown")
        else:
            return -1, "Error"

    def get_seivers_det(self, chars: str) -> Tuple[int, str]:
        """Get Sievers SSP value and name for a given character."""
        return self._get_ssp(chars, 0, self.ssp_sievers)

    def get_parker_det(self, chars: str) -> Tuple[int, str]:
        """Get Parker SSP value and name for a given character."""
        return self._get_ssp(chars, 1, self.ssp_parker)

    def get_sievers(self, consonant_phoneme: str) -> Sievers:
        """Get Sievers ssp value."""
        return self.__get_specific(consonant_phoneme, 0, 0)

    def get_parker(self, consonant_phoneme: str) -> Parker:
        """Get Parker ssp value."""
        return self.__get_specific(consonant_phoneme, 0, 1)

    def get_PoA(self, consonant_phoneme: str) -> PoA:
        """Get Place of articulation."""
        return self.__get_specific(consonant_phoneme, 1, 0)

    def get_sievers(self, consonant_phoneme: str) -> MoA:
        """Get Manner of articulation."""

        return self.__get_specific(consonant_phoneme, 1, 1)

    def get_is_voiced(self, consonant_phoneme: str) -> bool:
        """Get voiced or voiceless."""
        return self.__get_specific(consonant_phoneme, 1, 2)

    def __get_specific(self, consonant_phoneme: str, tuple_1: int, tuple_2: int):
        return (
            lambda x: self.feats_consonants.get(
                x, ((0, 0), (PoA.UNDEFINED, MoA.UNDEFINED, False))
            )[tuple_1][tuple_2]
        )(consonant_phoneme)


# 1.2 ARPAbet Phoneme
class ARPABETPhoneme:
    """
    ARPABETPhoneme class represents the ARPABET phoneme system used for representing
    the pronunciation of words in American English. This class provides functionality
    to handle ARPABET phonemes with one or two-letter representations.

    Attributes:
        num_letters (int): The number of letters to use for representing ARPABET phonemes.
            It can be either 1 or 2. Default is 2.

    Methods:
        __init__(self, num_letters: int = 2) -> None:
            Initializes the ARPABETPhoneme instance with the specified number of letters
            for representing ARPABET phonemes.

        reload_version(self, num_letters):
            Reloads the ARPABET version with the specified number of letters.

        define_alphabet_1(self):
            Defines ARPABET phonemes using a single letter representation.

        define_alphabet_2(self):
            Defines ARPABET phonemes using a two-letter representation.
    """

    def __init__(self, num_letters: int = 2) -> None:
        """
        Initializes the ARPABETPhoneme instance with the specified number of letters
        for representing ARPABET phonemes.

        Args:
            num_letters (int): The number of letters to use for representing ARPABET phonemes.
                It can be either 1 or 2. Default is 2.
        """
        self.define_alphabet_1()
        self.reload_version(num_letters)

    def reload_version(self, num_letters: int) -> None:
        """
        Reloads the ARPABET version with the specified number of letters.

        Args:
            num_letters (int): The number of letters to use for representing ARPABET phonemes.
                It can be either 1 or 2.

        Raises:
            ValueError: If `num_letters` is neither 1 nor 2.
        """
        if num_letters == 1:
            self.define_alphabet_1()
        elif num_letters == 2:
            self.define_alphabet_2()
        else:
            raise ValueError("ARPABET has only 1 or 2 letter versions.")
        self.all_phonemes = {
            getattr(self, attribute)
            for attribute in dir(self)
            if attribute.startswith("phoneme_")
        }

    def define_alphabet_1(self) -> None:
        """
        Defines ARPABET phonemes using a single letter representation.
        """
        # vowels (19 nos)
        self.phoneme_AA = "a"
        self.phoneme_AE = "@"
        self.phoneme_AH = "A"
        self.phoneme_AO = "c"
        self.phoneme_AW = "W"
        self.phoneme_AX = "x"
        self.phoneme_AXR = ""
        self.phoneme_AY = "Y"
        self.phoneme_EH = "E"
        self.phoneme_ER = "R"
        self.phoneme_EY = "e"
        self.phoneme_IH = "I"
        self.phoneme_IX = "X"
        self.phoneme_IY = "i"
        self.phoneme_OW = "o"
        self.phoneme_OY = "O"
        self.phoneme_UH = "U"
        self.phoneme_UW = "u"
        self.phoneme_UX = ""
        # consonants (31 nos)
        self.phoneme_B = "b"
        self.phoneme_CH = "C"
        self.phoneme_D = "d"
        self.phoneme_DH = "D"
        self.phoneme_DX = "F"
        self.phoneme_EL = "L"
        self.phoneme_EM = "M"
        self.phoneme_EN = "N"
        self.phoneme_F = "f"
        self.phoneme_G = "g"
        self.phoneme_H = "h"
        self.phoneme_JH = "J"
        self.phoneme_K = "k"
        self.phoneme_L = "l"
        self.phoneme_M = "m"
        self.phoneme_N = "n"
        self.phoneme_NG = "G"
        self.phoneme_NX = ""
        self.phoneme_P = "p"
        self.phoneme_Q = "Q"
        self.phoneme_R = "r"
        self.phoneme_S = "s"
        self.phoneme_SH = "S"
        self.phoneme_T = "t"
        self.phoneme_TH = "T"
        self.phoneme_V = "v"
        self.phoneme_W = "w"
        self.phoneme_WH = "WH"
        self.phoneme_Y = "y"
        self.phoneme_Z = "z"
        self.phoneme_ZH = "Z"

    def define_alphabet_2(self) -> None:
        """Defines ARPABET phonemes using a two-letter representation."""
        attrs = dir(self)
        for attr in attrs:
            if attr.startswith("phoneme_"):
                content = attr.split("_", 1)[1]
                setattr(self, attr, content)
