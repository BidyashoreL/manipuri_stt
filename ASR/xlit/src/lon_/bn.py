from typing import Set


class BN:
    """
    Class representing the Bengali character inventory.

    This class defines constants for various Bengali characters, including vowels,
    consonants, diacritics, digits, and punctuation marks. It also provides sets
    categorizing these characters into different groups such as independent vowels,
    consonants, matra/dependent vowels, diphthongs, digits, punctuation marks, etc.

    Attributes:
    - candrabindu: Bengali candrabindu character (ঁ)
    - anusvara: Bengali anusvara character (ং)
    - visarga: Bengali visarga character (ঃ)
    - ... (other Bengali characters)

    Methods:
    This class does not have methods but defines sets categorizing Bengali characters.
    """

    candrabindu: str = "\u0981"  # \u0981 -> ঁ
    anusvara: str = "\u0982"  # \u0982 -> ং
    visarga: str = "\u0983"  # \u0983 -> ঃ
    a: str = "\u0985"  # \u0985 -> অ
    aa: str = "\u0986"  # \u0986 -> আ
    i: str = "\u0987"  # \u0987 -> ই
    ii: str = "\u0988"  # \u0988 -> ঈ
    u: str = "\u0989"  # \u0989 -> উ
    uu: str = "\u098a"  # \u098a -> ঊ
    r_vocalic: str = "\u098b"  # \u098b -> ঋ
    e: str = "\u098f"  # \u098f -> এ
    ai: str = "\u0990"  # \u0990 -> ঐ
    o: str = "\u0993"  # \u0993 -> ও
    ao: str = "\u0994"  # \u0994 -> ঔ
    ka: str = "\u0995"  # \u0995 -> ক
    kha: str = "\u0996"  # \u0996 -> খ
    ga: str = "\u0997"  # \u0997 -> গ
    gha: str = "\u0998"  # \u0998 -> ঘ
    nga: str = "\u0999"  # \u0999 -> ঙ
    ca: str = "\u099a"  # \u099a -> চ
    cha: str = "\u099b"  # \u099b -> ছ
    ja: str = "\u099c"  # \u099c -> জ
    jha: str = "\u099d"  # \u099d -> ঝ
    nya: str = "\u099e"  # \u099e -> ঞ
    tta: str = "\u099f"  # \u099f -> ট
    ttha: str = "\u09a0"  # \u09a0 -> ঠ
    dda: str = "\u09a1"  # \u09a1 -> ড
    ddha: str = "\u09a2"  # \u09a2 -> ঢ
    nna: str = "\u09a3"  # \u09a3 -> ণ
    ta: str = "\u09a4"  # \u09a4 -> ত
    tha: str = "\u09a5"  # \u09a5 -> থ
    da: str = "\u09a6"  # \u09a6 -> দ
    dha: str = "\u09a7"  # \u09a7 -> ধ
    na: str = "\u09a8"  # \u09a8 -> ন
    pa: str = "\u09aa"  # \u09aa -> প
    pha: str = "\u09ab"  # \u09ab -> ফ
    ba: str = "\u09ac"  # \u09ac -> ব
    bha: str = "\u09ad"  # \u09ad -> ভ
    ma: str = "\u09ae"  # \u09ae -> ম
    ya: str = "\u09af"  # \u09af -> য
    ra: str = "\u09b0"  # \u09b0 -> র
    la: str = "\u09b2"  # \u09b2 -> ল
    sha: str = "\u09b6"  # \u09b6 -> শ
    ssa: str = "\u09b7"  # \u09b7 -> ষ
    sa: str = "\u09b8"  # \u09b8 -> স
    h: str = "\u09b9"  # \u09b9 -> হ
    nukta: str = "\u09bc"  # \u09bc -> ়
    avagraha: str = "\u09bd"  # \u09bd -> ঽ
    v_aa: str = "\u09be"  # \u09be -> া
    v_i: str = "\u09bf"  # \u09bf -> ি
    v_ii: str = "\u09c0"  # \u09c0 -> ী
    v_u: str = "\u09c1"  # \u09c1 -> ু
    v_uu: str = "\u09c2"  # \u09c2 -> ূ
    v_r_vocalic: str = "\u09c3"  # \u09c3 -> ৃ
    v_e: str = "\u09c7"  # \u09c7 -> ে
    v_ai: str = "\u09c8"  # \u09c8 -> ৈ
    v_o: str = "\u09cb"  # \u09cb ->  ো
    v_au: str = "\u09cc"  # \u09cc ->  ৌ
    virama: str = "\u09cd"  # \u09cd -> ্
    khanda_ta: str = "\u09ce"  # \u09ce -> ৎ
    mark_au: str = "\u09d7"  # \u09d7 -> ৗ
    rra: str = "\u09dc"  # \u09dc -> ড়
    rha: str = "\u09dd"  # \u09dd -> ঢ়
    yya: str = "\u09df"  # \u09df -> য়
    zero: str = "\u09e6"  # \u09e6 -> ০
    one: str = "\u09e7"  # \u09e7 -> ১
    two: str = "\u09e8"  # \u09e8 -> ২
    three: str = "\u09e9"  # \u09e9 -> ৩
    four: str = "\u09ea"  # \u09ea -> ৪
    five: str = "\u09eb"  # \u09eb -> ৫
    six: str = "\u09ec"  # \u09ec -> ৬
    seven: str = "\u09ed"  # \u09ed -> ৭
    eight: str = "\u09ee"  # \u09ee -> ৮
    nine: str = "\u09ef"  # \u09ef -> ৯
    w: str = "\u09f1"  # \u09f1 -> ৱ

    # Define diphthongs
    # Initial/Independent diphthongs
    in_ai: Set[str] = {
        aa + i,
        aa + ii,
        aa + ya,
        aa + yya,
    }
    in_xi: Set[str] = {ai}
    in_oi: Set[str] = {
        o + i,
        o + ii,
        o + ya,
        o + yya,
    }
    in_ui: Set[str] = {
        u + i,
        uu + i,
        u + ii,
        uu + ii,
        u + ya,
        uu + ya,
        u + yya,
        uu + yya,
    }
    in_au: Set[str] = {
        aa + u,
        aa + uu,
        aa + o,
    }
    in_xu: Set[str] = {ao}
    # Final/Dependent diphthongs
    fi_ai: Set[str] = {
        v_aa + i,
        v_aa + ii,
        v_aa + ya,
        v_aa + yya,
    }
    fi_xi: Set[str] = {v_ai}
    fi_oi: Set[str] = {
        v_o + i,
        v_o + ii,
        v_o + ya,
        v_o + yya,
    }
    fi_ui: Set[str] = {
        v_u + i,
        v_uu + i,
        v_u + ii,
        v_uu + ii,
        v_u + ya,
        v_uu + ya,
        v_u + yya,
        v_uu + yya,
    }
    fi_au: Set[str] = {
        v_aa + u,
        v_aa + uu,
        v_aa + o,
    }
    fi_xu: Set[str] = {v_au}

    # Define Sets
    # Independent Vowels
    main_set_V: Set[str] = {chr(char) for char in range(ord(a), ord(ao) + 1)}
    # Independent Consonants
    main_set_C: Set[str] = {chr(char) for char in range(ord(ka), ord(h) + 1)}.union(
        {
            rra,
            rha,
            yya,
            w,
        },
    )

    # Matra/Dependent Vowels
    fi_set_V: Set[str] = {
        chr(char) for char in range(ord(v_aa), ord(v_au) + 1)
    }.difference({v_r_vocalic})

    # Final/Dependent Consonants
    fi_set_C: Set[str] = {
        v_r_vocalic,
        khanda_ta,
        anusvara,
    }
    # Vowel written left
    L_vowels: Set[str] = {
        v_i,
        v_e,
        v_ai,
    }

    # Vowel written right
    R_vowels: Set[str] = {v_aa, v_ii}

    # Vowel written at bottom
    B_vowels: Set[str] = {v_u, v_uu}

    # Vowel written as enclosed
    E_vowels: Set[str] = {v_o, v_au}

    # Independent Diphthongs
    in_diphthong_set: Set[str] = in_ai | in_xi | in_ui | in_oi | in_au | in_xu

    # Dependent Diphthongs
    fi_diphthong_set: Set[str] = fi_ai | fi_xi | fi_ui | fi_oi | fi_au | fi_xu

    # Digits set
    digit_set: Set[str] = {chr(char) for char in range(ord(zero), ord(nine) + 1)}

    # Punctuation set
    punctuation_set: Set[str] = {
        candrabindu,
        visarga,
        avagraha,
    }

    incomplete_charset: Set[str] = {
        nukta,
        mark_au,
    }


class BNHelper:
    @staticmethod
    def has_char(char: str) -> bool:
        """Check if the character exists in the Bengali alphabet."""
        return BN.candrabindu <= char <= BN.w

    @staticmethod
    def has_digit(char: str) -> bool:
        """Check if the character is a digit."""
        return BN.zero <= char <= BN.nine
