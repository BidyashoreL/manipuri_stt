from typing import Set


class MM:
    """
    Class representing the Meetei Mayek character inventory.

    This class defines constants for various Meetei Mayek characters, including vowels,
    consonants, diphthongs, digits, and linguistic sets for syllabic analysis.

    Attributes:
    - kok: Meetei Mayek kok character (ꯀ)
    - sam: Meetei Mayek sam character (ꯁ)
    - lai: Meetei Mayek lai character (ꯂ)
    - ... (other Meetei Mayek characters)

    Methods:
    This class does not have methods but defines sets categorizing Meetei Mayek characters.
    """

    kok: str = "\uabc0"  # \uabc0 -> ꯀ
    sam: str = "\uabc1"  # \uabc1 -> ꯁ
    lai: str = "\uabc2"  # \uabc2 -> ꯂ
    mit: str = "\uabc3"  # \uabc3 -> ꯃ
    pa: str = "\uabc4"  # \uabc4 -> ꯄ
    na: str = "\uabc5"  # \uabc5 -> ꯅ
    chil: str = "\uabc6"  # \uabc6 -> ꯆ
    til: str = "\uabc7"  # \uabc7 -> ꯇ
    khou: str = "\uabc8"  # \uabc8 -> ꯈ
    ngou: str = "\uabc9"  # \uabc9 -> ꯉ
    thou: str = "\uabca"  # \uabca -> ꯊ
    wai: str = "\uabcb"  # \uabcb -> ꯋ
    yang: str = "\uabcc"  # \uabcc -> ꯌ
    huk: str = "\uabcd"  # \uabcd -> ꯍ
    un: str = "\uabce"  # \uabce -> ꯎ
    i: str = "\uabcf"  # \uabcf -> ꯏ
    pham: str = "\uabd0"  # \uabd0 -> ꯐ
    atiya: str = "\uabd1"  # \uabd1 -> ꯑ
    gok: str = "\uabd2"  # \uabd2 -> ꯒ
    jham: str = "\uabd3"  # \uabd3 -> ꯓ
    rai: str = "\uabd4"  # \uabd4 -> ꯔ
    ba: str = "\uabd5"  # \uabd5 -> ꯕ
    jil: str = "\uabd6"  # \uabd6 -> ꯖ
    dil: str = "\uabd7"  # \uabd7 -> ꯗ
    ghou: str = "\uabd8"  # \uabd8 -> ꯘ
    dhou: str = "\uabd9"  # \uabd9 -> ꯙ
    bham: str = "\uabda"  # \uabda -> ꯚ
    kok_lonsum: str = "\uabdb"  # \uabdb -> ꯛ
    lai_lonsum: str = "\uabdc"  # \uabdc -> ꯜ
    mit_lonsum: str = "\uabdd"  # \uabdd -> ꯝ
    pa_lonsum: str = "\uabde"  # \uabde -> ꯞ
    na_lonsum: str = "\uabdf"  # \uabdf -> ꯟ
    til_lonsum: str = "\uabe0"  # \uabe0 -> ꯠ
    ngou_lonsum: str = "\uabe1"  # \uabe1 -> ꯡ
    i_lonsum: str = "\uabe2"  # \uabe2 -> ꯢ
    onap: str = "\uabe3"  # \uabe3 -> ꯣ
    inap: str = "\uabe4"  # \uabe4 -> ꯤ
    anap: str = "\uabe5"  # \uabe5 -> ꯥ
    yenap: str = "\uabe6"  # \uabe6 -> ꯦ
    sounap: str = "\uabe7"  # \uabe7 -> ꯧ
    unap: str = "\uabe8"  # \uabe8 -> ꯨ
    cheinap: str = "\uabe9"  # \uabe9 -> ꯩ
    nung: str = "\uabea"  # \uabea -> ꯪ
    cheikhei: str = "\uabeb"  # \uabeb -> ꯫
    lum_iyek: str = "\uabec"  # \uabec -> ꯬
    apun_iyek: str = "\uabed"  # \uabed -> ꯭
    zero: str = "\uabf0"  # \uabf0 -> ꯰
    one: str = "\uabf1"  # \uabf1 -> ꯱
    two: str = "\uabf2"  # \uabf2 -> ꯲
    three: str = "\uabf3"  # \uabf3 -> ꯳
    four: str = "\uabf4"  # \uabf4 -> ꯴
    five: str = "\uabf5"  # \uabf5 -> ꯵
    six: str = "\uabf6"  # \uabf6 -> ꯶
    seven: str = "\uabf7"  # \uabf7 -> ꯷
    eight: str = "\uabf8"  # \uabf8 -> ꯸
    nine: str = "\uabf9"  # \uabf9 -> ꯹

    # Define diphthongs
    # Initial/Independent diphthongs
    in_ai: str = f"{atiya}{anap}{i_lonsum}"  # ꯑꯥꯢ (AI)
    in_xi: str = f"{atiya}{cheinap}"  # ꯑꯩ (XI)
    in_oi: str = f"{atiya}{onap}{i_lonsum}"  # ꯑꯣꯢ (OI)
    in_ui: str = f"{un}{i_lonsum}"  # ꯎꯢ (UI)
    in_au: str = f"{atiya}{anap}{un}"  # ꯑꯥꯎ (AU)
    in_xu: str = f"{atiya}{sounap}"  # ꯑꯧ (XU)
    # Final/Dependent diphthongs
    fi_ai: str = f"{anap}{i_lonsum}"  # ꯥꯢ (AI)
    fi_xi: str = cheinap  # ꯩ (XI)
    fi_oi: str = f"{onap}{i_lonsum}"  # ꯣꯢ (OI)
    fi_ui: str = f"{unap}{i_lonsum}"  # ꯨꯢ (UI)
    fi_au: str = f"{anap}{un}"  # ꯥꯎ (AU)
    fi_xu: str = sounap  # ꯧ (XU)

    # Define Sets
    # Vowels
    mapum_set_V: Set[str] = {un, i, atiya}
    cheitap_set_V: Set[str] = {chr(char) for char in range(ord(onap), ord(cheinap) + 1)}
    lonsum_set_V: Set[str] = {i_lonsum}
    # Consonants
    mapum_set_C: Set[str] = {chr(char) for char in range(ord(kok), ord(huk) + 1)}.union(
        {pham},
        {chr(char) for char in range(ord(gok), ord(bham) + 1)},
    )
    cheitap_set_C: Set[str] = {nung}
    lonsum_set_C: Set[str] = {
        chr(char) for char in range(ord(kok_lonsum), ord(ngou_lonsum) + 1)
    }
    # Separation of Alphabet
    mapum_set: Set[str] = mapum_set_C.union(mapum_set_V)
    cheitap_set: Set[str] = cheitap_set_C.union(cheitap_set_V)
    lonsum_set: Set[str] = lonsum_set_C.union(lonsum_set_V)

    # Diphthongs Set
    in_diphthong_set: Set[str] = {
        in_ai,
        in_xi,
        in_oi,
        in_ui,
        in_au,
        in_xu,
    }
    fi_diphthong_set: Set[str] = {
        fi_ai,
        fi_xi,
        fi_oi,
        fi_ui,
        fi_au,
        fi_xu,
    }

    # Digits set
    digit_set: Set[str] = {chr(char) for char in range(ord(zero), ord(nine) + 1)}

    # Linguistic Sets (Syllabic)
    in_nucleus_set: Set[str] = mapum_set_V.union(in_diphthong_set)
    fi_nucleus_set: Set[str] = cheitap_set_V.union(fi_diphthong_set)


class MMHelper:
    @staticmethod
    def has_char(char: str) -> bool:
        """Check if the character exists in the Meetei Mayek alphabet."""
        return MM.kok <= char <= MM.nine

    @staticmethod
    def has_digit(char: str) -> bool:
        """Check if the character is a digit."""
        return MM.zero <= char <= MM.nine
