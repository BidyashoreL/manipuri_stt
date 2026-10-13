from typing import Dict, Set

from ..lon_ import BN, MM, Cleaner


class Baseline:
    """
    Bengali to Meetei Mayek
    Replication of Nongmeikapam et. al.'s work
    """

    def __init__(self) -> None:
        bn = BN()
        mm = MM()

        iyek_ipee: Dict[str, Set[str]] = {
            mm.kok: {bn.ka},
            mm.sam: {bn.cha, bn.ssa, bn.sa, bn.sha},
            mm.lai: {bn.la},
            mm.mit: {bn.ma},
            mm.pa: {bn.pa},
            mm.na: {bn.nya, bn.nna, bn.na},
            mm.chil: {bn.ca},
            mm.til: {bn.tta, bn.ta, bn.khanda_ta},
            mm.khou: {bn.kha},
            mm.ngou: {bn.anusvara, bn.nga},
            mm.thou: {bn.ttha, bn.tha},
            mm.wai: {bn.w},
            mm.yang: {bn.yya, bn.ya},
            mm.huk: {bn.h},
            mm.un: {bn.u, bn.uu},
            mm.i: {bn.i, bn.ii},
            mm.pham: {bn.pha},
            mm.atiya: {bn.a},
            mm.gok: {bn.ga},
            mm.jham: {bn.jha},
            mm.rai: {
                bn.r_vocalic,
                bn.v_r_vocalic,
                bn.rra,
                bn.ra,
                bn.rha,
            },
            mm.ba: {bn.ba},
            mm.jil: {bn.ja},
            mm.dil: {bn.dda, bn.da},
            mm.ghou: {bn.gha},
            mm.dhou: {bn.ddha, bn.dha},
            mm.bham: {bn.bha},
        }

        v_letters: Dict[str, Set[str]] = {
            f"{mm.atiya}{mm.anap}": {bn.aa},
            f"{mm.atiya}{mm.yenap}": {bn.e},
            f"{mm.atiya}{mm.cheinap}": {bn.ai},
            f"{mm.atiya}{mm.onap}": {bn.o},
            f"{mm.atiya}{mm.sounap}": {bn.ao},
            f"{mm.atiya}{mm.nung}": {f"{bn.aa}{bn.anusvara}"},
        }

        cheitap_iyek: Dict[str, Set[str]] = {
            mm.onap: {bn.v_o},
            mm.inap: {bn.v_i, bn.v_ii},
            mm.anap: {bn.v_aa},
            mm.yenap: {bn.v_e},
            mm.sounap: {bn.v_au},
            mm.unap: {bn.v_u, bn.v_uu},
            mm.cheinap: {bn.v_ai},
            mm.nung: {bn.anusvara},
        }

        cheising_iyek: Dict[str, Set[str]] = {
            mm.one: {bn.one},
            mm.two: {bn.two},
            mm.three: {bn.three},
            mm.four: {bn.four},
            mm.five: {bn.five},
            mm.six: {bn.six},
            mm.seven: {bn.seven},
            mm.eight: {bn.eight},
            mm.nine: {bn.nine},
            mm.zero: {bn.zero},
        }

        lonsum_iyek: Dict[str, Set[str]] = {
            mm.kok_lonsum: {f"{bn.ka}{bn.virama}"},
            mm.lai_lonsum: {f"{bn.la}{bn.virama}"},
            mm.mit_lonsum: {f"{bn.ma}{bn.virama}"},
            mm.pa_lonsum: {f"{bn.pa}{bn.virama}"},
            mm.na_lonsum: {
                f"{bn.nna}{bn.virama}",
                f"{bn.na}{bn.virama}",
            },
            mm.til_lonsum: {
                f"{bn.tta}{bn.virama}",
                f"{bn.ta}{bn.virama}",
            },
            mm.ngou_lonsum: {f"{bn.nga}{bn.virama}"},
        }

        self.original_map = {
            **iyek_ipee,
            **v_letters,
            **cheitap_iyek,
            **cheising_iyek,
            **lonsum_iyek,
        }

        self.charmap: Dict[str, str] = {
            char_bn: char_mm
            for char_mm, char_bn_list in self.original_map.items()
            for char_bn in char_bn_list
        }
        self.sorted_keys = sorted(self.charmap.keys(), key=len, reverse=True)

    def transliterate(self, word_bn: str):
        word_mm = word_bn
        for key in self.sorted_keys:
            word_mm = word_mm.replace(key, self.charmap[key])
        word_mm = Cleaner.filter_mm_utf(word_mm)
        return word_mm


class BaselineExtended:
    """
    Bengali to Meetei Mayek
    Added Gyanendro et. al.'s work
    """

    def __init__(self) -> None:
        bn = BN()
        mm = MM()
        self.baseline = Baseline()
        self.extra_charmap = {
            f"{bn.virama}{bn.ya}": f"{mm.apun_iyek}{mm.yang}",
            f"{bn.virama}{bn.yya}": f"{mm.apun_iyek}{mm.yang}",
            f"{bn.virama}{bn.ra}": f"{mm.apun_iyek}{mm.rai}",
            f"{bn.virama}{bn.rha}": f"{mm.apun_iyek}{mm.rai}",
            f"{bn.virama}{bn.rra}": f"{mm.apun_iyek}{mm.rai}",
            f"{bn.virama}{bn.la}": f"{mm.apun_iyek}{mm.lai}",
            f"{bn.virama}{bn.w}": f"{mm.apun_iyek}{mm.wai}",
        }

    def transliterate(self, word_bn: str):
        # Deep Clean
        word_mm = Cleaner.deepclean_bn_utf(word_bn)
        # Implement extra parts first
        for key in self.extra_charmap:
            word_mm = word_mm.replace(key, self.extra_charmap[key])
        # Then implement the baseline
        for key in self.baseline.sorted_keys:
            word_mm = word_mm.replace(key, self.baseline.charmap[key])

        word_mm = Cleaner.filter_mm_utf(Cleaner.deepclean_mm_utf(word_mm))
        return word_mm
