from typing import Dict, List
from ..lon_ import Phoneme, ARPABETPhoneme, PhonemeInventory, BN, MM


class ARPA2MM:
    def get_map():
        arpa = ARPABETPhoneme(num_letters=2)
        mm_to_arpabet: Dict[str | List[str]] = {
            Phoneme.k.value: [arpa.phoneme_K],
            Phoneme.kh.value: [],
            Phoneme.g.value: [arpa.phoneme_G],
            Phoneme.gh.value: [],
            Phoneme.ng.value: [arpa.phoneme_NG],
            Phoneme.c.value: [arpa.phoneme_CH],
            Phoneme.z.value: [arpa.phoneme_JH, arpa.phoneme_Z, arpa.phoneme_ZH],
            Phoneme.zh.value: [],
            Phoneme.t.value: [arpa.phoneme_DX, arpa.phoneme_T],
            Phoneme.th.value: [arpa.phoneme_TH],
            Phoneme.d.value: [arpa.phoneme_D],
            Phoneme.dh.value: [arpa.phoneme_DH],
            Phoneme.n.value: [arpa.phoneme_N, arpa.phoneme_NX],
            Phoneme.p.value: [arpa.phoneme_P],
            Phoneme.ph.value: [arpa.phoneme_F],
            Phoneme.b.value: [arpa.phoneme_B],
            Phoneme.bh.value: [arpa.phoneme_V],
            Phoneme.m.value: [arpa.phoneme_M],
            Phoneme.j.value: [arpa.phoneme_Y],
            Phoneme.r.value: [arpa.phoneme_R],
            Phoneme.w.value: [arpa.phoneme_W, arpa.phoneme_WH],
            Phoneme.l.value: [arpa.phoneme_L],
            Phoneme.s.value: [arpa.phoneme_S, arpa.phoneme_SH],
            Phoneme.h.value: [arpa.phoneme_H],
            # vowels - monophthongs
            Phoneme.i.value: [arpa.phoneme_IH, arpa.phoneme_IX, arpa.phoneme_IY],
            Phoneme.e.value: [arpa.phoneme_AE, arpa.phoneme_EH],
            Phoneme.x.value: [arpa.phoneme_AH, arpa.phoneme_AX, arpa.phoneme_Q],
            Phoneme.u.value: [arpa.phoneme_UH, arpa.phoneme_UW, arpa.phoneme_UX],
            Phoneme.o.value: [arpa.phoneme_AO, arpa.phoneme_OW],
            Phoneme.a.value: [arpa.phoneme_AA],
            # extras
            Phoneme.au.value: [arpa.phoneme_AW],
            f"{Phoneme.x.value}{Phoneme.r.value}": [arpa.phoneme_AXR, arpa.phoneme_ER],
            Phoneme.ai.value: [arpa.phoneme_AY, arpa.phoneme_EY],
            Phoneme.oi.value: [arpa.phoneme_OY],
            f"{Phoneme.x.value}{Phoneme.l.value}": [arpa.phoneme_EL],
            f"{Phoneme.x.value}{Phoneme.m.value}": [arpa.phoneme_EM],
            f"{Phoneme.x.value}{Phoneme.n.value}": [arpa.phoneme_EN],
        }
        return {
            ipa: phoneme
            for phoneme, ipa_list in mm_to_arpabet.items()
            for ipa in ipa_list
            if ipa_list
        }
