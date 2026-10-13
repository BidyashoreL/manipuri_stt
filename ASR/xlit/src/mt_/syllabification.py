from typing import List, Tuple

from ..lon_ import BN, Phoneme, PhonemeInventory, PoA, MoA, Sievers


class Syllabification:
    def __init__(self) -> None:
        self.pi = PhonemeInventory()

    def get_split_tags(
        self,
        char_seq: List[str],
        phoneme_seq: List[str],
    ) -> List[bool]:
        # Initialise split point to False
        # split_points[idx] <- after char
        # split_points[idx - 1] <- before char and so on
        last_idx = len(char_seq) - 1
        split_tags = [False] * len(char_seq)
        split_tags.append(True)

        glide_and_rhotic = {Phoneme.r.value, Phoneme.j.value, Phoneme.w.value}

        # 1. char based
        for i, char in enumerate(char_seq):
            # Independent vowel, diphthongs and /H/
            if char in BN.in_diphthong_set | BN.main_set_V | {BN.h}:
                if i > 0:
                    split_tags[i - 1] = True
            # dependent vowel and diphthongs
            elif char in BN.fi_set_V | BN.fi_diphthong_set:
                if i != last_idx and char in BN.fi_xu:
                    split_tags[i] = True
                if i > 1:
                    if (
                        char_seq[i - 1] in BN.main_set_C
                        and char_seq[i - 2] != BN.virama
                    ):
                        split_tags[i - 2] = True
            # dependent consonants & xu
            elif char in BN.fi_set_C:
                if i != last_idx:
                    split_tags[i] = True
                if i > 1:
                    if (
                        char_seq[i - 1] in BN.main_set_C
                        and char_seq[i - 2] != BN.virama
                    ):
                        split_tags[i - 2] = True
            # Independent/main consonants
            else:
                pass

        # 2. phoneme + char based
        # Find invalid clusters and split them
        # Find valid cluster and mark nearest possible split point
        for i, phoneme in enumerate(phoneme_seq):
            if phoneme == BN.virama:

                if i > 0 and i < last_idx:
                    # 1. Invalid clusters
                    # Same phoneme
                    if phoneme_seq[i - 1] == phoneme_seq[i + 1]:
                        split_tags[i] = True
                    # VCCV
                    elif (
                        1 < i < last_idx - 1
                        and char_seq[i - 2]
                        in BN.main_set_V.union(
                            BN.in_diphthong_set, BN.fi_set_V, BN.fi_diphthong_set
                        ).difference(BN.fi_xu)
                        and char_seq[i + 2]
                        in BN.fi_set_V.union(BN.fi_diphthong_set).difference(BN.fi_xu)
                        and phoneme_seq[i + 1] not in glide_and_rhotic
                    ):
                        split_tags[i] = True
                    # IV + Nasal + Plosive
                    elif (
                        1 < i < last_idx - 1
                        and self.pi.get_sievers(phoneme_seq[i - 1]) == MoA.NASAL
                        and self.pi.get_sievers(phoneme_seq[i + 1]) == MoA.PLOSIVE
                        and char_seq[i - 2] in BN.main_set_V | BN.in_diphthong_set
                    ):
                        split_tags[i] = True
                    # plosive + plosive & plosive + nasal
                    elif self.pi.get_sievers(phoneme_seq[i - 1]) == MoA.PLOSIVE and (
                        self.pi.get_sievers(phoneme_seq[i + 1])
                        in {MoA.PLOSIVE, MoA.NASAL}
                    ):
                        split_tags[i] = True
                    # dip in ssp and next phoneme being vowel is raised
                    # nasal + plosive + V
                    elif (
                        i < last_idx - 1
                        and self.pi.get_sievers(phoneme_seq[i - 1]) == MoA.NASAL
                        and self.pi.get_sievers(phoneme_seq[i + 1]) == MoA.PLOSIVE
                        and phoneme_seq[i + 2]
                        in self.pi.phoneme_set_V | self.pi.phoneme_set_D
                    ):
                        split_tags[i] = True

                    # glide + liquid
                    elif (
                        i < last_idx - 1
                        and self.pi.get_sievers(phoneme_seq[i - 1]) == Sievers.GLIDE
                        and self.pi.get_sievers(phoneme_seq[i + 1]) == Sievers.LIQUID
                    ):
                        split_tags[i] = True

                    # 2. Valid clusters
                    # split before when syllable initial consonant cluster is detected
                    if phoneme_seq[i + 1] in glide_and_rhotic:
                        if i > 3 and phoneme_seq[i - 3] == Phoneme.s.value:
                            split_tags[i - 4] = True
                        elif i > 1:
                            split_tags[i - 2] = True

                    # Split/cluster condition for cluster with L
                    if (
                        i > 2
                        and phoneme_seq[i + 1] == Phoneme.l.value
                        and char_seq[i - 2] in BN.fi_diphthong_set | BN.fi_set_V
                    ):
                        split_tags[i] = True
                    # split after when syllable final consonant cluster is detected

        # Return split_points
        return split_tags
