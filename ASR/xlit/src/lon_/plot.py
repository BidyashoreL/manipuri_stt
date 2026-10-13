from typing import List
from matplotlib import pyplot as plt
import numpy as np
from .phoneme import PhonemeInventory


def plot_ssp(words_in_phonemes: List[str]) -> None:
    """
    Plot Sonority Sequencing Principle (SSP) for phonemes.

    This function plots the Sonority Sequencing Principle (SSP) for the given list of phonemes.

    Args:
    - words_in_phonemes (List[str]): List of phonemes representing words.

    Returns:
    - None
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    mmP = PhonemeInventory()
    for i, phonemes in enumerate(words_in_phonemes):
        row = i // 2
        col = i % 2

        sonority_values = [
            mmP.get_sievers(
                phoneme,
            )[0]
            for phoneme in phonemes
        ]
        y = np.array(sonority_values)

        phoneme_labels = [f"{i}/{phoneme}" for i, phoneme in enumerate(phonemes)]

        axes[row, col].plot(phoneme_labels, y, marker="*", label=phonemes)

    plt.tight_layout()
    plt.show()
