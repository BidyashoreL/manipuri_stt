from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

from src.gc_ import GlyphCorrection
from run import run
from utils import read_list


def plot():
    # Run in evaluate mode
    gc = GlyphCorrection()
    run(gc.correct_words, "evaluate")

    # Get edit distances data from data/gc_/proposed/comparison.txt
    edit_distances = read_edit_distances(Path("data/gc_/proposed/comparison.txt"))
    counts = Counter(edit_distances)
    total_count = sum(counts.values())
    distances = sorted(counts.keys())

    # Exclude zero edit distance
    distances = [distance for distance in distances if distance != 0]
    occurrences = [counts[distance] for distance in distances]
    percentages = [count / total_count * 100 for count in occurrences]

    # Plot the bar chart
    fig, ax = plt.subplots()
    bars = ax.bar(distances, percentages, color="b", alpha=0.6)

    ax.set_xlabel("Edit distance")
    ax.set_ylabel("Percentage (%)")
    plt.title("Glyph Correction Error Distribution")

    # Annotate bars with percentage
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax.annotate(
            f"{percentage:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.savefig(Path("data/gc_/graph.png"))
    plt.show()


def read_edit_distances(file: Path):
    edit_distances = []
    for line in read_list(file):
        *others, ed = line.split("\t")
        if ed:
            edit_distance = int(ed)
            edit_distances.append(edit_distance)
    return edit_distances


if __name__ == "__main__":
    plot()
