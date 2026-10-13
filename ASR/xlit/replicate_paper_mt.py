import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from src.mt_ import MTransliteration
from src.mt_base_.b2m import Baseline, BaselineExtended
from run import run
from utils import read_list, write_list


# Plot the result values
def plot(corpus: Dict[str, Path]):
    metrics = ("Accuracy", "CER")
    model_names = ("Baseline", "Baseline 2", "Proposed")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    for i, metric in enumerate(metrics):
        data_points = {corpus_dir: [] for corpus_dir in corpus}
        for corpus_name, corpus_dir in corpus.items():
            w_values, c_values = [], []
            for model_name in (x.lower().replace(" ", "_") for x in model_names):
                res_file = Path(corpus_dir) / model_name / "result.txt"
                res_lines = read_list(res_file)
                w_values.append(100 - float(res_lines[0]))
                c_values.append(float(res_lines[1]))
            data_points[corpus_name].extend(
                w_values if metric == "Accuracy" else c_values
            )

        # Plot on the corresponding subplot
        ax = axs[i]
        for data_type, values in data_points.items():
            ax.plot(values, label=f"{data_type}")

        ax.set_xlabel("Models")
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(f"{metric} Comparison")
        ax.legend(title="Corpus")
        ax.grid(True)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(model_names)
        # ax.set_ylim([0, 100])

        data = pd.DataFrame(data_points, index=list(model_names))
        print(f"\n{metric}\n{data}")

    plt.savefig(Path("data/corpus/graph.png"))
    plt.tight_layout()
    plt.show()


# Run and evaluate all corpus
def evaluate(subdir: Dict[str, Path]):
    mt = MTransliteration()
    base1 = Baseline()
    base2 = BaselineExtended()
    for corpus_dir in subdir.values():
        file = corpus_dir / "target.txt"
        print(
            f"Corpus: ",
            corpus_dir.as_posix().split("/")[-1].capitalize().replace("_", " "),
        )
        run(
            mt.transliterate_words,
            "evaluate",
            model_name="Proposed",
            root_dir=corpus_dir,
        )
        run(
            base1.transliterate,
            "evaluate",
            model_name="Baseline",
            root_dir=corpus_dir,
        )
        run(
            base2.transliterate,
            "evaluate",
            model_name="Baseline 2",
            root_dir=corpus_dir,
        )


# Preparation
def prepare_paper_mt(subdir: Dict[str, Path]):
    """Prepare corpus transcription by splitting words into various categories."""
    words_news = {
        line.split("\t")[1].strip() for line in read_list("data/corrected.txt")
    }
    words_literature = read_list(subdir[3] / "words.txt")
    transcribed = read_list("data/transcribed.txt")

    count_ind, count_exo, count_ne, count_hy, count_rem = 0, 0, 0, 0, 0
    words_ind, words_exo = set(), set()
    # transcribed.txt --> code
    for i, line in enumerate(tqdm(transcribed, desc="Extracting words"), start=1):
        # print(f"{i=}. {line=}")
        word_bn, _, dist_id = line.split("\t")
        dist_id = int(dist_id)
        if dist_id == 1:
            words_ind.add(word_bn)
            count_ind += 1
        elif dist_id > 1:
            words_exo.add(word_bn)
            if dist_id == 2:
                count_exo += 1
            if dist_id == 3:
                count_ne += 1
            if dist_id == 4:
                count_hy += 1
        elif dist_id == 0:
            count_rem += 1

    words_dict: Dict[int, List[str]] = {
        0: [],  # Indigenous words
        1: [],  # Exotic words
        2: [],  # News subset
        3: [],  # Literature subset
    }
    # Start splitting
    for line in tqdm(transcribed, desc="Preparing corpus"):
        word_bn, word_mm, _ = line.split("\t")
        new_line = f"{word_bn}\t{word_mm}"
        if word_bn in words_ind:
            words_dict[0].append(new_line)
        if word_bn in words_exo:
            words_dict[1].append(new_line)
        if word_bn in words_news:
            words_dict[2].append(new_line)
        if word_bn in words_literature:
            words_dict[3].append(new_line)

    for i, corpus_dir in subdir.items():
        corpus_dir.mkdir(exist_ok=True)
        write_list(corpus_dir / "target.txt", words_dict[i])

    print(
        f"Information:\n"
        f"{len(transcribed)=}\n"
        f"{len(words_ind)=} | {len(words_exo)=}\n"
        f"{len(words_news)=} | {len(words_literature)=}\n"
    )
    print(
        f"Corpus Distribution\n"
        f"{count_ind=} | {count_exo=} | {count_ne=} | {count_hy=}\n"
        f"{(count_ind+count_exo+count_ne+count_hy)=} | {count_rem=}\n"
    )


if __name__ == "__main__":
    root_dir = Path("data/corpus")
    subsets = (
        "Indigenous words",
        "Exotic words",
        "News Corpus",
        "Literature Corpus",
    )
    subdir = {i: root_dir / x.lower().replace(" ", "_") for i, x in enumerate(subsets)}
    corpus = {x: root_dir / x.lower().replace(" ", "_") for i, x in enumerate(subsets)}

    # prepare_paper_mt(subdir)
    # evaluate(subdir)
    plot(corpus)
