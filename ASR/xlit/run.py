from pathlib import Path
from typing import Callable

from tqdm import tqdm
import enchant

from src.lon_.cleaner import Cleaner
from src.gc_ import GlyphCorrection
from src.mt_ import MTransliteration
from src.mt_base_.b2m import Baseline, BaselineExtended
from utils import read_dict, save_wordmap, write_dict, write_list


all_modes = [
    "evaluate",
    "simple",
    "detailed",
    "wordmap",
]


def run(
    func: Callable,
    mode: str,
    model_name: str = "Proposed",
    root_dir: str | Path = "",
) -> None:
    """
    Runs the given function in the specified mode.

    Parameters:
    func (Callable): The function to be executed.
    mode (str): The mode in which to run the function. Options are "evaluate", "simple", "detailed", and "wordmap".
    model_name (str, optional): The name of the model. Defaults to "Proposed".
    root_dir (str | Path, optional): The root directory for input/output files. Defaults to an empty string.

    Returns:
    None
    """
    model_name = model_name.lower().replace(" ", "_")
    if mode not in all_modes:
        print(f"{mode} is not a valid mode.")
        return

    root_dir = (
        (Path("data/gc_") if func.__name__.startswith("correct") else Path("data/mt_"))
        if not root_dir
        else Path(root_dir)
    )
    print(
        f"Run info:\n{func.__name__=} | {mode=} | {model_name=} | root_dir={root_dir.as_posix()}"
    )

    # 1. Evaluate mode
    if mode == "evaluate":
        evaluate(func, root_dir, model_name)
        return

    data_file = root_dir / "words.txt"
    content = data_file.read_text(encoding="utf-8").strip()
    output = func(content, show_steps=True) if mode == "detailed" else func(content)

    # 2. Simple mode
    if mode == "simple":
        output_file = root_dir / "output.txt"
        output_file.write_text(output, encoding="utf-8")
    # 3. Detailed mode
    elif mode == "detailed":
        # Step details
        detailed_file = root_dir / "detailed.txt"
        write_list(
            detailed_file,
            data=[f"{x}\t{y}" for x, y in zip(content.split("\n"), output.split("\n"))],
        )
    # 4. Wordmap mode
    elif mode == "wordmap":
        # Unique words
        unique_file = root_dir / "unique.txt"
        uniq_words = sorted(set(output.split("\n")))
        write_list(unique_file, uniq_words)
        # Wordmap
        wordmap_file = root_dir / "wordmap.txt"
        wordmap = {x: y for x, y in zip(content.split("\n"), output.split("\n"))}
        save_wordmap(wordmap=wordmap, wordmap_file=wordmap_file)


def evaluate(
    func: Callable,
    root_dir: str | Path,
    model_name: str,
):
    output_dir = Path(root_dir) / model_name
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory for evaluation: {output_dir.as_posix()}")
    target_file, result_file, output_file, comparison_file = (
        root_dir / "target.txt",
        output_dir / "result.txt",
        output_dir / "output.txt",
        output_dir / "comparison.txt",
    )
    target_dict = read_dict(target_file)
    inputs = sorted(target_dict.keys())
    output_dict = {x: func(x) for x in tqdm(inputs, desc=f"Executing {model_name}")}
    num_mismatch = 0  # Number of words with error
    comparison = []
    err = 0  # Total edit distance
    M, N = len(inputs), 0
    for x in tqdm(inputs, desc=f"Evaluating {model_name}"):
        target = target_dict[x]
        output = output_dict[x]
        if target != output:
            num_mismatch += 1
        edit_distance = enchant.utils.levenshtein(target, output)
        err += edit_distance
        comparison.append(f"{x}\t{target}\t{output}\t{edit_distance}")
        N += max(len(target), len(output))

    evaluation = (
        f"{(num_mismatch/M)*100:.02f}\n{(err/N)*100:.02f}\n"
        f"Word Level Accuracy={(1-num_mismatch/M)*100=:.02f}% | {num_mismatch=} | {M=}\n"
        f"Character Level Accuracy={(1-err/N)*100=:.02f}% | {err=} | {N=}"
    )
    print(evaluation)
    result_file.write_text(evaluation)
    write_list(output_file, [f"{x}\t{y}" for x, y in output_dict.items()])
    write_list(comparison_file, comparison)


# Preparation
def prepare_files(
    src_file: str | Path,
    target_dir: str | Path,
) -> None:
    print(f"Preparing files:\n{src_file} --> --> --> {target_dir}\n")
    target_dict = read_dict(src_file)
    write_dict(Path(target_dir) / "target.txt", target_dict)
    write_list(Path(target_dir) / "words.txt", target_dict.keys(), True)
    print(f"No of words: {len(target_dict)}")


# Runner functions
def run_gc():
    prepare_files("data/corrected.txt", "data/gc_")
    gc = GlyphCorrection()
    for mode in all_modes:
        run(gc.correct_words, mode)


def run_mt():
    prepare_files("data/transcribed.txt", "data/mt_")
    mt = MTransliteration()
    for mode in all_modes:
        run(mt.transliterate_words, mode)
    base1 = Baseline()
    run(base1.transliterate, "evaluate", model_name="Baseline")
    base2 = BaselineExtended()
    run(base2.transliterate, "evaluate", model_name="Baseline 2")


if __name__ == "__main__":
    run_gc()
    run_mt()


# Clean
def clean_target(file: str | Path) -> None:
    """Clean the target file."""
    c = Cleaner()
    target_file: Path = Path(file)
    new_lines = set()
    lines = target_file.read_text(encoding="utf-8").strip().split("\n")
    for line in tqdm(lines, desc="Cleaning..."):
        word_bn, word_mm = line.split("\t")
        cleaned_word_bn = word_bn
        cleaned_word_mm = c.clean_mm_utf(word_mm)
        new_lines.add(f"{cleaned_word_bn}\t{cleaned_word_mm}")
    target_file.write_text("\n".join(sorted(new_lines)), encoding="utf-8")


# clean_target("data/mt_/target.txt")
