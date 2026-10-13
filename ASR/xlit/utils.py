import csv
import json
from pathlib import Path
from typing import Collection, Dict, List, Tuple


def save_wordmap(wordmap: Dict[str, str], wordmap_file: str | Path):
    """Save wordmap to .txt, .json, and .csv files."""
    txt_path = Path(wordmap_file).with_suffix(".txt")
    txt_path.write_text(
        "\n".join([f"{word}\t{corrected}" for word, corrected in wordmap.items()]),
        encoding="utf-8",
    )

    json_path = Path(wordmap_file).with_suffix(".json")
    json_path.write_text(json.dumps(wordmap, ensure_ascii=False), encoding="utf-8")

    csv_path = Path(wordmap_file).with_suffix(".csv")
    with csv_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=("lang1", "lang2"))
        writer.writeheader()
        for word1, word2 in wordmap.items():
            writer.writerow({"lang1": word1, "lang2": word2})


def read_list(
    file: str | Path,
    sort: bool = False,
) -> List[str]:
    """Read a file and return a list of lines."""
    data = Path(file).read_text(encoding="utf-8").strip().split("\n")
    return sorted(data) if sort else data


def read_dict(
    file: str | Path,
    delimiter: str = "\t",
    kv_field: Tuple[int, int] = (0, 1),
) -> Dict[str, str]:
    """Read a file and return a dictionary."""
    return str_to_dict(
        Path(file).read_text(encoding="utf-8").strip(), delimiter, kv_field
    )


def write_list(
    file: str | Path,
    data: Collection,
    sort: bool = False,
) -> None:
    """Write a list to a file."""
    data = "\n".join(sorted(data) if sort else list(data))
    return Path(file).write_text(data, encoding="utf-8")


def write_dict(
    file: str | Path,
    data: Dict,
    delimiter: str = "\t",
) -> None:
    """Write a dictionary to a file."""
    Path(file).write_text(dict_to_str(data, delimiter), encoding="utf-8")


def str_to_dict(
    string: str,
    delimiter: str = "\t",
    kv_field: tuple[int, int] = (0, 1),
) -> Dict[str, str]:
    """Convert a delimited string to a dictionary."""
    return {
        line.split(delimiter)[kv_field[0]]: line.split(delimiter)[kv_field[1]]
        for line in string.split("\n")
    }


def dict_to_str(
    dictionary: dict,
    delimiter: str = "\t",
) -> str:
    """Convert a dictionary to a delimited string."""
    return "\n".join([f"{key}{delimiter}{val}" for key, val in dictionary.items()])
