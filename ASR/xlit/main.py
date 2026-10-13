import argparse

from gc_ import GlyphCorrection
from mt_ import MTransliteration
from run import run, run_mt, run_gc


def main() -> None:
    run_gc()
    run_mt()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run from main")
    parser.add_argument("-m", action="store_true", help="Select module mt")
    parser.add_argument("-g", action="store_true", help="Select module gc")
    parser.add_argument("-d", action="store_true", help="Enable detailed mode")
    parser.add_argument("-w", action="store_true", help="Enable wordmap mode")
    parser.add_argument("-e", action="store_true", help="Enable evaluation mode")
    parser.add_argument(
        "--root", help="Directory path which contains words.txt or targets.txt"
    )

    args = parser.parse_args()

    if args.m:
        mt = MTransliteration()
        func = mt.transliterate_words
    elif args.g:
        gc = GlyphCorrection()
        func = gc.correct_words
    else:
        main()

    mode = (
        "detailed"
        if args.d
        else ("wordmap" if args.w else ("evaluation" if args.e else "simple"))
    )
    run(func=func, mode=mode, root_dir=args.root)
    # Should contain targets.txt (evaluation) or words.txt (others) in args.root directory
