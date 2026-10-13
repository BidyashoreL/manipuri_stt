from .correction import Correction

__all__ = ["GlyphCorrection", "evaluate_gc"]


class GlyphCorrection:

    def __init__(self) -> None:
        self.correction = Correction()

    def correct_words(
        self,
        text: str,
        show_steps: bool = False,
    ) -> str:
        words = []
        for word in text.split():
            words.append(self.correct(text=word, include_steps=show_steps))
        return "\n".join(words)

    def correct(
        self,
        text: str,
        include_steps: bool = False,
    ) -> str:
        return self.correction.correct(text=text, include_steps=include_steps)
