import inspect
from src.mt_ import MTransliteration

print("=== transliterate_words implementation ===")
print(inspect.getsource(MTransliteration.transliterate_words))

print("\n=== transliterate implementation ===")
print(inspect.getsource(MTransliteration.transliterate))
