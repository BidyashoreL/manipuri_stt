import json
import re
from pathlib import Path

MANIPURI_SUFFIXES = [
    "thabadamak",
    "tharakpanina",
    "rakpanina",
    "panina",
    "ragasu",
    "lagasu",
    "basina",
    "sidi",
    "dadi",
    "gadi",
    "haina",
    "badi",
    "padi",
    "basi",
    "tago",
    "ramba",
    "lammi",
    "gani",
    "ge",
    "le",
    "gi",
    "di",
    "se",
    "si",
    "pa",
    "ba",
]

PASSTHROUGH_LOANWORDS = {
    "india",
    "nepal",
    "myanmar",
    "japan",
    "china",
    "football",
    "example",
    "restaurant",
    "hotel",
    "beef",
    "biryani",
    "delivered",
    "exposed",
    "journalist",
    "questions",
    "media",
    "thank",
    "you",
    "register",
    "number",
}

VOWEL_PAT_1 = re.compile(r"eikhoi|aykhoy|ekhoi")
VOWEL_PAT_2 = re.compile(r"ee|ei|ay|ai")
VOWEL_PAT_3 = re.compile(r"oo|ow|ou")
VOWEL_PAT_4 = re.compile(r"aa")

CONS_PAT_KH = re.compile(r"kh")
CONS_PAT_TH = re.compile(r"t")
CONS_PAT_PH_F = re.compile(r"ph|f")
CONS_PAT_SH = re.compile(r"sh")
CONS_PAT_Z = re.compile(r"z")

REPEAT_CHARS = re.compile(r"(\w)\1+")


def phonetic_collapse(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = VOWEL_PAT_1.sub("aykhoy", text)
    text = VOWEL_PAT_2.sub("i", text)
    text = VOWEL_PAT_3.sub("u", text)
    text = VOWEL_PAT_4.sub("a", text)
    text = CONS_PAT_KH.sub("k", text)
    text = CONS_PAT_TH.sub("t", text)
    text = CONS_PAT_PH_F.sub("p", text)
    text = CONS_PAT_SH.sub("s", text)
    text = CONS_PAT_Z.sub("j", text)
    text = REPEAT_CHARS.sub(r"\1", text)
    return text


def stem_manipuri_word(word: str) -> str:
    cleaned_word = word.lower().strip()
    for suffix in MANIPURI_SUFFIXES:
        if cleaned_word.endswith(suffix) and len(cleaned_word) > len(suffix) + 2:
            return cleaned_word[: -len(suffix)]
    return cleaned_word


class ManipuriPhoneticEngine:

    def __init__(self, json_path: Path = None):
        self.exact_map = {}
        self.phonetic_root_map = {}
        if json_path and json_path.exists():
            self.load_dictionary(json_path)

    def load_dictionary(self, json_path: Path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for standard_key, variants in data.items():
            sk_lower = standard_key.lower().strip()
            self.exact_map[sk_lower] = standard_key
            self.phonetic_root_map[phonetic_collapse(sk_lower)] = standard_key

            for var in variants:
                v_lower = var.lower().strip()
                self.exact_map[v_lower] = standard_key
                self.phonetic_root_map[phonetic_collapse(v_lower)] = standard_key

    def resolve_word(self, word: str) -> str:
        w_lower = word.lower().strip()
        if not w_lower:
            return ""

        # Update ACOUSTIC_PATCHES locally inside this block to catch compound misses from your logs
        ACOUSTIC_PATCHES = {
            "meheshila": "messages",
            "barat": "bharat",
            "prajah": "praja",
            "taudoxingasi": "thougangsing",
            "bashaasee": "peisase",
            "harhapana": "har hapana",
            "kamainamapaan": "kamaina mapaan",
            "hoegihachanggi": "eikhoi hakchanggi",
            "nidaasua": "yonphamdasu",
            "well-disciplined": "well disciplined",
            "i don't know": "",
            "janpam": "yonpham",
            "hangia": "hangge",
            "suparkamayinahakwadake": "super kamaina hapkadage",  # Fixing Sample 01 giant fusion
        }

        if w_lower in ACOUSTIC_PATCHES:
            patched = ACOUSTIC_PATCHES[w_lower]
            if not patched:
                return ""
            if " " in patched:
                return " ".join(self.resolve_word(seg) for seg in patched.split())
            w_lower = patched

        if w_lower.isdigit() or w_lower in PASSTHROUGH_LOANWORDS:
            return w_lower

        if w_lower in self.exact_map:
            return self.exact_map[w_lower]

        stemmed = stem_manipuri_word(w_lower)
        if stemmed in self.exact_map:
            return self.exact_map[stemmed]

        collapsed = phonetic_collapse(w_lower)
        if collapsed in self.phonetic_root_map:
            return self.phonetic_root_map[collapsed]

        collapsed_stemmed = phonetic_collapse(stemmed)
        if collapsed_stemmed in self.phonetic_root_map:
            return self.phonetic_root_map[collapsed_stemmed]

        if len(w_lower) > 7:
            for i in range(3, len(w_lower) - 2):
                prefix, suffix = w_lower[:i], w_lower[i:]
                if prefix in self.exact_map and suffix in self.exact_map:
                    return f"{self.exact_map[prefix]} {self.exact_map[suffix]}"
                p_stem, s_stem = stem_manipuri_word(prefix), stem_manipuri_word(suffix)
                if p_stem in self.exact_map and s_stem in self.exact_map:
                    return f"{self.exact_map[p_stem]} {self.exact_map[s_stem]}"

        best_match = None
        min_distance = 999

        # FIX: STRICT DISTANCE THRESHOLD
        max_allowed_distance = 1 if len(collapsed) <= 5 else 2

        for baseline_collapsed, standard_word in self.phonetic_root_map.items():
            if abs(len(collapsed) - len(baseline_collapsed)) > max_allowed_distance:
                continue

            # FIX: STRICT FIRST LETTER GUARD (Stops "marung" mapping to "papung", "marang" -> "hameng")
            if (
                collapsed
                and baseline_collapsed
                and collapsed[0] != baseline_collapsed[0]
            ):
                continue

            vowels_w = set(re.sub(r"[^aeiou]", "", collapsed))
            vowels_b = set(re.sub(r"[^aeiou]", "", baseline_collapsed))
            if vowels_w != vowels_b and len(vowels_w.intersection(vowels_b)) == 0:
                continue

            distance = self._levenshtein_distance(collapsed, baseline_collapsed)
            if distance <= max_allowed_distance and distance < min_distance:
                min_distance = distance
                best_match = standard_word

        if best_match:
            return best_match

        return f"[{word}]"

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def clean_sentence(self, text: str) -> str:
        if not text:
            return ""
        return " ".join(self.resolve_word(w) for w in text.split())
