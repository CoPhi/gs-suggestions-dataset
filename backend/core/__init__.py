import re
import unicodedata

from typing import Literal

# Implementazione leggera in pure-Python equivalente a CLTK (`GreekRegexSentenceTokenizer` e `normalize_grc`):
# evita di importare a runtime il pacchetto `cltk` (~237 MB di RSS e dipendenza da `git` nel container `gs-api`).
_TONOS_TO_OXIA_TABLE = str.maketrans(
    {
        "\u03ac": "\u1f71",  # ά -> ά
        "\u03ad": "\u1f73",  # έ -> έ
        "\u03ae": "\u1f75",  # ή -> ή
        "\u03af": "\u1f77",  # ί -> ί
        "\u03cc": "\u1f79",  # ό -> ό
        "\u03cd": "\u1f7b",  # ύ -> ύ
        "\u03ce": "\u1f7d",  # ώ -> ώ
    }
)
_ODD_PUNCT_TABLE = str.maketrans("", "", "\u2018\u201c\u2019\u201d")  # ‘ “ ’ ”


def normalize_grc(text: str) -> str:
    """Normalizzazione canonica del greco antico equivalente a `cltk.alphabet.grc.grc.normalize_grc`."""
    # Converte gli accenti tonos (greco moderno) in oxia, applica NFKC e rimuove virgolette tipografiche curve
    converted = text.translate(_TONOS_TO_OXIA_TABLE)
    normalized = unicodedata.normalize("NFKC", converted)
    return normalized.translate(_ODD_PUNCT_TABLE)


class GreekRegexSentenceTokenizer:
    """Tokenizer di frasi basato su regex equivalente a `cltk.sentence.grc.GreekRegexSentenceTokenizer`."""

    def __init__(self) -> None:
        self.language = "greek"
        self.sent_end_chars = [".", ";", "·"]
        self.sent_end_chars_regex = "|".join(self.sent_end_chars)
        self.pattern = rf"(?<=[{self.sent_end_chars_regex}])\s"
        self._compiled_pattern = re.compile(self.pattern)

    def tokenize(self, text: str, model=None) -> list[str]:
        """Suddivide il testo greco in frasi sui delimitatori `.`, `;` e `·` seguiti da spazio."""
        return self._compiled_pattern.split(text)


# Tokenizer testo -> frasi (str -> list[str])
sentence_tokenizer = GreekRegexSentenceTokenizer()
_LANGUAGE = "grc"

GAP_TOKEN = "<gap/>"
UNK_TOKEN = "<UNK>"

#sequenze di dattili canonici incompleta trovati nei testi
INCOMPLETE_DACTYL_PATTERNS = [
    r"⏑⏑‒", #dattilo canonico
    r"‒⏑⏑‒", #dattilo chiuso
    r"‒⏑⏑", #dattilo parziale
    r"⏑", #singolo breve
    r"‒", #singolo lungo
    r"‒⏑⏑‒⏑⏑‒", 
    r"‒⏑⏑‒⏑⏑‒⏑",
    r"⏑⏑‒⏑", 
    r"‒⏑", 
    r"⏑⏑‒", 
    r"⏑‒⏑⏑", 
    r"⏑⏑‒⏑⏑",  
]
COMBINED_DACTYL_PATTERNS = re.compile('|'.join(f'({p})' for p in INCOMPLETE_DACTYL_PATTERNS))

"""
    Queste espressioni regolari vogliono codificare pattern che rappresentano gli elementi editoriali presenti nella convenzione Leiden+.
    https://papyri.info/docs/leiden_plus
"""
PUNCTUATION_REGEX = re.compile(r"[.·,;:!?'⁑]")
BRACKETS_REGEX = re.compile(r"\[(.*?)\]")
UNMATCHED_BRACKETS_REGEX = re.compile(r"[\[\]]")
SUPPLEMENTS_REGEX = re.compile (r"(\[[^\]]+\])")
MISSING_LINES_REGEX = re.compile(r"⟦-+\⟧")
MARKER_REGEX = re.compile(r"&lt;(.*?)&gt;")
LITERAL_MARKER_REGEX = re.compile(r"<(?!gap/?>|UNK>)(.*?)>")
UNKNOWN_LEFT_MARKER_REGEX = re.compile(r"<gap/>lt;(.*?)&gt;")
EXTENDED_LINE_RIGHT_MARKER_REGEX = re.compile (r"break=\"no\"/&gt;") #
EXPUNCTION_REGEX = re.compile(r"\{\{(.*?)\}\}|\{(.*?)\}")
VACAT_REGEX = re.compile(r"vac\.|vacat")
NOTES_REGEX = re.compile(r"‡\d+")
OBELISK_REGEX = re.compile(r"†.*?†")
LATIN_REFERENCE_REGEX = re.compile(r"\([^)]*[a-zA-Z][^)]*\)\.?\s*")

# Tabella dei diacritici
_DIACRITIC_TABLE = {
    cp: None
    for cp in range(0x0300, 0x1FFF)  # Greek Extended + Combining Diacritical Marks
    if unicodedata.combining(chr(cp))
}

# Case Folding type 
_CASE_FOLDING = Literal["upper", "lower", "fold", "none"]
