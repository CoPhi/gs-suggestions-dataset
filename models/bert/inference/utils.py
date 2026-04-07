import re 

from backend.core import SUPPLEMENTS_REGEX
from typing import Optional, Tuple
import unicodedata

def to_24letters_greek_lower(s: str) -> str:
    """
    Rimuove spiriti, accenti e diacritici dal greco politonico,
    porta tutto in minuscolo e mantiene solo le 24 lettere canoniche.
    """
    # Normalize to NFD to separate base letters and diacritics
    s = unicodedata.normalize("NFD", s)
    
    # Remove non-spacing marks (diacritics)
    s = ''.join(ch for ch in s if unicodedata.category(ch) != 'Mn')
    
    # Convert to lowercase
    s = s.lower()
    
    # Map both σ and ς to lunate sigma ϲ
    s = s.replace("σ", "ϲ").replace("ς", "ϲ")
    
    # Keep only the 24 canonical lowercase Greek letters + optional dot
    allowed = set("αβγδεζηθικλμνξοπρϲτυφχψω. []")
    s = ''.join(ch for ch in s if ch in allowed)
    
    return s

def convert_lacuna_to_masks(text: str, mask_token: str) -> Optional[Tuple[str, int, str]]:
    """
    Converte le lacune nel testo in token mascherati.

    Args:
        text (str): Il testo contenente lacune da convertire.
        mask_token (str): Il token mascherato da utilizzare.

    Returns:
        Optional[Tuple[str, int]]: Una tupla contenente il testo con la lacuna sostituita dal token mascherato e la lunghezza della lacuna, oppure None se non viene trovata una singola lacuna.
    """

    print(f"Converting lacuna to masks in text: {text} with mask_token: {mask_token}")

    clean_text = to_24letters_greek_lower(text)
    print(f"Cleaned text: {clean_text}")

    gap_matches = list(SUPPLEMENTS_REGEX.finditer(clean_text))
    print(f"Found gap matches: {gap_matches}")
    if len(gap_matches) != 1:
        return

    seq = gap_matches[0]

    start, end = seq.start(), seq.end()
    
    print(f"Gap found from index {start} to {end}, gap text: '{clean_text[start:end]}'")

    print(f"Returning mask conversion result {to_24letters_greek_lower(clean_text.replace(clean_text[start:end], mask_token))}")
    return (
        clean_text.replace(clean_text[start:end], mask_token),
        end - start - 2,
        clean_text[start+1:end-1]
    )

def to_greek_lower(text: str) -> str:
    """
    Converte una stringa in minuscolo e sostituisce i caratteri greci 'ϲ' e 'Ϲ' con 'σ'.

    Args:
        text (str): La stringa di input da convertire.

    Returns:
        str: La stringa convertita in minuscolo con i caratteri 'ϲ' e 'Ϲ' sostituiti da 'σ'.
    """

    return text.lower().replace("ϲ", "σ").replace("Ϲ", "σ")

def string_to_regex(pattern: str) -> str:
    regex = ''.join(['.' if char == '.' else re.escape(char) for char in pattern])
    return regex


if __name__ == "__main__":
    # Esempio di utilizzo
    text = "Ἀριστοτέλης ἐστὶν ὁ [..]ξαν ἀληθῆ φαίνε."
    mask_token = "[MASK]"
    result = convert_lacuna_to_masks(text, mask_token)
    print(result)