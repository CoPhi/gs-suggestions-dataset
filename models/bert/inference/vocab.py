import os
import re
from typing import Optional, Set

from backend.core.preprocess import normalize_greek, remove_punctuation

_vocab_cache: Optional[Set[str]] = None


def get_normalized_vocab(filepath: str = "grc.wl") -> Set[str]:
    """
    Carica e normalizza il vocabolario dal file specificato. Utilizza una cache globale per evitare ricaricamenti multipli.
    Ogni parola appartenente al vocabolario subisce una serie di pulizie: rimozione di spazi, caratteri invisibili, punteggiatura e conversione a maiuscolo (case-folding).
    In questo modo si garantisce che il vocabolario sia coerente con la normalizzazione effettuata durante il fine-tuning.
    """
    global _vocab_cache
    if _vocab_cache is not None:
        return _vocab_cache

    _vocab_cache = set()
    if not os.path.exists(filepath):
        return _vocab_cache

    def _strict_sanitize(text: str) -> str:
        return re.sub(r"[\s\u200B-\u200D\uFEFF]", "", text).strip()

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            word = line.strip()
            if not word:
                continue

            norm = normalize_greek(
                word, case_folding="lower", strip_diacritics_flag=True
            )
            norm = remove_punctuation(norm)
            norm = _strict_sanitize(norm)

            if norm:
                _vocab_cache.add(norm)

    return _vocab_cache
