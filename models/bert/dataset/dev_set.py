# models/evaluation/dev_set.py
from dataclasses import dataclass, field
from backend.core.preprocess import (
    clean_supplements,
    process_editorial_marks,
    normalize_greek,
    get_tokens_from_clean_text,
    remove_punctuation,
)
from backend.core import SUPPLEMENTS_REGEX
import re

@dataclass
class DevCase:
    x: str  # training text con lacuna [MASK-placeholder]
    y: list[str]  # gold labels normalizzate
    gap_length: int  # caratteri alfabetici della lacuna
    left_context: str  # sottostringa sinistra della lacuna
    right_context: str  # sottostringa destra della lacuna
    corpus_id: str
    abs_id: str
    raw_supplement: str  # supplemento originale non normalizzato


def build_dev_case(
    ab: dict, normalize: bool = False, strip_diacritics: bool = False, case_folding: bool = False
) -> list[DevCase]:
    """
    Dato un blocco anonimo di MAAT, estrae tutti i casi (X, Y)
    in cui è presente un supplemento papirologico.

    Args:
        ab: blocco anonimo MAAT (dict con 'training_text', 'corpus_id', ecc.)
        normalize: se True, normalizza X e Y con `normalize_greek` (rimozione diacritici)
        strip_diacritics: se True, rimuove i diacritici da X e Y

    Returns:
        lista di DevCase, uno per ogni supplemento trovato nel blocco
    """
    training_text = ab.get("training_text", "")
    if not training_text:
        return []

    # Trova tutti i supplementi con la loro posizione nel testo
    matches = list(SUPPLEMENTS_REGEX.finditer(training_text))
    if not matches:
        return []

    # Calcola le gold label tramite clean_supplements (già normalizzate)
    gold_labels = clean_supplements(
        training_text,
        case_folding=normalize,
        strip_diacritics=strip_diacritics,
        normalize=normalize,
    )

    cases = []
    for match, (gold_tokens, gap_len) in zip(matches, gold_labels):
        if not gold_tokens or gap_len == 0:
            continue

        placeholder = "." * gap_len
        x_raw = (
            training_text[: match.start()]
            + f"[{placeholder}]"
            + training_text[match.end() :]
        )

        x_clean = process_editorial_marks(x_raw)
        if normalize:
            x_clean = normalize_greek(x_clean, case_folding=case_folding)

        # Estrai left/right context rispetto al placeholder
        left, right = _split_context(x_clean)

        cases.append(
            DevCase(
                x=x_clean,
                y=gold_tokens,
                gap_length=gap_len,
                left_context=left,
                right_context=right,
                corpus_id=ab.get("corpus_id", ""),
                abs_id=ab.get("abs_id", ""),
                raw_supplement=match.group(0),
            )
        )

    return cases


def _split_context(x_clean: str) -> tuple[str, str]:
    """
    Separa il contesto sinistro e destro rispetto al placeholder della lacuna.
    """
    # Il placeholder dopo normalize potrebbe essere alterato: cerca i punti
    gap_pattern = re.compile(r"\[\.+\]")
    m = gap_pattern.search(x_clean)
    if not m:
        return x_clean, ""

    left = x_clean[: m.start()].strip()
    right = x_clean[m.end() :].strip()
    return left, right


def build_dev_set(
    herc_abs: list[dict],
    normalize: bool = False,
    min_gap_length: int = 1,
    max_gap_length: int | None = None,
) -> list[DevCase]:
    """
    Costruisce il dev set completo dai blocchi anonimi ercolanesi.

    Args:
        herc_abs:       blocchi anonimi dei papiri ercolanesi
        normalize:      normalizza X e Y
        min_gap_length: scarta lacune troppo corte (rumore)
        max_gap_length: scarta lacune troppo lunghe (opzionale)
    """
    dev_set = []
    for ab in herc_abs:
        for case in build_dev_case(ab, normalize=normalize):
            if case.gap_length < min_gap_length:
                continue
            if max_gap_length and case.gap_length > max_gap_length:
                continue
            dev_set.append(case)
    return dev_set
