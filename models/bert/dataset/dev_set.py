# models/evaluation/dev_set.py
from dataclasses import dataclass
import re
from backend.core.preprocess import (
    clean_supplements,
    transpile,
    remove_punctuation,
    get_tokens_from_clean_text,
)
from backend.core import SUPPLEMENTS_REGEX


@dataclass
class DevCase:
    x: str  # training text con lacuna [MASK-placeholder]
    y: list[str]  # gold labels normalizzate
    gap_length: int  # numero dei caratteri alfabetici della lacuna
    corpus_id: str
    file_id: str


def build_dev_cases(
    ab: dict,
    normalize: bool = False,
    strip_diacritics: bool = False,
    case_folding: bool = False,
) -> list[DevCase]:
    """
    Dato un blocco anonimo di MAAT, estrae tutti i casi (X, Y)
    in cui è presente un supplemento papirologico.

    Args:
        ab: blocco anonimo MAAT (dict con 'training_text', 'corpus_id', ecc.)
        normalize: se True, normalizza X e Y (case folding)
        strip_diacritics: se True, rimuove i diacritici da X e Y
        case_folding: se True, converte il testo in maiuscolo

    Returns:
        lista di DevCase, uno per ogni supplemento trovato nel blocco
    """
    training_text = ab.get("training_text", ab.get("text", ""))
    if not training_text:
        return []

    # Trova tutti i supplementi con la loro posizione nel testo
    matches = list(SUPPLEMENTS_REGEX.finditer(training_text))
    if not matches:
        return []

    # Calcola le gold label tramite clean_supplements (già normalizzate)
    gold_labels = clean_supplements(
        training_text,
        case_folding=case_folding,
        strip_diacritics=strip_diacritics,
        normalize=normalize,
    )

    cases = []
    for match, (expanded_tokens, gap_len) in zip(matches, gold_labels):
        if not expanded_tokens or gap_len == 0:
            continue

        gold_tokens = get_tokens_from_clean_text(
            remove_punctuation(
                transpile(
                    match.group(0),
                    case_folding=case_folding,
                    strip_diacritics=strip_diacritics,
                    normalize=normalize,
                )
            )
        )

        if not gold_tokens:
            continue

        # placeholder temporaneo per la lacuna
        tgt_placeholder = "ΩΩΩMASKΩΩΩ"
        placeholder = "." * gap_len
        target_lacuna_repr = f"[{placeholder}]"

        x_raw = (
            training_text[: match.start()]
            + tgt_placeholder
            + training_text[match.end() :]
        )

        # Transpile si occupa di ripulire i markup editoriali e trasformare
        # le altre lacune in <UNK> tramite `clean_tokens`
        x_clean = transpile(
            x_raw,
            case_folding=case_folding,
            strip_diacritics=strip_diacritics,
            normalize=normalize,
        )

        # Il placeholder dopo eventuale normalizzazione e case folding
        search_target = (
            tgt_placeholder.upper() if (case_folding and normalize) else tgt_placeholder
        )

        # Sostituisce il marker temporaneo con la lacuna fedelmente riprodotta
        x_clean = x_clean.replace(search_target, target_lacuna_repr)
        
        # Validazione: assicurarsi che ci sia esattamente una lacuna e che il marker sia stato sostituito correttamente
        if len(re.findall(r"\[\.+\]", x_clean)) != 1:
            continue
        
        cases.append(
            DevCase(
                x=x_clean,
                y=gold_tokens,
                gap_length=gap_len,
                corpus_id=ab.get("corpus_id", ""),
                file_id=ab.get("file_id", ""),
            )
        )

    return cases


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
        for case in build_dev_cases(ab, normalize=normalize):
            if case.gap_length < min_gap_length:
                continue
            if max_gap_length and case.gap_length > max_gap_length:
                continue
            dev_set.append(case)
    return dev_set
