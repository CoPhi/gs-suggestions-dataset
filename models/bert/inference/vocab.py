from __future__ import annotations

import gzip
import os
import re
from pathlib import Path
from typing import Any, Optional, Sequence, Set

from backend.core import UNK_TOKEN
from backend.core.preprocess import (
    normalize_greek,
    process_editorial_marks,
    remove_punctuation,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_TLG_VOCAB_PATH = PROJECT_ROOT / "backend" / "core" / "resources" / "tlg.wl.gz"

# Uniche parole greche autonome di 1 sola lettera (ὁ/ὅ, ἡ/ἤ/ἦ, ἃ, ὦ)
VALID_SINGLE_CHAR_WORDS = frozenset({"ο", "η", "α", "ω"})

_AFFIX_LACUNA_REGEX = re.compile(r"([^\W\d_]*)\[(\.+|\-+)\]([^\W\d_]*)")
_TARGET_LACUNA_REGEX = re.compile(r"\[(\.+|\-+)\]")
_PRE_NORMALIZED_GREEK_REGEX = re.compile(r"^[α-ωσ]+$")

_vocab_cache: Optional[Set[str]] = None
_vocab_cache_by_path: dict[str, Set[str]] = {}


def _strict_sanitize(text: str) -> str:
    return re.sub(r"[\s\u200B-\u200D\uFEFF]", "", text).strip()


def normalize_for_tlg_lookup(word: str) -> str:
    """
    Normalizza una forma greca secondo lo standard canonico del lessico TLG:
    - rimozione diacritici (`strip_diacritics_flag=True`)
    - normalizzazione CLTK e case folding Unicode (`case_folding="fold"`, che uniforma `ς` -> `σ`)
    - rimozione punteggiatura e spazi invisibili.
    """
    if not word:
        return ""
    norm = normalize_greek(word, case_folding="fold", strip_diacritics_flag=True)
    norm = remove_punctuation(norm)
    return _strict_sanitize(norm)


def _resolve_vocab_path(filepath: Optional[str] = None) -> Optional[Path]:
    """
    Risolve il percorso del file lessicale TLG privilegiando l'artefatto pre-normalizzato
    compresso `backend/core/resources/tlg.wl.gz` (incluso anche nel container `gs-api`).
    """
    if filepath and filepath not in ("tlg.wl.gz", str(DEFAULT_TLG_VOCAB_PATH)):
        custom = Path(filepath)
        if custom.exists():
            return custom
        rel_custom = PROJECT_ROOT / filepath
        if rel_custom.exists():
            return rel_custom
        return None

    if DEFAULT_TLG_VOCAB_PATH.exists():
        return DEFAULT_TLG_VOCAB_PATH

    if filepath:
        direct = Path(filepath)
        if direct.exists():
            return direct

    return None


def get_normalized_vocab(filepath: Optional[str] = None) -> Set[str]:
    """
    Carica e restituisce il vocabolario greco normalizzato del TLG (`backend/core/resources/tlg.wl.gz`),
    usando una cache in-memory.

    L'artefatto pre-normalizzato viene caricato direttamente in O(N) senza dover rieseguire
    la normalizzazione CLTK riga per riga a runtime.
    """
    global _vocab_cache

    resolved_path = _resolve_vocab_path(filepath)
    if resolved_path is None:
        return set()

    cache_key = str(resolved_path.resolve())
    if cache_key in _vocab_cache_by_path:
        _vocab_cache = _vocab_cache_by_path[cache_key]
        return _vocab_cache

    loaded_vocab: Set[str] = set()
    open_fn = (
        (lambda p: gzip.open(p, "rt", encoding="utf-8"))
        if str(resolved_path).endswith(".gz")
        else (lambda p: open(p, "r", encoding="utf-8"))
    )

    with open_fn(resolved_path) as f:
        for line in f:
            word = line.strip()
            if not word:
                continue

            # Fast-path per file pre-normalizzati (es. tlg.wl.gz): già privi di diacritici,
            # maiuscole e sigma finale 'ς'.
            if _PRE_NORMALIZED_GREEK_REGEX.match(word):
                norm = word
            else:
                norm = normalize_for_tlg_lookup(word)

            if not norm:
                continue
            if len(norm) == 1 and norm not in VALID_SINGLE_CHAR_WORDS:
                continue

            loaded_vocab.add(norm)

    _vocab_cache_by_path[cache_key] = loaded_vocab
    _vocab_cache = loaded_vocab
    return loaded_vocab


def extract_lacuna_affixes(text_with_gap: str) -> tuple[str, str]:
    """
    Estrae il prefisso (`word_prefix`) e il suffisso (`word_suffix`) strettamente adiacenti
    alla prima lacuna target `[...]` o `[---]` nel contesto, applicando preventivamente
    la pulizia dei segni editoriali e isolando eventuali lacune secondarie (`<UNK>`).
    """
    if not text_with_gap:
        return "", ""

    parts = re.split(r"(\[.*?\])", text_with_gap)
    cleaned_loose_dots = "".join(
        re.sub(r"\.{2,}", f" {UNK_TOKEN} ", part) if i % 2 == 0 else part
        for i, part in enumerate(parts)
    )
    processed = process_editorial_marks(cleaned_loose_dots, preserve_lacunae=True)

    target_match = _TARGET_LACUNA_REGEX.search(processed)
    if not target_match:
        return "", ""

    affix_match = _AFFIX_LACUNA_REGEX.search(processed)
    if not affix_match or affix_match.start(2) != target_match.start(1):
        return "", ""

    word_prefix = affix_match.group(1) or ""
    word_suffix = affix_match.group(3) or ""
    if word_prefix.upper() == "UNK" or UNK_TOKEN in word_prefix:
        word_prefix = ""
    if word_suffix.upper() == "UNK" or UNK_TOKEN in word_suffix:
        word_suffix = ""

    return word_prefix, word_suffix


def reconstruct_word_with_context(
    supplement: str,
    context_with_gap: str = "",
    word_prefix: Optional[str] = None,
    word_suffix: Optional[str] = None,
    already_full_word: bool = False,
) -> tuple[str, str]:
    """
    Esegue la *boundary reconstruction* concatenando il suggerimento con il prefisso e/o
    suffisso strettamente adiacenti alla lacuna nel contesto.

    Args:
        supplement: Il supplemento proposto dal modello per la lacuna.
        context_with_gap: Contesto testuale originale contenente la lacuna `[...]` / `[---]`.
        word_prefix: Prefisso esplicito (se None, estratto da `context_with_gap`).
        word_suffix: Suffisso esplicito (se None, estratto da `context_with_gap`).
        already_full_word: Se True (es. modelli N-grammi), `supplement` rappresenta già la parola
                           intera ricostruita e non vengono ri-concatenati gli affissi.

    Returns:
        Tupla `(reconstructed_display, reconstructed_normalized)`:
        - `reconstructed_display`: Forma ricostruita in minuscolo leggibile per il frontend.
        - `reconstructed_normalized`: Forma normalizzata con `case_folding="fold"` e senza diacritici per il lookup.
    """
    if not supplement or not isinstance(supplement, str):
        return "", ""

    clean_supp = supplement.strip()
    if not clean_supp:
        return "", ""

    if already_full_word:
        raw_reconstructed = clean_supp
    else:
        if word_prefix is None or word_suffix is None:
            ext_prefix, ext_suffix = extract_lacuna_affixes(context_with_gap)
            prefix = ext_prefix if word_prefix is None else word_prefix
            suffix = ext_suffix if word_suffix is None else word_suffix
        else:
            prefix = word_prefix
            suffix = word_suffix

        raw_reconstructed = f"{prefix}{clean_supp}{suffix}"

    display_form = normalize_greek(
        raw_reconstructed,
        case_folding="lower",
        strip_diacritics_flag=False,
    ).strip()

    # Se la stringa contiene più token separati da spazio (es. N-grammi con num_tokens > 1),
    # normalizza ciascun token mantenendo lo spazio singolo come separatore.
    if " " in raw_reconstructed:
        norm_tokens = [
            normalize_for_tlg_lookup(tok)
            for tok in raw_reconstructed.split()
            if tok.strip()
        ]
        norm_form = " ".join(t for t in norm_tokens if t)
    else:
        norm_form = normalize_for_tlg_lookup(raw_reconstructed)

    return display_form, norm_form


def check_tlg_attestation(
    supplement: str,
    context_with_gap: str = "",
    word_prefix: Optional[str] = None,
    word_suffix: Optional[str] = None,
    already_full_word: bool = False,
    vocab_path: Optional[str] = None,
    vocab: Optional[Set[str]] = None,
) -> tuple[bool, str]:
    """
    Verifica l'attestazione lessicale nel TLG della forma ottenuta tramite *boundary reconstruction*
    (`word_prefix + supplement + word_suffix`).

    Returns:
        Tupla `(tlg_attested, reconstructed_word)`.
    """
    display_form, norm_form = reconstruct_word_with_context(
        supplement=supplement,
        context_with_gap=context_with_gap,
        word_prefix=word_prefix,
        word_suffix=word_suffix,
        already_full_word=already_full_word,
    )
    if not norm_form:
        return False, display_form

    active_vocab = vocab if vocab is not None else get_normalized_vocab(vocab_path)
    if not active_vocab:
        return False, display_form

    tokens = norm_form.split()
    is_attested = bool(tokens) and all(tok in active_vocab for tok in tokens)
    return is_attested, display_form


def postprocess_tlg_attestations(
    candidates: Sequence[tuple[str, float]],
    context_with_gap: str = "",
    word_prefix: Optional[str] = None,
    word_suffix: Optional[str] = None,
    already_full_word: bool = False,
    vocab_path: Optional[str] = None,
    vocab: Optional[Set[str]] = None,
) -> list[dict[str, Any]]:
    """
    Operazione di post-processing non invasiva eseguita dopo la decodifica (es. HCB Beam Search):
    preserva inalterati l'ordinamento e gli score dei candidati, arricchendo ciascun suggerimento
    con l'esito del lookup lessicale sul TLG (`tlg_attested`) e la parola ricostruita (`reconstructed_word`).
    """
    active_vocab = vocab if vocab is not None else get_normalized_vocab(vocab_path)
    if not already_full_word and (word_prefix is None or word_suffix is None):
        ext_prefix, ext_suffix = extract_lacuna_affixes(context_with_gap)
        resolved_prefix = ext_prefix if word_prefix is None else word_prefix
        resolved_suffix = ext_suffix if word_suffix is None else word_suffix
    else:
        resolved_prefix = word_prefix or ""
        resolved_suffix = word_suffix or ""

    annotated: list[dict[str, Any]] = []
    for supp, score in candidates:
        if not isinstance(supp, str):
            continue
        is_attested, recon_word = check_tlg_attestation(
            supplement=supp,
            context_with_gap=context_with_gap,
            word_prefix=resolved_prefix,
            word_suffix=resolved_suffix,
            already_full_word=already_full_word,
            vocab=active_vocab,
        )
        annotated.append(
            {
                "token_str": supp,
                "score": float(score),
                "tlg_attested": is_attested,
                "reconstructed_word": recon_word,
            }
        )
    return annotated
