"""
Costruzione del training set grezzo (model-agnostic) per il finetuning
di modelli BERT su testi in greco antico dal corpus MAAT.

Il dataset prodotto preserva diacritici e punteggiatura: la normalizzazione
model-specific avviene a valle in `load.py` tramite `prepare_dataset_for_model`.
"""

from __future__ import annotations

from datasets import Dataset
from tqdm import tqdm
from backend.core import UNK_TOKEN
from backend.core.cleaner import get_sentences, get_tokens_from_clean_text
from models.bert.finetuning import MIN_SENT_TOKEN_TRESHOLD

# Helpers interni


def _join_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)

# Filtraggio qualità (word-level, model-agnostic)


def is_quality_sentence(
    tokens: list[str],
    unk_ratio_threshold: float = 0.1,
) -> bool:
    """
    Verifica i criteri di qualità su word token.

    Una frase è accettabile se:
    - ha almeno MIN_SENT_TOKEN_TRESHOLD token
    - la frazione di token [UNK] è inferiore a *unk_ratio_threshold*

    Args:
        tokens:               Lista di word token della frase.
        unk_ratio_threshold:  Frazione massima di [UNK] consentita.

    Returns:
        True se la frase supera entrambe le soglie.
    """
    if len(tokens) < MIN_SENT_TOKEN_TRESHOLD:
        return False
    
    unk_count = sum(1 for t in tokens if t == UNK_TOKEN)
    return unk_count < len(tokens) * unk_ratio_threshold
    
# Costruzione del training set

def build_train_sentences(abs_: list) -> list[str]:
    """
    Estrae le frasi grezze dai blocchi anonimi MAAT applicando solo
    la pulizia editoriale (markup rimosso, lacune → [UNK]).

    Diacritici, punteggiatura e casing originale sono **preservati**
    per garantire la compatibilità con qualsiasi tokenizer BERT.

    Args:
        abs_: Lista di blocchi anonimi (filtrati per language == "grc").

    Returns:
        Lista di stringhe, una per frase accettata dal filtro qualità.
    """
    sentences: list[str] = []
    for sent_tkns in tqdm(
        get_sentences(
            abs_,
            case_folding=False,
            remove_punct=False,
            normalize=False,
            strip_diacritics=False,
        ),
        desc="Building train sentences",
        unit="sentence",
        leave=False,
    ):
        if not is_quality_sentence(sent_tkns):
            continue
        sentences.append(_join_tokens(sent_tkns))
    return sentences


def build_train_set(abs_: list) -> Dataset:
    """
    Produce il train set HuggingFace grezzo dal corpus MAAT.

    Wrapper di `build_train_sentences` che restituisce un oggetto
    `Dataset` con colonna 'text', pronto per il push sull'Hub o per
    la normalizzazione model-specific tramite `prepare_dataset_for_model`.

    Args:
        abs_: Lista di blocchi anonimi MAAT.

    Returns:
        Dataset HuggingFace con colonna 'text'.
    """
    return Dataset.from_dict({"text": build_train_sentences(abs_)})
