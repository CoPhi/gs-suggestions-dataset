"""
Costruzione del training set grezzo (model-agnostic) per il finetuning
di modelli BERT su testi in greco antico dal corpus MAAT.

Il dataset prodotto preserva diacritici e punteggiatura: la normalizzazione
model-specific avviene a valle in `load.py` tramite `prepare_dataset_for_model`.
"""

from __future__ import annotations
from backend.core import _CASE_FOLDING

from typing import Any
from datasets import Dataset
from tqdm import tqdm
from backend.core import UNK_TOKEN
from backend.core.cleaner import get_sentences, SentenceRecord
from models.bert.finetuning import MIN_SENT_TOKEN_TRESHOLD

# Helpers interni


def _create_flat_record(record: SentenceRecord) -> dict[str, Any]:
    """
    Unisce i token della frase in un'unica stringa separata da spazi ("text") e
    appiattisce tutti i campi di metadati del blocco anonimo al primo livello del dizionario.

    Args:
        record: Il SentenceRecord contenente i token e i metadati da formattare.
    Returns:
        dict: Un dizionario piatto pronto per essere inserito nel Dataset Hugging Face.
    """
    flat_record: dict[str, Any] = {"text": " ".join(record["sentence_tokens"])}
    if record.get("metadata"):
        flat_record.update(record["metadata"])
    return flat_record


# Filtraggio qualità (word-level, model-agnostic)


def is_quality_sentence(
    tokens: list[str],
    unk_ratio_threshold: float = 0.1,
) -> bool:
    """
    Verifica i criteri di qualità su word token.

    Una frase è accettabile se:
    - ha almeno MIN_SENT_TOKEN_TRESHOLD token
    - la frazione di token <UNK> è inferiore a *unk_ratio_threshold*

    Args:
        tokens:               Lista di word token della frase.
        unk_ratio_threshold:  Frazione massima di [UNK] consentita.

    Returns:
        True se la frase supera entrambe le soglie.
    """
    if len(tokens) < MIN_SENT_TOKEN_TRESHOLD:
        return False

    unk_count = sum(1 for t in tokens if t == UNK_TOKEN)
    return unk_count <= len(tokens) * unk_ratio_threshold


# Costruzione del training set


def build_train_sentences(
    abs_: list[dict[str, Any]], case_folding: _CASE_FOLDING = "none"
) -> list[dict[str, Any]]:
    """
    Estrae le frasi grezze dai blocchi anonimi MAAT applicando solo
    la pulizia editoriale (markup rimosso, lacune → <UNK>).

    Diacritici, punteggiatura e casing originale sono **preservati**
    per garantire la compatibilità con qualsiasi tokenizer BERT.

    Args:
        abs_: Lista di blocchi anonimi (filtrati per language == "grc").
        case_folding: Tipo di case folding da applicare ("none", "upper", ecc.).

    Returns:
        Lista di dizionari piatti, ciascuno rappresentante una frase con i relativi metadati.
    """
    sentences: list[dict[str, Any]] = []
    for record in tqdm(
        get_sentences(
            abs_,
            case_folding=case_folding,
            remove_punct=False,
            normalize=False,
            strip_diacritics=False,
            metadata=True,
        ),
        desc="Building train sentences",
        unit="sentence",
        leave=False,
    ):
        if not is_quality_sentence(record["sentence_tokens"]):
            continue
        sentences.append(_create_flat_record(record))
    return sentences


def build_train_set(
    abs_: list[dict[str, Any]], case_folding: _CASE_FOLDING = "none"
) -> Dataset:
    """
    Produce il train set HuggingFace grezzo dal corpus MAAT.

    Wrapper di `build_train_sentences` che restituisce un oggetto
    `Dataset` pronto per il push sull'Hub o per la normalizzazione
    model-specific tramite `prepare_dataset_for_model`.

    Args:
        abs_: Lista di blocchi anonimi MAAT.
        case_folding: Tipo di case folding da applicare.

    Returns:
        Dataset HuggingFace contenente la frase e tutti i metadati piatti del blocco originario.
    """
    return Dataset.from_list(build_train_sentences(abs_, case_folding=case_folding))
