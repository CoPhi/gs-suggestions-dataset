from typing import Literal
import numpy as np
from transformers import AutoTokenizer

from backend.core.cleaner import get_sentences, load_abs
from backend.core.preprocess import get_tokens_from_clean_text
from models.bert.finetuning import get_model_config

ModelType = Literal["bert", "ngram"]

_NGRAM_CONFIG = {
    "strip_diacritics": True,
    "remove_punct": True,
    "case_folding": "upper",
}


def compute_corpus_token_stats(
    corpus_paths: list[str] = None,
    checkpoint: str | None = None,
    model_type: ModelType = "bert",
) -> dict:
    """
    Calcola statistiche descrittive sui token per un insieme di corpus .
    Args:
        corpus_paths: Lista di percorsi ai file di testo da analizzare.
        checkpoint: Percorso al checkpoint del modello (richiesto per BERT).
        model_type: Tipo di modello da utilizzare ("bert" o "ngram").
    Returns:
        Un dizionario contenente la media, varianza, deviazione standard e conteggio dei token.
    Raises:
        ValueError: Se il model_type è "bert" ma il checkpoint non è fornito

    Esempi di utilizzo:
        >>> compute_corpus_token_stats(
            corpus_paths=["tlg", "DCLP"],None,"ngram")
        >>> compute_corpus_token_stats(
            corpus_paths=["tlg", "DCLP"],"CNR-ILC/gs-AristoBERTo","bert")
    """
    if model_type == "bert":
        if not checkpoint:
            raise ValueError("Il checkpoint è obbligatorio per i modelli BERT.")
        config = get_model_config(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenize_fn = tokenizer.tokenize
    elif model_type == "ngram":
        config = _NGRAM_CONFIG
        tokenize_fn = lambda s: get_tokens_from_clean_text(s)
    else:
        raise ValueError(
            f"model_type '{model_type}' non supportato. Usa 'bert' o 'ngram'."
        )

    data = load_abs(corpus_set=corpus_paths)
    sentences = get_sentences(
        abs=data,
        strip_diacritics=config.get("strip_diacritics", False),
        remove_punct=config.get("remove_punct", False),
        case_folding=config.get("case_folding", None),
    )

    token_counts = [len(tokenize_fn(s)) for s in sentences]

    if not token_counts:
        return {"mean": 0.0, "variance": 0.0, "std": 0.0, "count": 0}

    arr = np.array(token_counts)
    return {
        "mean": float(arr.mean()),
        "variance": float(arr.var()),
        "std": float(arr.std()),
        "count": len(arr),
    }
