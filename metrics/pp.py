from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import flatten, padded_everygram_pipeline
from train.training import get_sentences
from config.settings import N


def perplexity(lm: LanguageModel, test_abs: list, n=N) -> float:
    """
    Valuta un modello di linguaggio calcolando la perplessità sui dati di test.

    Args:
        lm (LanguageModel): Il modello di linguaggio da valutare.
        test_abs (list): Una lista di blocchi anonimi (abs) di test.
        n (int, opzionale): La dimensione degli n-grammi. Default è N.

    Returns:
        float: La perplessità del modello sui dati di test.

    Raises:
        ValueError: Se il modello non è stato addestrato correttamente.
        RuntimeError: Se si verifica un errore nel calcolo della perplessità.
    """

    test_ngrams, _ = padded_everygram_pipeline(n, get_sentences(abs=test_abs))

    if not lm.vocab:
        raise ValueError("Il modello non è stato addestrato correttamente.")
    try:
        return round(lm.perplexity(flatten(test_ngrams)), 2)
    except ZeroDivisionError:
        raise RuntimeError(
            "Errore nel calcolo della perplessità. Verifica i dati di test e di addestramento."
        )
