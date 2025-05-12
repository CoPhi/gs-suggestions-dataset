from typing import Generator
from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import padded_everygram_pipeline
from train.training import get_sentences
from config.settings import N, LAMBDA
from math import pow
from statistics import mean
from metrics.accuracy import compute_log_prob


def perplexity(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    test_abs: list,
    lambda_weight: float = LAMBDA,
    n=N,
) -> float:
    """
    Valuta un modello di linguaggio calcolando la perplessità sui dati di test.

    Args:
        g_lm (LanguageModel): Il modello linguistico globale, addestrato su più varietà e generi di testi.
        d_lm (LanguageModel): Modello linguistico di dominio, specializzato in un dominio di interesse
        test_abs (list): Una lista di blocchi anonimi (abs) di test.
        n (int, opzionale): La dimensione degli n-grammi. Default è N.

    Returns:
        float: La perplessità del modello sui dati di test.

    Raises:
        ValueError: Se il modello non è stato addestrato correttamente.
        RuntimeError: Se si verifica un errore nel calcolo della perplessità.
    """

    if not g_lm.vocab or not d_lm.vocab:
        raise ValueError(
            "Uno o entrambi i modelli non sono stati addestrati correttamente."
        )

    test_ngrams, _ = padded_everygram_pipeline(n, get_sentences(abs=test_abs))

    return pow(2.0, interpolated_entropy(g_lm, d_lm, test_ngrams, lambda_weight, n))


def interpolated_entropy(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    ngrams: Generator,
    lambda_weight: float,
    n: int = N,
) -> float:
    """
    Calcola l'entropia interpolata tra due modelli di linguaggio.

    Args:
        g_lm (LanguageModel): Modello linguistico globale.
        d_lm (LanguageModel): Modello linguistico di dominio.
        ngrams (Generator): Generatore di n-grammi dai dati di test.
        lambda_weight (float): Peso per l'interpolazione tra i modelli.
        n (int, opzionale): Dimensione degli n-grammi. Default è N.

    Returns:
        float: Entropia interpolata calcolata.

    Raises:
        ValueError: Se non ci sono n-grammi validi.
    """
    total_log_prob = 0.0
    total_token_count = 0

    for sent in ngrams:
        valid_ngrams = (ngram for ngram in sent if len(ngram) == n)
        for ngram in valid_ngrams:
            total_log_prob += compute_log_prob(g_lm, d_lm, ngram[-1], ngram[:-1], lambda_weight)
            total_token_count += 1

    if total_token_count == 0:
        raise ValueError("Nessun n-gramma valido trovato.")

    return -total_log_prob / total_token_count