"""
Modulo per il calcolo della perplessità e dell'entropia interpolata
sui modelli linguistici a n-grammi.
"""

from math import pow
from typing import Generator

from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import padded_everygram_pipeline

from backend.config.settings import LAMBDA, N
from backend.core.cleaner import get_sentences
from models.ngrams.metrics.utils import interpolated_log_score


def perplexity(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    test_abs: list,
    lambda_weight: float = LAMBDA,
    n: int = N,
) -> float:
    """
    Valuta un modello di linguaggio calcolando la perplessità sui dati di test.

    Args:
        g_lm: Modello linguistico globale, addestrato su più varietà di testi.
        d_lm: Modello linguistico di dominio, specializzato nel dominio di interesse.
        test_abs: Lista di blocchi anonimi (abs) di test.
        lambda_weight: Peso per l'interpolazione tra i due modelli. Default è LAMBDA.
        n: Dimensione degli n-grammi. Default è N.

    Returns:
        La perplessità del modello sui dati di test.

    Raises:
        ValueError: Se uno o entrambi i modelli non sono stati addestrati correttamente.
    """
    if not g_lm.vocab or not d_lm.vocab:
        raise ValueError(
            "Uno o entrambi i modelli non sono stati addestrati correttamente."
        )

    test_ngrams, _ = padded_everygram_pipeline(n, get_sentences(abs=test_abs))
    entropy = interpolated_entropy(g_lm, d_lm, test_ngrams, lambda_weight, n)
    return pow(2.0, entropy)


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
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        ngrams: Generatore di n-grammi dai dati di test.
        lambda_weight: Peso per l'interpolazione tra i due modelli.
        n: Dimensione degli n-grammi. Default è N.

    Returns:
        L'entropia interpolata calcolata.

    Raises:
        ValueError: Se non ci sono n-grammi validi nei dati di test.
    """
    total_log_prob = 0.0
    total_token_count = 0

    for sent in ngrams:
        for ngram in (g for g in sent if len(g) == n):
            total_log_prob += interpolated_log_score(
                g_lm, d_lm, ngram[-1], list(ngram[:-1]), lambda_weight
            )
            total_token_count += 1

    if total_token_count == 0:
        raise ValueError("Nessun n-gramma valido trovato.")

    return -total_log_prob / total_token_count