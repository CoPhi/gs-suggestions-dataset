"""
Modulo per il calcolo della top-K accuracy del modello a n-grammi
su insiemi di blocchi anonimi di test.
"""

from nltk.lm.models import LanguageModel
from tqdm import tqdm

from backend.config.settings import ALPHA, BATCH_SIZE, BETA, DELTA, K_PRED, LAMBDA, N
from backend.core.preprocess import clean_supplements
from models.ngrams.metrics import _LANGUAGE
from models.ngrams.metrics.utils import (
    check_supplement,
    get_beam_size,
    get_best_K_predictions_from_context,
    get_context_from_test_case,
)

def get_topK_accuracy(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    test_abs: list,
    lambda_weight: float = LAMBDA,
    batch_size: int = BATCH_SIZE,
    n: int = N,
    k_pred: int = K_PRED,
    alpha: float = ALPHA,
    beta: float = BETA,
    delta: float = DELTA,
) -> float:
    """
    Calcola la top-K accuracy del modello a n-grammi su un insieme di blocchi anonimi.

    Args:
        g_lm: Modello di linguaggio generico, addestrato su tutto il dataset.
        d_lm: Modello di linguaggio di dominio, addestrato su un dominio specifico.
        test_abs: Lista di blocchi anonimi di testo per il test.
        lambda_weight: Peso dell'interpolazione lineare. Default è LAMBDA.
        batch_size: Dimensione del batch per l'elaborazione. Default è BATCH_SIZE.
        n: Dimensione degli n-grammi. Default è N.
        k_pred: Numero di predizioni da generare per ogni contesto. Default è K_PRED.
        alpha: Peso del punteggio della loss. Default è ALPHA.
        beta: Peso del punteggio della edit distance. Default è BETA.
        delta: Peso del punteggio della length penalty. Default è DELTA.

    Returns:
        L'accuratezza del modello espressa come valore in [0, 1], arrotondato a 5 decimali.
    """
    correct_predictions = 0
    total_predictions = 0

    batches = range(0, len(test_abs), batch_size)
    for start in tqdm(batches, desc="Accuracy", unit="ab", leave=False):
        batch = test_abs[start : start + batch_size]

        for ab in batch:
            if ab["language"] != _LANGUAGE:
                continue

            correct, total = _evaluate_block(
                ab, g_lm, d_lm, lambda_weight, n, k_pred, alpha, beta, delta
            )
            correct_predictions += correct
            total_predictions += total

    return round(correct_predictions / total_predictions, 5)


def _evaluate_block(
    ab: dict,
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    lambda_weight: float,
    n: int,
    k_pred: int,
    alpha: float,
    beta: float,
    delta: float,
) -> tuple[int, int]:
    """
    Valuta un singolo blocco anonimo e restituisce il numero di predizioni
    corrette e totali.

    Args:
        ab: Blocco anonimo contenente training_text e test_cases.
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        lambda_weight: Peso dell'interpolazione lineare.
        n: Dimensione degli n-grammi.
        k_pred: Numero di predizioni top-K da generare.
        alpha: Peso del punteggio della loss.
        beta: Peso del punteggio della edit distance.
        delta: Peso del punteggio della length penalty.

    Returns:
        Tupla (correct, total) con i conteggi per il blocco.
    """
    supplements = clean_supplements(ab["training_text"])
    if not supplements:
        return 0, 0

    correct = 0
    total = 0

    for i, test_case_obj in enumerate(ab["test_cases"]):
        if i >= len(supplements):
            break

        suppl_seq, suppl_len_lacuna = supplements[i]

        if not check_supplement(suppl_seq):
            continue

        context, head_suppl, tail_suppl, _ = get_context_from_test_case(
            test_case_obj["test_case"], n
        )

        predictions = get_best_K_predictions_from_context(
            g_lm=g_lm,
            d_lm=d_lm,
            context=context,
            len_lacuna=suppl_len_lacuna,
            lambda_weight=lambda_weight,
            len_suppl_words=len(suppl_seq),
            suppl_words=suppl_seq,
            head_suppl=head_suppl,
            tail_suppl=tail_suppl,
            n=n,
            k_pred=k_pred,
            beam_size=get_beam_size(k_pred, 4),
            mod="acc",
            alpha=alpha,
            beta=beta,
            delta=delta,
        )

        if suppl_seq in predictions:
            correct += 1
        total += 1

    return correct, total
