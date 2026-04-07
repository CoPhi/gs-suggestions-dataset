from nltk.lm.models import LanguageModel

from models.ngrams.metrics.utils import (
    get_beam_size,
    get_context_from_test_case,
    get_best_K_predictions_from_context,
    nll_score,
)
from backend.config.settings import ALPHA, BETA, DELTA, N, K_PRED, LM_TYPE, LAMBDA

def generate_k_suggests(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    context: str,
    num_tokens: int,
    lm_type: str = LM_TYPE,
    n: int = N,
    k_pred: int = K_PRED,
    lambda_weight: float = LAMBDA,
) -> list[str]:
    """
    Genera k-predizioni utilizzando un modello di linguaggio.

    Args:
        lm (LanguageModel): Il modello di linguaggio da utilizzare per generare le parole.
        context (str): Il contesto testuale da utilizzare come seme per la generazione.
        num_words (int): Il numero di parole da generare.
        n (int, opzionale): Dimensione degli ngrammi del modello, default N (settings.py).
        k_pred (int, opzionale): Il numero di parole da predire, default K_PRED (settings.py).

    Returns:
        List[list[str]]: Una lista di parole generate dal modello di linguaggio.

    Raises:
        ValueError: Se il modello di linguaggio non è stato caricato correttamente.
    """
    if not (g_lm and d_lm):
        raise ValueError("Il modelli non sono stati caricati correttamente.")

    seq, head, tail, len_lacuna = get_context_from_test_case(
        context, n=n, case_folding=True
    )

    predictions = get_best_K_predictions_from_context(
        g_lm=g_lm,
        d_lm=d_lm,
        context=seq,
        len_lacuna=len_lacuna,
        lambda_weight=lambda_weight,
        lm_type=lm_type,
        len_suppl_words=num_tokens,
        head_suppl=head,
        tail_suppl=tail,
        n=n,
        k_pred=k_pred,
        beam_size=get_beam_size(k_pred, 4),
        mod="infer",
        alpha=ALPHA,
        beta=BETA,
        delta=DELTA,
    )

    return [
        (
            " ".join(pred).lower(),
            pow(
                2,
                -nll_score(
                    g_lm,
                    d_lm,
                    lambda_weight,
                    g_lm.vocab.lookup(seq[(1 - n) :]),
                    pred,
                    lm_type,
                ),
            ),
        )
        for pred in predictions
    ]
