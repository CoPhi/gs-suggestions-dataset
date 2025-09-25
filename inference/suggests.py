import argparse
from math import exp

from nltk.lm.models import LanguageModel

from predictions.ngrams import (
    get_beam_size,
    get_context_from_test_case,
    get_best_K_predictions_from_context,
    nll_score,
)
from config.settings import ALPHA, BETA, DELTA, N, K_PRED, LM_TYPE, LAMBDA
from train import load_lm


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


if __name__ == "__main__":
    g_lm, _ = load_lm("General_model")
    d_lm, _ = load_lm("Domain_model")

    parser = argparse.ArgumentParser(description="Infer using the trigram model.")

    parser.add_argument(
        "--context", type=str, help="Context for word generation (for infer mode)"
    )
    parser.add_argument(
        "--num_words", type=int, help="Number of words to generate (for infer mode)"
    )

    args = parser.parse_args()  # Analizza gli argomenti passati
    words = generate_k_suggests(g_lm, d_lm, args.context, args.num_words)
    print(words)
