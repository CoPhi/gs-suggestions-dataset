import argparse
from math import exp

from nltk.lm.models import LanguageModel

from metrics.accuracy import get_context_from_test_case, get_best_K_predictions_from_context, loss
from config.settings import N, K_PRED, LM_TYPE
from train import load_lm


def generate_k_suggests(
    lm: LanguageModel,
    context: str,
    num_tokens: int,
    lm_type: str = LM_TYPE,
    n: int = N,
    k_pred: int = K_PRED,
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
    if not lm:
        raise ValueError("Il modello non è stato caricato correttamente.")


    context, head, tail = get_context_from_test_case(context, n=n, case_folding=True)

    predictions = get_best_K_predictions_from_context(
            lm=lm,
            context=context,
            lm_type=lm_type,
            len_suppl_words=num_tokens, 
            head_suppl=head,
            tail_suppl=tail,
            n=n,
            k_pred=k_pred,
            mod="infer",
            alpha=1,
            beta=0,
        )
    
    return [
        (" ".join(pred).lower(),  exp(- loss(lm, lm.vocab.lookup(context[(1 - n):]), pred, lm_type)))
        for pred in predictions 
    ]


if __name__ == "__main__":
    lm, _ = load_lm()

    parser = argparse.ArgumentParser(description="Infer using the trigram model.")

    parser.add_argument(
        "--context", type=str, help="Context for word generation (for infer mode)"
    )
    parser.add_argument(
        "--num_words", type=int, help="Number of words to generate (for infer mode)"
    )

    args = parser.parse_args()  # Analizza gli argomenti passati
    words = generate_k_suggests(lm, args.context, args.num_words)
    print(words)
