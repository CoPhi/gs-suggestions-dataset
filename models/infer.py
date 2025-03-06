import argparse

from nltk.lm.models import LanguageModel

from models.evaluate import get_context, get_K_predictions
from config.settings import tokenizer, N, K_PRED
from models.training import load_lm

def generate_k_suggests(lm: LanguageModel, context: str, num_words: int, n=N, k_pred=K_PRED) -> list[str]:
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

    return [" ".join(pred).lower() for pred in get_K_predictions(lm, get_context(context, n=n), num_words, n, k_pred)]


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
    print (words)
