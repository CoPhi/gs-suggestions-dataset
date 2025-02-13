import argparse

from cltk.core.data_types import Doc
from nltk.lm.models import LanguageModel

from config.settings import tokenizer, N
from utils.preprocess import clean_text
from models.training import load_lm


def generate_words(lm: LanguageModel, context: str, num_words: int, n=N):
    """
    Genera una sequenza di parole utilizzando un modello di linguaggio.

    Args:
        lm (LanguageModel): Il modello di linguaggio da utilizzare per generare le parole.
        context (str): Il contesto testuale da utilizzare come seme per la generazione.
        num_words (int): Il numero di parole da generare.
        n (int, opzionale): Dimensione degli ngrammi del modello, default N (settings.py).

    Returns:
        List[str]: Una lista di parole generate dal modello di linguaggio.

    Raises:
        ValueError: Se il modello di linguaggio non è stato caricato correttamente.
    """
    if not lm:
        raise ValueError("Il modello non è stato caricato correttamente.")

    return lm.generate(
        num_words=num_words,
        text_seed=tokenizer.run(input_doc=Doc(raw=clean_text(context))).tokens[
            (1-n):
        ],
    )


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

    words = generate_words(lm, args.context, args.num_words)
    print (" ".join(words))
