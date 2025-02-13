import re
import os
import pandas as pd
from tqdm import tqdm

from cltk.core.data_types import Doc

from nltk.util import everygrams, pad_sequence
from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import (
    pad_both_ends,
    flatten,
)

from utils.preprocess import clean_text, clean_supplements
from models.training import get_sentences, load_lm
from config.settings import (
    tokenizer,
    sentence_tokenizer,
    K_PRED,
    BATCH_SIZE,
    N,
    LM_TYPE,
    TEST_SIZE,
    GAMMA,
)


def print_MLE_params_to_csv(params: dict) -> None:
    file_exists = os.path.isfile("MLE_results.csv")
    df = pd.DataFrame([params])
    df.to_csv("MLE_results.csv", mode="a", header=not file_exists, index=False)


def print_LIDSTONE_params_to_csv(params: dict) -> None:
    file_exists = os.path.isfile("LIDSTONE_results.csv")
    df = pd.DataFrame([params])
    df.to_csv("LIDSTONE_results.csv", mode="a", header=not file_exists, index=False)

def get_dist_words_context(lm: LanguageModel, context: str, n=N):
    """
    Genera la distribuzione di frequenze delle parole per un dato contesto.
    Se la distribuzione è vuota, riduce progressivamente il contesto fino a trovare una distribuzione non vuota. (Backoff)

    :param lm: Modello di linguaggio con il metodo `context_counts`
    :param context: Lista di token che rappresenta il contesto
    :param n: Dimensione del modello (numero di parole nel contesto)
    :return: Lista ordinata di tuple (parola, frequenza)
    """

    for i in range(n - 1, 0, -1):  # Riduci il contesto da n-1 fino a 1 parola
        dist_words_context = sorted(
            lm.context_counts(lm.vocab.lookup(context[(1 - i) :])).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        if dist_words_context:  # Se troviamo una distribuzione non vuota, restituiamola
            return dist_words_context
    return []  # Se nessun contesto ha una distribuzione valida, restituisci lista vuota


def get_predictions (lm : LanguageModel, context : list[str], suppl_words:list[str],  n=N, k_pred=K_PRED):
    """
    Genera previsioni di completamento del testo basate su un modello di linguaggio.

    Il metodo utilizza un contesto dato per ottenere una distribuzione di frequenza delle parole 
    e genera previsioni per completare il testo in base alle parole più probabili.

    :param lm: Istanza del modello di linguaggio (`LanguageModel`).
    :param context: Lista di tokens che rappresenta il contesto iniziale.
    :param suppl_words: lista di stringhe (tokens) di riferimento che rappresenta la gold label e la lunghezza desiderata del completamento.
    :param n: Dimensione del modello di n-grammi (default: `N`).
    :param k_pred: Numero massimo di previsioni da generare (default: `K_PRED`).

    :return: Lista di previsioni, dove ogni previsione è una lista di parole generate.
    
    Il metodo segue questi passi:
    1. Recupera la distribuzione delle parole in base al contesto usando `get_dist_words_context`.
    2. Seleziona le `k_pred` parole più probabili.
    3. Per ogni parola, estende il contesto e genera parole successive fino alla lunghezza di `suppl_words`.
    4. Restituisce una lista di liste, dove ogni lista rappresenta una sequenza generata.
    """
    predictions = [] #k predizioni
    dist_words_context = get_dist_words_context(lm, context, n)
    for word, _ in dist_words_context[:k_pred]:
        curr_context = context + [word]  # contesto adattato
        curr_pred = [word]
        while len(curr_pred) < len(suppl_words):
            token = lm.generate(
                text_seed=curr_context[(1 - n) :],
                num_words=1,
            )
            curr_pred.append(token)
            curr_context.append(token)

        predictions.append(curr_pred)
    
    return predictions
    


def perplexity(lm: LanguageModel, test_abs: list, n=N) -> float:
    """
    Valuta un modello di linguaggio calcolando la perplessità sui dati di test.

    Args:
        lm (LanguageModel): Il modello di linguaggio da valutare.
        test_abs (list): Una lista di abstract di test.
        n (int, opzionale): La dimensione degli n-grammi. Default è N.

    Returns:
        float: La perplessità del modello sui dati di test.

    Raises:
        ValueError: Se il modello non è stato addestrato correttamente.
        RuntimeError: Se si verifica un errore nel calcolo della perplessità.
    """

    test_ngrams = []
    for sentence in get_sentences(test_abs):
        test_ngrams.extend(
            list(
                everygrams(
                    sequence=pad_sequence(
                        sequence=sentence,
                        n=n,
                        pad_right=True,
                        right_pad_symbol="</s>",
                        pad_left=True,
                        left_pad_symbol="<s>",
                    ),
                    max_len=n,
                )
            )
        )

    if not lm.vocab:
        raise ValueError("Il modello non è stato addestrato correttamente.")
    try:
        return round(lm.perplexity(test_ngrams), 2)
    except ZeroDivisionError:
        raise RuntimeError(
            "Errore nel calcolo della perplessità. Verifica i dati di test e di addestramento."
        )


def accuracy(
    lm: LanguageModel, test_abs: list, batch_size=BATCH_SIZE, n=N, k_pred=K_PRED
) -> float:
    """
    Calcola l'accuratezza del modello di linguaggio su un dataset di test.

    Args:
        lm (LanguageModel): Il modello di linguaggio da valutare.
        test_abs (list): Lista di blocchi anonimi di testo per il test.
        batch_size (int, opzionale): La dimensione del batch per l'elaborazione. Default è `BATCH_SIZE`.
        n (int, opzionale): Dimensioni degli ngrammi del modello linguistico. Default è `N`.
        k_pred (int, opzionale): Il numero di predizioni da generare per ogni contesto. Default è `K_PRED`.

    Returns:
        float: L'accuratezza del modello espressa in percentuale.
    """
    correct_predictions = 0
    total_predictions = 0

    for start in tqdm(
        range(0, len(test_abs), batch_size),
        desc="Calcolo accuracy",
        unit="ab",
        leave=False,
    ):
        batch = test_abs[start : start + batch_size]  # batch di blocchi anonimi
        for ab in batch:
            if ab["language"] == "grc":
                supplements = clean_supplements(ab['training_text']) #supplements puliti
                if not supplements:
                    continue  # non ci sono blocchi da predire

                for i, obj in enumerate(ab["test_cases"]):
                    if i >= len(supplements):
                        break

                    test_case = obj["test_case"]

                    context = list(
                        flatten(
                            [
                                list(
                                    pad_both_ends(
                                        tokenizer.run(input_doc=Doc(raw=sent)).tokens,
                                        n=n,
                                    )
                                )
                                for sent in sentence_tokenizer.tokenize(
                                    clean_text(
                                        re.sub(r"[^\s]+\[", "[", test_case).split("[")[
                                            0
                                        ]
                                    )
                                )
                            ]
                        )
                    )[: (1 - n)] #prendo il contesto a sinistra della parentesi dal test_case

                    predictions = get_predictions(lm, context, supplements[i], n, k_pred)

                    if supplements[i] in predictions:
                        correct_predictions += 1

                    total_predictions += 1

    return round((correct_predictions / total_predictions) * 100, 2)


if __name__ == "__main__":
    lm, test_abs = load_lm()  # carico il modello
    acc = accuracy(lm, test_abs)
    if LM_TYPE == "LIDSTONE":
        pp = perplexity(lm, test_abs)
        print("Perplexity: ", pp)
        print("Accuracy: ", acc)
        print_LIDSTONE_params_to_csv(
            {
                "K_PRED": K_PRED,
                "TEST_SIZE": TEST_SIZE,
                "DIMENSION": N,
                "BATCH_SIZE": BATCH_SIZE,
                "GAMMA": GAMMA,
                "ACCURACY": acc,
                "PERPLEXITY": pp,
            }
        )

    if LM_TYPE == "MLE":
        print("Accuracy: ", acc)
        print_MLE_params_to_csv(
            {
                "K_PRED": K_PRED,
                "DIMENSION": N,
                "BATCH_SIZE": BATCH_SIZE,
                "TEST_SIZE": TEST_SIZE,
                "ACCURACY": acc,
            }
        )
