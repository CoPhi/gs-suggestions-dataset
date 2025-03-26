from collections import Counter

import pickle
import gc
from typing import Optional

from nltk.lm.models import MLE, Lidstone, LanguageModel
from nltk.lm.preprocessing import padded_everygram_pipeline

from train import get_sentences, split_abs, load_abs, save_lm, load_lm


from config.settings import (
    TEST_SIZE,
    LM_TYPE,
    N,
    GAMMA,
    DIMENSIONS,
    LM_TYPES,
    GAMMAS,
)


def train_lm(
    train_abs: list, lm_type=LM_TYPE, min_freq=3, gamma: Optional[float] = GAMMA, n=N
) -> LanguageModel:
    """
    Addestra un modello di linguaggio sulle frasi di addestramento fornite.

    Args:
        train_abs (list): Lista di blocchi anonimi di addestramento.
        lm_type (str, opzionale): Tipo di modello di linguaggio da addestrare ('MLE' o altro). Default è LM_TYPE.
        min_freq (int, opzionale): filtro per la minima frequenza per gli elementi nel vocabolario del modello, se un token ha una frequenza assoluta minore viene contata come sconosciuta (`UNK`)
        gamma (float, opzionale): Parametro di smoothing per il modello Lidstone. Default è GAMMA.
        n (int, opzionale): Ordine degli n-grammi. Default è N.

    Returns:
        LanguageModel: Modello di linguaggio addestrato.
    """

    if lm_type not in LM_TYPES:
        raise ValueError(
            f"Language model type {lm_type} not found in available language model types"
        )

    if n not in DIMENSIONS:
        raise ValueError(f"Dimension {n} not found in available dimensions")

    if lm_type == "MLE":

        lm = MLE(n)
    else:
        if gamma is None or gamma not in GAMMAS:
            raise ValueError("Unvalid gamma for Lidstone smoothing")

        lm = Lidstone(gamma, n)

    train_ngrams, vocab_tokens = padded_everygram_pipeline(
        order=n, text=get_sentences(train_abs)
    )

    token_counts = Counter(vocab_tokens)

    lm.fit(
        train_ngrams,
        [token for token, freq in token_counts.items() if freq >= min_freq],
    )

    gc.collect()

    return lm


def pipeline_train(
    lm_type=LM_TYPE,
    gamma: Optional[float] = GAMMA,
    n=N,
    test_size=TEST_SIZE,
    corpus_set=None,
) -> tuple:
    """
    Esegue il processo di addestramento del modello.

    La funzione esegue i seguenti passaggi:
    1. Carica i dati dalla lista degli anonymous blocks.
    2. Divide i dati in set di addestramento e di test.
    3. Addestra il modello di linguaggio (LM) utilizzando il set di addestramento.
    4. Salva il modello addestrato e il set di test.

    Returns:
        tuple: Una tupla contenente il modello linguistico (lm) ed il test set
    """
    train_abs, test_abs = split_abs(abs=load_abs(corpus_set), test_size=test_size)
    lm = train_lm(train_abs, lm_type=lm_type, gamma=gamma, n=n)
    return lm, test_abs


if __name__ == "__main__":
    lm, test_abs = pipeline_train()
    save_lm(lm=lm, test_abs=test_abs)
