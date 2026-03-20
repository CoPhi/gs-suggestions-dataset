from typing import Optional

from nltk.lm.models import MLE, Lidstone, LanguageModel
from nltk.lm.preprocessing import padded_everygram_pipeline
from train import (
    get_sentences,
    load_abs,
    save_lm,
    load_specific_domain_abs,
)

from config.settings import (
    LM_TYPE,
    N,
    GAMMA,
    MIN_FREQ,
    DIMENSIONS,
    LM_TYPES,
    MIN_FREQS,
)
from train.cleaner import split_abs


def train_lm(
    train_abs: list,
    lm_type=LM_TYPE,
    min_freq: Optional[int] = MIN_FREQ,
    gamma: Optional[float] = GAMMA,
    n=N,
) -> tuple[LanguageModel, LanguageModel]:
    """
    Addestra un modello di linguaggio n-gram sui dati di addestramento forniti.

        train_abs (list): Lista di frasi o blocchi di testo anonimi utilizzati per l'addestramento del modello.
        lm_type (str, opzionale): Tipo di modello di linguaggio da addestrare ('MLE' per Maximum Likelihood Estimation o altro). Il valore predefinito è LM_TYPE.
        min_freq (int, opzionale): Frequenza minima richiesta affinché un token venga incluso nel vocabolario del modello; i token con frequenza inferiore saranno trattati come sconosciuti (`UNK`). Il valore predefinito è MIN_FREQ.
        gamma (float, opzionale): Parametro di smoothing per il modello Lidstone. Necessario solo se si utilizza Lidstone. Il valore predefinito è GAMMA.
        n (int, opzionale): Ordine degli n-grammi da utilizzare per il modello. Il valore predefinito è N.

        tuple[LanguageModel, LanguageModel]: Una tupla contenente due modelli linguistici: uno globale e uno specifico di dominio.

    Raises:
        ValueError: Se il tipo di modello di linguaggio, la dimensione degli n-grammi o la frequenza minima non sono tra quelli disponibili, oppure se il parametro gamma non è valido per Lidstone.
    """

    if lm_type not in LM_TYPES:
        raise ValueError(
            f"Language model type {lm_type} not found in available language model types"
        )

    if n not in DIMENSIONS:
        raise ValueError(f"Dimension {n} not found in available dimensions")

    if min_freq not in MIN_FREQS:
        raise ValueError(
            f"frequence {min_freq} not found in available frequencies for training"
        )

    if lm_type == "MLE":
        lm = MLE(n)

    else:
        if gamma is None:
            raise ValueError("Unvalid gamma for Lidstone smoothing")

        lm = Lidstone(gamma, n) 

    train_ngrams, vocab_tokens = padded_everygram_pipeline(
        order=n, text=get_sentences(abs=train_abs)
    )

    lm.fit(train_ngrams, vocab_tokens)
    return lm


def pipeline_train(
    lm_type=LM_TYPE,
    gamma: Optional[float] = GAMMA,
    min_freq: Optional[int] = MIN_FREQ,
    n=N,
    corpus_set=None,
    budget: Optional[int] = None,
) -> tuple:
    """
    Esegue il processo di addestramento del modello linguistico.
    Questa funzione segue i seguenti passaggi:
    1. Carica i dati dagli anonymous blocks specificati.
    2. Filtra i dati per dominio specifico.
    3. Suddivide i dati di dominio in set di addestramento e validazione (dev).
    4. Addestra due modelli linguistici:
        - Un modello generale (g_lm) sui dati esclusi dal set di validazione di dominio.
        - Un modello di dominio (d_lm) sui dati di addestramento di dominio.
    5. Restituisce entrambi i modelli addestrati e il set di validazione di dominio.
    Args:
         lm_type (str, opzionale): Tipo di modello linguistico da addestrare.
         gamma (float, opzionale): Parametro di smoothing per il modello linguistico.
         min_freq (int, opzionale): Frequenza minima per includere un token nel vocabolario.
         n (int): Ordine del modello n-gram.
         corpus_set (iterable, opzionale): Insieme dei dati di partenza.
         budget (int, opzionale): Numero massimo di elementi da utilizzare.
         tuple: Una tupla contenente il modello linguistico generale (g_lm),
                  il modello linguistico di dominio (d_lm) e il set di validazione di dominio (dev_domain_abs).
    """

    train_abs = load_abs(corpus_set, budget)
    domain_abs = load_specific_domain_abs(abs=train_abs)
    train_domain_abs, dev_domain_abs = split_abs(domain_abs, test_size=0.2)

    g_lm = train_lm(
        train_abs=[ab for ab in train_abs if ab not in dev_domain_abs],
        lm_type=lm_type,
        min_freq=min_freq,
        gamma=gamma,
        n=n,
    )

    d_lm = train_lm(
        train_abs=train_domain_abs,
        lm_type=lm_type,
        min_freq=min_freq,
        gamma=gamma,
        n=n,
    )

    return g_lm, d_lm, dev_domain_abs


if __name__ == "__main__":
    g_lm, d_lm, dev_abs = pipeline_train()
    save_lm(
        lm=g_lm, dev_abs=dev_abs, checkpoint="General_model"
    )  # Salvataggio modello generale
    save_lm(
        lm=d_lm, dev_abs=list(), checkpoint="Domain_model"
    )  # Salvataggio modello specifico di dominio
