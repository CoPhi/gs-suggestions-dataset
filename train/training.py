import gc
from typing import Optional

from nltk.lm.models import MLE, Lidstone, LanguageModel
from nltk.lm.preprocessing import padded_everygram_pipeline
from train import get_sentences, load_test_abs, load_abs, save_lm, load_specific_domain_abs

from config.settings import (
    TEST_SIZE,
    LM_TYPE,
    N,
    GAMMA,
    MIN_FREQ,
    DIMENSIONS,
    LM_TYPES,
    MIN_FREQS,
)


def train_lm(
    train_abs: list,
    lm_type=LM_TYPE,
    min_freq: Optional[int] = MIN_FREQ,
    gamma: Optional[float] = GAMMA,
    n=N,
) -> tuple[LanguageModel, LanguageModel]:
    """
    Addestra un modello di linguaggio sulle frasi di addestramento fornite.

    Args:
        train_abs (list): Lista di blocchi anonimi di addestramento.
        lm_type (str, opzionale): Tipo di modello di linguaggio da addestrare ('MLE' o altro). Default è LM_TYPE.
        min_freq (int, opzionale): filtro per la minima frequenza per gli elementi nel vocabolario del modello, se un token ha una frequenza assoluta minore viene contata come sconosciuta (`UNK`)
        gamma (float, opzionale): Parametro di smoothing per il modello Lidstone. Default è GAMMA.
        n (int, opzionale): Ordine degli n-grammi. Default è N.

    Returns:
        tupla di modelli linguistici, di cui uno globale e uno specifico di dominio
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

        g_lm = MLE(n)  # modello di dominio generale
        d_lm = MLE(n)  # modello di dominio specifico
    else:
        if gamma is None:
            raise ValueError("Unvalid gamma for Lidstone smoothing")

        g_lm = Lidstone(gamma, n)  # modello di dominio generale
        d_lm = Lidstone(gamma, n)  # modello di dominio specifico

    g_train_ngrams, g_vocab_tokens = padded_everygram_pipeline(
        order=n, text=get_sentences(abs=train_abs)
    )

    g_lm.fit(g_train_ngrams, g_vocab_tokens)

    d_train_ngrams, d_vocab_tokens = padded_everygram_pipeline(
        order=n, text=get_sentences(abs=load_specific_domain_abs(abs=train_abs))
    )

    d_lm.fit(d_train_ngrams, d_vocab_tokens)
    return g_lm, d_lm


def pipeline_train(
    lm_type=LM_TYPE,
    gamma: Optional[float] = GAMMA,
    min_freq: Optional[int] = MIN_FREQ,
    n=N,
    corpus_set=None,
    budget: Optional[int] = None,
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
    g_lm, d_lm = train_lm(
        train_abs=load_abs(corpus_set, budget),
        lm_type=lm_type,
        min_freq=min_freq,
        gamma=gamma,
        n=n,
    )
    
    test_abs = load_test_abs()
    return g_lm, d_lm, test_abs


if __name__ == "__main__":
    g_lm, d_lm, test_abs = pipeline_train()
    save_lm(
        lm=g_lm, test_abs=test_abs, checkpoint="General_model"
    )  # Salvataggio modello generale
    save_lm(
        lm=d_lm, test_abs=list(), checkpoint="Domain_model"
    )  # Salvataggio modello specifico di dominio
