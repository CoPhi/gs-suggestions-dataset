from collections import Counter
import json
import pickle
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from cltk.core.data_types import Doc
from nltk.lm.models import MLE, Lidstone, LanguageModel
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import padded_everygram_pipeline

from config.settings import (
    DATA_PATH,
    TEST_SIZE,
    LM_TYPE,
    N,
    GAMMA,
    sentence_tokenizer,
)
from utils.preprocess import clean_text_from_gaps, get_tokens_from_clean_text, remove_punctuation


def load_abs() -> list:
    """
    Carica e restituisce una lista di anonymous block (ab) dai file JSON presenti nel percorso specificato.
    Il percorso dei file JSON è determinato dalla variabile globale DATA_PATH.
    Ogni file JSON viene aperto e il suo contenuto viene aggiunto alla lista degli anonymous blocks.
    Returns:
        list: Una lista contenente gli anonymous block (abs) caricati dai file JSON.
    """
    dataset = []
    for file_path in tqdm(
        list(DATA_PATH.glob("*.json")),
        desc="Processing MAAT corpus",
        unit="file",
        leave=False,
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            abs = json.load(f)
            dataset.extend([ab for ab in abs if ab["language"] == "grc"]) #prendo i blocchi anonimi con lingua greca

    return dataset


def split_abs(abs: list, test_size=TEST_SIZE) -> tuple:
    """
    Divide una lista di anonymous block in due sottoinsiemi per l'addestramento e il test.
    Args:
        abs (list): Lista di anonymous block da dividere.
        test_size (float, opzionale): Dimensione del test set
    Returns:
        tuple: Una tupla contenente due liste, la prima per l'addestramento e la seconda per il test.
    """

    train_abs, test_abs = train_test_split(abs, test_size=test_size)
    return train_abs, test_abs


def get_sentences(abs: list) -> list:
    """
    Estrae e processa le frasi di addestramento da una lista di blocchi anonimi fornita.
    Questo metodo filtra e processa il 'training_text' da ciascun oggetto nella lista di input 'ab'.
    Include solo i testi in cui la 'language' è 'grc'.
    Le frasi vengono tokenizzate e aggiunte alla lista di frasi.
    Args:
        ab (list): Una lista di oggetti, ciascuno contenente le chiavi 'training_text' e 'language'.
    Returns:
        list: Una lista di frasi di addestramento tokenizzate.
    """
    sentences = []
    for obj in tqdm(abs, desc="Processing anonymous blocks", unit="ab", leave=False):
        if obj["training_text"] and obj["language"] == "grc":
            for sent in sentence_tokenizer.tokenize(
                text=clean_text_from_gaps(obj["training_text"])
            ):
                if sent:
                    sentences.append(
                        get_tokens_from_clean_text(
                            remove_punctuation(sent)
                            )
                    )
    return sentences


def train_lm(
    train_abs: list, lm_type=LM_TYPE, min_freq=2, gamma=GAMMA, n=N
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
    if lm_type == "MLE":
        lm = MLE(n)
    else:
        lm = Lidstone(gamma, n)

    train_ngrams, vocab_tokens = padded_everygram_pipeline(
        order=n, text=get_sentences(train_abs)
    )

    token_counts = Counter(vocab_tokens)

    lm.fit(
        train_ngrams,
        [token for token, freq in token_counts.items() if freq >= min_freq],
    )
    
    #Rilascio la oggetti che non mi servono più per liberare memoria
    del train_abs, train_ngrams, vocab_tokens, token_counts
    gc.collect()
    
    return lm


def pipeline_train(lm_type=LM_TYPE, gamma=GAMMA, n=N, test_size=TEST_SIZE):
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
    train_abs, test_abs = split_abs(abs=load_abs(), test_size=test_size)
    lm = train_lm(train_abs, lm_type=lm_type, gamma=gamma, n=n)
    save_lm(lm=lm, test_abs=test_abs)
    return lm, test_abs


def save_lm(lm: LanguageModel, test_abs: list, n=N, lm_type=LM_TYPE) -> None:
    """
    Salva un modello di linguaggio (LanguageModel) su disco come file pickle.
    Args:
        lm (LanguageModel): Il modello di linguaggio da salvare.
        test_abs (list): Una lista di anonymous blocks di test.
        n (int, opzionale): Dimensione degli ngrammi del modello. Default è N.
        lm_type (str, opzionale): Il tipo di modello linguistico da salvare su disco. Default è `LM_TYPE`.
    Returns:
        None: Questa funzione non ritorna nulla.
    """

    model_data = {"lm": lm, "test_ab": test_abs}
    with open(f"{lm_type}_{n}.pkl", "wb") as f:
        pickle.dump(model_data, f)
        print("Language model saved.")


def load_lm(n=N, lm_type=LM_TYPE) -> None:
    """
    Carica un modello linguistico da un file pickle.
    Args:
        n (int, optional): Dimensione degli ngrammi del modello. Default è `N`.
        lm_type (str, optional): Il tipo di modello linguistico da caricare. Default è `LM_TYPE`.
    Returns:
        tuple: Una tupla contenente il modello linguistico (lm) e i dati test_ab se il caricamento ha successo, altrimenti `None`.
    Raises:
        FileNotFoundError: Se il file pickle specificato non esiste.
        pickle.UnpicklingError: Se c'è un errore durante l'unpickling del file.
    """

    with open(f"{lm_type}_{n}.pkl", "rb") as f:
        model_data = pickle.load(f)
        lm = model_data["lm"]
        test_ab = model_data["test_ab"]
        print("Language model loaded.")
        return lm, test_ab

    return None


if __name__ == "__main__":
    pipeline_train()
