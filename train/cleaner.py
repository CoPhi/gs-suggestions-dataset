import json
import pickle
from nltk.lm.api import LanguageModel
from sklearn.model_selection import train_test_split
from train import sentence_tokenizer
from tqdm import tqdm
from config.settings import CORPUS_NAMES, DATA_PATH, TEST_SIZE, TEST_SIZES, LM_TYPE, N

from utils.preprocess import (
    clean_text_from_gaps,
    get_tokens_from_clean_text,
    remove_punctuation,
)


def load_abs(corpus_set: set = None) -> list:
    """
    Carica e restituisce una lista di anonymous block (ab) dai file JSON presenti nel percorso specificato.
    Il percorso dei file JSON è determinato dalla variabile globale DATA_PATH.
    Ogni file JSON viene aperto e il suo contenuto viene aggiunto alla lista degli anonymous blocks.

    Args:
        corpus_set (set, optional): Un insieme di ID di corpus. Se fornito, solo gli anonymous block
                                    appartenenti a questi ID di corpus saranno inclusi nella lista risultante.
                                    Se non fornito, tutti gli anonymous block con lingua greca ("grc") saranno inclusi.

    Returns:
        list: Una lista contenente gli anonymous block (abs) caricati dai file JSON.
    """

    if corpus_set is not None:
        for corpus_id in corpus_set:
            if corpus_id not in CORPUS_NAMES:
                raise ValueError(
                    f"Corpus ID {corpus_id} not found in available corpus names"
                )

    dataset = []
    for file_path in tqdm(
        list(DATA_PATH.glob("*.json")),
        desc="Processing MAAT corpus",
        unit="file",
        leave=False,
    ):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                abs = json.load(f)
                if corpus_set is None:
                    dataset.extend([ab for ab in abs if ab["language"] == "grc"])
                else:
                    dataset.extend(
                        [
                            ab
                            for ab in abs
                            if ab["language"] == "grc" and ab["corpus_id"] in corpus_set
                        ]
                    )
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {file_path}: {e}")

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

    if test_size not in TEST_SIZES:
        raise ValueError(f"Test size {test_size} not found in available test sizes")

    train_abs, test_abs = train_test_split(abs, test_size=test_size)
    return train_abs, test_abs


def get_sentences(abs: list, remove_punct: bool = True) -> list:
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
                    if remove_punct:
                        sentences.append(
                            get_tokens_from_clean_text(remove_punctuation(sent))
                        )
                    else:
                        sentences.append(get_tokens_from_clean_text(sent))

    return sentences


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
