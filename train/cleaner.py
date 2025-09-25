import json
import pickle
import random
from typing import Optional
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
from nltk.lm.preprocessing import flatten

def check_ab(ab: dict, corpus_set: Optional[list[str]] = None) -> bool:
    if ab.get("language") != "grc":
        return False

    corpus_id = ab.get("corpus_id")
    if corpus_set is None or corpus_id is None:
        return True  # accetta qualsiasi corpus_id

    return corpus_id in corpus_set or corpus_id == "unknown"


def load_abs(
    corpus_set: Optional[list[str]] = None, budget: Optional[int] = None
) -> list:
    """
    Carica e restituisce una lista di anonymous block (ab) dai file JSON presenti nel percorso specificato.
    Il percorso dei file JSON è determinato dalla variabile globale DATA_PATH.
    Ogni file JSON viene aperto e il suo contenuto viene aggiunto alla lista degli anonymous blocks.

    Args:
        corpus_set (set, optional): Un insieme di ID di corpus. Se fornito, solo gli anonymous block
                                    appartenenti a questi ID di corpus saranno inclusi nella lista risultante.
                                    Se non fornito, tutti gli anonymous block con lingua greca ("grc") saranno inclusi.
        budget (int, optional): Un valore percentuale che determina la dimensione del sottoinsieme da restituire.
                                    Se fornito, la funzione restituirà un sottoinsieme casuale della lista
                                    di anonymous block, limitato alla percentuale specificata.
                                    Se non fornito, la funzione restituirà l'intera lista di anonymous block.
    Raises:
        ValueError: Se un ID di corpus fornito non è presente nell'insieme CORPUS_NAMES.
        FileNotFoundError: Se si verifica un errore durante la lettura dei file JSON.
        json.JSONDecodeError: Se si verifica un errore durante il caricamento del contenuto JSON.

    Returns:
        list: Una lista contenente gli anonymous block (abs) caricati dai file JSON.
    """

    # Validare i corpus dentro `corpus_set`
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

        if "test_abs" in str(file_path):  # Skip test files
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                abs = json.load(f)
                dataset.extend([ab for ab in abs if check_ab(ab, corpus_set)])
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {file_path}: {e}")

    if budget is not None:
        subset_size = int(len(dataset) * (budget / 100))
        return random.sample(dataset, subset_size)

    return dataset


def load_specific_domain_abs(abs: list, domain_title: str = "P.Herc.") -> list:
    """
    Restituisce un sottoinsieme dei blocchi anonimi il cui dominio è rappresentato dal titolo immesso in `domain_title`
    """

    def title_matches(title_field):
        if isinstance(title_field, str):
            if domain_title:
                return domain_title in title_field
        return False

    return [ab for ab in abs if title_matches(ab["title"]) and ab["language"] == "grc"]


def load_test_abs() -> list:
    """
    Carica i blocchi anonimi di test da un file JSON.
    Returns:
        list: Una lista di dizionari contenenti i blocchi anonimi di test.
    """
    with open(next(DATA_PATH.glob("test_abs.json"), None), "r", encoding="utf-8") as f:
        return json.load(f)


def split_abs(abs: list, test_size=TEST_SIZE) -> tuple:
    """
    Divide una lista di anonymous block in due sottoinsiemi per l'addestramento e il test.
    Args:
        abs (list): Lista di anonymous block da dividere.
        test_size (float, opzionale): Dimensione del test set
    Returns:
        tuple[list, list]: Una tupla contenente due liste, la prima per l'addestramento e la seconda per il test.
    """

    if test_size not in TEST_SIZES:
        raise ValueError(f"Test size {test_size} not found in available test sizes")

    train_abs, test_abs = train_test_split(abs, test_size=test_size)
    return train_abs, test_abs


def get_sentences(
    abs: list, remove_punct: bool = True, case_folding: bool = True
) -> list[list[str]]:
    """
    Estrae e processa le frasi di addestramento da una lista di blocchi anonimi fornita.
    Questo metodo filtra e processa il 'training_text' da ciascun oggetto nella lista di input 'abs'.
    Include solo i testi in cui la 'language' è 'grc'.
    Le frasi vengono tokenizzate e aggiunte alla lista di frasi.
    Args:
        ab (list): Una lista di oggetti, ciascuno contenente le chiavi 'training_text' e 'language'.
        remove_punct (bool, opzionale): Se `True`, rimuove la punteggiatura dalle frasi. Default è `True`.
        case_folding (bool, opzionale): Se `True`, applica il case folding al testo. Default è `True`.
    Raises:
        ValueError: Se il testo di addestramento è vuoto o se la lingua non è 'grc'.
    Returns:
        list: Una lista di frasi di addestramento tokenizzate.
    """
    sentences = []
    for obj in tqdm(abs, desc="Processing anonymous blocks", unit="ab", leave=False):
        if obj["training_text"] and obj["language"] == "grc":
            for sent in sentence_tokenizer.tokenize(
                text=clean_text_from_gaps(
                    obj["training_text"], case_folding=case_folding
                )
            ):
                if sent:
                    if remove_punct:
                        sentences.append(
                            get_tokens_from_clean_text(remove_punctuation(sent))
                        )
                    else:
                        sentences.append(get_tokens_from_clean_text(sent))
    return sentences


def save_lm(lm: LanguageModel, checkpoint: str, dev_abs: list, n=N) -> None:
    """
    Salva un modello di linguaggio (`LanguageModel`) su disco come file pickle.
    Args:
        lm (LanguageModel): Il modello di linguaggio da salvare.
        dev_abs (list): Una lista di anonymous blocks di dev.
        n (int, opzionale): Dimensione degli ngrammi del modello. Default è N.
        checkpoint (str): Checkpoint del modello linguistico da salvare su disco.
    Returns:
        None: Questa funzione non ritorna nulla.
    """

    model_data = {"lm": lm, "dev_abs": dev_abs}
    with open(f"{checkpoint}_{n}.pkl", "wb") as f:
        pickle.dump(model_data, f)
        print("Language model saved.")


def load_lm(checkpoint: str, n=N) -> tuple[LanguageModel, list]:
    """
    Carica un modello linguistico da un file pickle.
    Args:
        n (int, optional): Dimensione degli ngrammi del modello. Default è `N`.
        checkpoint (str, optional):Checkpoint del modello linguistico da caricare.
    Returns:
        tuple[LanguageModel, list]: Una tupla contenente il modello linguistico (lm) e i dati `dev_abs` se il caricamento ha successo, altrimenti `None`.
    Raises:
        FileNotFoundError: Se il file pickle specificato non esiste.
        pickle.UnpicklingError: Se c'è un errore durante l'unpickling del file.
    """

    with open(f"{checkpoint}_{n}.pkl", "rb") as f:
        model_data = pickle.load(f)
        lm = model_data["lm"]
        dev_abs = model_data["dev_abs"]
        print("Language model loaded.")
        return lm, dev_abs


# if __name__ == "__main__": 
    #Voglio stampare il numero di blocchi anonimi di cui sono composti i vari set
    # train_abs = load_abs()
    # domain_abs = load_specific_domain_abs(abs=train_abs)
    # train_domain_abs, dev_domain_abs = split_abs(domain_abs, test_size=0.2)
    # test_abs = load_test_abs()
    # print(f"Total anonymous blocks loaded (training set): {len(train_abs) - len(dev_domain_abs)}")
    # print(f"Total anonymous blocks loaded (dev set): {len(dev_domain_abs)}")
    # print(f"Total anonymous blocks loaded (test set): {len(test_abs)}")

    # print (f"Numero di token in training set: {len(list(flatten(get_sentences([ab for ab in train_abs if ab not in dev_domain_abs]))))}")
    # print (f"Numero di token in dev set: {len(list(flatten(get_sentences(domain_abs))))}")
    # print (f"Numero di token in test set: {len(list(flatten(get_sentences(test_abs))))}")
    
    # print (f"Numero di frasi in training set: {len(get_sentences([ab for ab in train_abs if ab not in dev_domain_abs]))}")
    # print (f"Numero di frasi in dev set: {len(get_sentences(domain_abs))}")
    # print (f"Numero di frasi in test set: {len(get_sentences(test_abs))}")