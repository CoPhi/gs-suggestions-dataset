import json
import pickle
import random
from typing import Optional
from nltk.lm.api import LanguageModel
from sklearn.model_selection import train_test_split
from models.ngrams.train import sentence_tokenizer
from tqdm import tqdm
from backend.config.settings import (
    CORPUS_NAMES,
    DATA_PATH,
    TEST_SIZE,
    TEST_SIZES,
    N,
    RANDOM_SEED,
)
from backend.core.preprocess import (
    clean_text_from_gaps,
    get_tokens_from_clean_text,
    remove_punctuation,
)


def check_ab(ab: dict, corpus_set: Optional[list[str]] = None) -> bool:
    if ab.get("language") != "grc":
        return False

    corpus_id = ab.get("corpus_id")
    if corpus_set is None or corpus_id is None:
        return True

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

    corpus_set_fast = None
    if corpus_set is not None:
        corpus_set_fast = set(corpus_set)
        for corpus_id in corpus_set_fast:
            if corpus_id not in CORPUS_NAMES:
                raise ValueError(
                    f"Corpus ID {corpus_id} not found in available corpus names"
                )

    dataset = []

    json_files = [f for f in DATA_PATH.glob("*.json") if f.name != "test_abs.json"]

    for file_path in tqdm(
        json_files,
        desc="Processing MAAT corpus",
        unit="file",
        leave=False,
    ):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                abs_data = json.load(f)
                dataset.extend(ab for ab in abs_data if check_ab(ab, corpus_set_fast))
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading {file_path}: {e}")

    if budget is not None:
        subset_size = int(len(dataset) * (budget / 100))
        return random.Random(RANDOM_SEED).sample(dataset, subset_size)

    return dataset


def load_specific_domain_abs(abs: list, domain_title: str = "P.Herc.") -> list:
    """
    Restituisce un sottoinsieme dei blocchi anonimi il cui dominio è rappresentato dal titolo immesso in `domain_title`
    """

    if not domain_title:
        return []

    return [
        ab
        for ab in abs
        if ab.get("language") == "grc"
        and isinstance(ab.get("title"), str)
        and domain_title in ab["title"]
    ]


def load_test_abs() -> list:
    """
    Carica i blocchi anonimi di test da un file JSON.
    Returns:
        list: Una lista di dizionari contenenti i blocchi anonimi di test.
    """
    with open(DATA_PATH / "test_abs.json", "r", encoding="utf-8") as f:
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

    train_abs, test_abs = train_test_split(
        abs, test_size=test_size, random_state=RANDOM_SEED
    )
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

    if remove_punct:
        def process_sent(s):
            return get_tokens_from_clean_text(remove_punctuation(s))
    else:
        process_sent = get_tokens_from_clean_text

    for obj in tqdm(abs, desc="Processing anonymous blocks", unit="ab", leave=False):
        training_text = obj.get("training_text")

        # obj.get e check string per il fail fast.
        if obj.get("language") == "grc" and training_text:
            clean_text = clean_text_from_gaps(training_text, case_folding=case_folding)
            for sent in sentence_tokenizer.tokenize(text=clean_text):
                if sent:
                    sentences.append(process_sent(sent))

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
