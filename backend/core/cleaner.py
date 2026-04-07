import json
import random
from typing import Optional
from sklearn.model_selection import train_test_split
from backend.core import sentence_tokenizer, _LANGUAGE
from tqdm import tqdm
from backend.config.settings import (
    CORPUS_NAMES,
    DATA_PATH,
    TEST_SIZE,
    TEST_SIZES,
    RANDOM_SEED,
)
from backend.core.preprocess import (
    clean_text_from_gaps,
    get_tokens_from_clean_text,
    remove_punctuation,
)


def check_ab(ab: dict, corpus_set: Optional[list[str]] = None) -> bool:
    if ab.get("language") != _LANGUAGE:
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
        if ab.get("language") == _LANGUAGE
        and isinstance(ab.get("title"), str)
        and domain_title in ab["title"]
    ]


def load_test_set() -> list:
    """
    Carica i blocchi anonimi di test da un file JSON.
    Returns:
        list: Una lista di dizionari contenenti i blocchi anonimi di test.
    """
    with open(DATA_PATH / "test_abs.json", "r", encoding="utf-8") as f:
        return json.load(f)


# Metodi per la divisione dei blocchi anonimi in train e test


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


def split_abs_herc_dev(
    abs: list,
    test_size: float = TEST_SIZE,
    domain_title: str = "P.Herc.",
) -> tuple[list, list]:
    """
    Divide i blocchi anonimi in train e dev, assicurando che il dev set
    contenga esclusivamente testi dei Papiri di Ercolano.

    Args:
        abs: tutti i blocchi anonimi caricati da load_abs()
        test_size: proporzione del dev set rispetto ai soli blocchi P.Herc.
        domain_title: prefisso del titolo usato per identificare i P.Herc.

    Returns:
        (train_abs, dev_abs)
    """
    herc_abs = load_specific_domain_abs(abs, domain_title)
    non_herc_abs = [ab for ab in abs if ab not in set(map(id, herc_abs))]

    herc_train, herc_dev = train_test_split(
        herc_abs,
        test_size=test_size,
        random_state=RANDOM_SEED,
    )

    train_abs = non_herc_abs + herc_train
    return train_abs, herc_dev


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
        if obj.get("language") == _LANGUAGE and training_text:
            clean_text = clean_text_from_gaps(training_text, case_folding=case_folding)
            for sent in sentence_tokenizer.tokenize(text=clean_text):
                if sent:
                    sentences.append(process_sent(sent))

    return sentences
