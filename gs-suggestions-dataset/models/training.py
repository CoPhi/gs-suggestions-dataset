from collections import Counter
import json
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
from cltk.core.data_types import Doc
from nltk.lm.models import MLE, Lidstone, LanguageModel
from nltk.lm.vocabulary import Vocabulary
from nltk.lm.preprocessing import (
    padded_everygram_pipeline,
)

from config.settings import DATA_PATH, TEST_SIZE, N, sentence_tokenizer, tokenizer
from utils.preprocess import clean_text


def load_abs() -> list:
    """
    Carica e restituisce una lista di anonymous block (ab) dai file JSON presenti nel percorso specificato.
    Il percorso dei file JSON è determinato dalla variabile globale DATA_PATH.
    Ogni file JSON viene aperto e il suo contenuto viene aggiunto alla lista degli anonymous blocks.
    Returns:
        list: Una lista contenente gli anonymous block (abs) caricati dai file JSON.
    """
    abs = []
    for file_path in tqdm(
        list(DATA_PATH.glob("*.json")),
        desc="Processing MAAT corpus",
        unit="file",
        leave=False,
    ):
        with open(file_path, "r") as f:
            data = json.load(f)
            abs.extend(data)

    return abs


def split_abs(abs: list) -> tuple:
    """
    Divide una lista di anonymous block in due sottoinsiemi per l'addestramento e il test.
    Args:
        abs (list): Lista di anonymous block da dividere.
    Returns:
        tuple: Una tupla contenente due liste, la prima per l'addestramento e la seconda per il test.
    """

    train_abs, test_abs = train_test_split(abs, test_size=TEST_SIZE)
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
                text=clean_text(obj["training_text"])
            ):
                if sent:
                    sentences.append(tokenizer.run(input_doc=Doc(raw=sent)).tokens)
    return sentences


def filter_vocab(vocab_tokens, min_freq: int):
    """
    Filtra il vocabolario rimuovendo i token con frequenza inferiore a min_freq.

    Args:
        vocab (Vocabulary): Il vocabolario da filtrare.
        min_freq (int): La frequenza minima richiesta per mantenere un token nel vocabolario.

    Returns:
        Vocabulary: Il vocabolario filtrato.
    """
    token_counts = Counter(vocab_tokens)
    return Vocabulary(
        [token for token, freq in token_counts.items() if freq >= min_freq]
    )


def train_lm(train_abs: list) -> LanguageModel:
    """
    Addestra il modello sulle frasi di addestramento prelevate dall'insieme train_ab.
    Le frasi subiscono un controllo per rimuovere token non validi nella fase di addestramento.
    """

    lm = MLE(N)

    train_ngrams, vocab_tokens = padded_everygram_pipeline(
        order=N, text=get_sentences(train_abs)
    )

    lm.fit(train_ngrams, filter_vocab(vocab_tokens, 2))
    return lm


def pipeline_train():
    train_abs, test_abs = split_abs(load_abs())
    save_lm(lm=train_lm(train_abs), test_abs=test_abs)


def save_lm(lm: LanguageModel, test_abs: list, model_path="trigram_lm_MLE.pkl") -> None:
    """
    Salva il modello linguistico su disco.

    Args:
        model_path (str): Percorso per salvare il modello.
    """
    model_data = {"lm": lm, "test_ab": test_abs}
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
        print("Language model saved.")


def load_lm(model_path="trigram_lm_MLE.pkl") -> None:
    """
    Carica il modello linguistico da disco.
    Args:
        model_path (str): Percorso da cui caricare il modello.
    """
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        lm = model_data["lm"]
        test_ab = model_data["test_ab"]
        print("Language model loaded.")
        return lm, test_ab

    return None


if __name__ == "__main__":
    pipeline_train()
