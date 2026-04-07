import pickle
from nltk.lm.api import LanguageModel
from backend.config.settings import N


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
