import re
TRAIN_DATASET_CHECKPOINT = "CNR-ILC/maat-corpus-train"
TEST_DATASET_CHECKPOINT = "CNR-ILC/maat-corpus-test"
BERT_TOKENS_PER_WORD = 1.77

CHUNK_SIZE = 50  # Define the desired chunk size
BERT_UNK_TOKEN = "[UNK]"
BERT_MAX_SEQ_LENGTH = 510  # 512 - 2 ([CLS] + [SEP])

# Valori soglia per la considerazione delle frasi
MAX_UNK_TOKEN_TRESHOLD = 5
MIN_SENT_TOKEN_TRESHOLD = 10
MAX_MASK_TOKEN_TRESHOLD = 10
MIN_MASK_TOKEN_TRESHOLD = 1

LACUNAE_REGEX = re.compile(r"\[([. ]+)\]")

# Configurazione specifica per modello BERT.
# AristoBERTo si basa su GreekBERT (tokenizer per greco moderno):
#   non riconosce punteggiatura e spiriti diacritici del greco antico → vanno rimossi.
# GreBerta e Logion hanno tokenizer addestrati specificamente per il greco antico:
#   riconoscono punteggiatura e diacritici → vanno preservati.
BERT_MODEL_CONFIG = {
    "CNR-ILC/gs-aristoBERTo": {
        "remove_punct": True,
        "strip_diacritics": True,
    },
    "CNR-ILC/gs-GreBerta": {
        "remove_punct": False,
        "strip_diacritics": False,
    },
    "CNR-ILC/gs-Logion": {
        "remove_punct": False,
        "strip_diacritics": False,
    },
}


def get_model_config(checkpoint: str) -> dict:
    """
    Restituisce la configurazione di preprocessing per un dato checkpoint BERT.

    Args:
        checkpoint (str): Il nome del checkpoint del modello BERT.

    Returns:
        dict: Configurazione con chiavi 'remove_punct' e 'strip_diacritics'.

    Raises:
        ValueError: Se il checkpoint non è presente nella configurazione.
    """
    if checkpoint not in BERT_MODEL_CONFIG:
        raise ValueError(
            f"Checkpoint '{checkpoint}' non trovato in BERT_MODEL_CONFIG. "
            f"Checkpoint disponibili: {list(BERT_MODEL_CONFIG.keys())}"
        )
    return BERT_MODEL_CONFIG[checkpoint]


from .utils import *