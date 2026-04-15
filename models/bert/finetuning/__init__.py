import re
from cltk.sentence.grc import GreekRegexSentenceTokenizer

sentence_tokenizer = GreekRegexSentenceTokenizer()

TRAIN_DATASET_CHECKPOINT = "CNR-ILC/maat-corpus-train"
TEST_DATASET_CHECKPOINT = "CNR-ILC/maat-corpus-test"

OUTPUT_DIR = "./models/bert/finetuning/gs/gs-greBERTa"
LOGS_DIR = "./models/bert/finetuning/gs/gs-greBERTa-logs"

CHUNK_SIZE = 50  # Dimensione del chunk per il push su Hugging Face Hub
BERT_UNK_TOKEN = "[UNK]"
BERT_MAX_SEQ_LENGTH = 510  # 512 - 2 ([CLS] + [SEP])

# Valori soglia per la considerazione delle frasi
MAX_UNK_TOKEN_TRESHOLD = 5
MIN_SENT_TOKEN_TRESHOLD = 10
MAX_MASK_TOKEN_TRESHOLD = 10
MIN_MASK_TOKEN_TRESHOLD = 1

# Configurazione specifica per modello BERT.
#
# case_folding: "upper" | "lower" | None
#   - "upper": converte in maiuscolo (AristoBERTo, GreBerta per il fine-tuning GS)
#   - "lower": converte in minuscolo (Logion, come da paper Cowen-Breen et al. 2023)
#   - None:    preserva il casing originale del testo
#
# AristoBERTo si basa su GreekBERT (tokenizer per greco moderno):
#   non riconosce punteggiatura e spiriti diacritici del greco antico → vanno rimossi.
# GreBerta ha un tokenizer addestrato specificamente per il greco antico:
#   riconosce punteggiatura e diacritici politonici, ma il fine-tuning GS usa uppercase
#   e strip_diacritics per coerenza con il formato dei papiri.
# Logion (Cowen-Breen et al. 2023): lowercase + strip diacritics, punteggiatura preservata.
BERT_MODEL_CONFIG = {
    "CNR-ILC/gs-aristoBERTo": {
        "remove_punct": True,
        "strip_diacritics": True,
        "case_folding": "upper",
    },
    "CNR-ILC/gs-GreBerta": {
        "remove_punct": False,
        "strip_diacritics": True,
        "case_folding": "upper",
    },
    "CNR-ILC/gs-Logion": {
        "remove_punct": False,
        "strip_diacritics": True,
        "case_folding": "lower",
    },
}

# Mappa checkpoint fine-tuned → checkpoint base di partenza per i pesi
BASE_MODEL_MAP = {
    "CNR-ILC/gs-aristoBERTo": "Jacobo/aristoBERTo",
    "CNR-ILC/gs-GreBerta": "bowphs/GreBerta",
    "CNR-ILC/gs-Logion": "cabrooks/LOGION-50k_wordpiece",
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