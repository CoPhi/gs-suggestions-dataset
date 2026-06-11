import threading
from cltk.sentence.grc import GreekRegexSentenceTokenizer
import wandb
import os
from dotenv import load_dotenv

load_dotenv()


sentence_tokenizer = GreekRegexSentenceTokenizer()

TRAIN_DATASET_CHECKPOINT = "CNR-ILC/gs-dataset-train"
TEST_DATASET_CHECKPOINT = "CNR-ILC/gs-dataset-eval"

OUTPUT_DIR = "./models/bert/finetuning/gs/gs-greBERTa"

CHUNK_SIZE = 50  # Dimensione del chunk per il push su Hugging Face Hub
BERT_MAX_SEQ_LENGTH = 510  # 512 - 2 ([CLS] + [SEP])

# Valori soglia per la considerazione delle frasi
MAX_UNK_TOKEN_TRESHOLD = 5
MIN_SENT_TOKEN_TRESHOLD = 10
MAX_MASK_TOKEN_TRESHOLD = 10
MIN_MASK_TOKEN_TRESHOLD = 1

# Lunghezza massima dello span da mascherare (per il collator MLM)
# Per ora viene settato a 3 per approssimare lacuna di lunghezza fino a ~6 caratteri
MAX_SPAN_LENGTH = 3

# placeholder lacuna
GAP_TOKEN = "<GAP_TEMP_INFILL>"


# Configurazione specifica per modello BERT.
#
# case_folding: "upper" | "lower" | "none"
#   - "upper": converte in maiuscolo (AristoBERTo, GreBerta)
#   - "lower": converte in minuscolo (Logion, Cowen-Breen et al. 2023)
#   - "none": preserva il casing originale del testo


class ModelRegistry:
    """
    Singleton pattern per gestire in maniera centralizzata le configurazioni di preprocessing
    e la calibrazione degli iperparametri per i diversi modelli BERT.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelRegistry, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Mappa checkpoint fine-tuned -> configurazioni e iperparametri
        self.configs = {
            "CNR-ILC/gs-aristoBERTo": {
                "remove_punct": True,
                "strip_diacritics": True,
                "case_folding": "lower",
                "hyperparameters": {
                    "chunk_size": 256,
                    "batch_size": 128,
                    "lr": 1.187383488773285e-06,
                    "epochs": 2,
                    "num_layers_to_freeze": 0,
                    "weight_decay": 0.1,
                    "warmup_ratio": 0.1,
                    "mlm_probability": 0.2,
                    "max_span_length": 3,
                    "lr_scheduler_type": "cosine",
                },
            },
            "CNR-ILC/gs-GreBerta": {
                "remove_punct": True,
                "strip_diacritics": True,
                "case_folding": "lower",
                "hyperparameters": {
                    "chunk_size": 128,
                    "batch_size": 128,
                    "lr": 1e-6,
                    "epochs": 3,
                    "num_layers_to_freeze": 0,
                    "weight_decay": 0.1,
                    "warmup_ratio": 0.1,
                    "mlm_probability": 0.1,
                    "max_span_length": 2,
                    "lr_scheduler_type": "cosine",
                },
            },
            "CNR-ILC/gs-Logion": {
                "remove_punct": True,
                "strip_diacritics": True,
                "case_folding": "lower",
                "hyperparameters": {
                    "chunk_size": 256,
                    "batch_size": 128,
                    "lr": 2e-5,
                    "epochs": 3,
                    "num_layers_to_freeze": 10,
                    "weight_decay": 0.01,
                    "warmup_ratio": 0.1,
                    "mlm_probability": 0.1,
                    "max_span_length": 3,
                    "lr_scheduler_type": "linear",
                },
            },
        }

        # Mappa checkpoint fine-tuned → checkpoint base di partenza per i pesi
        self.base_model_map = {
            "CNR-ILC/gs-aristoBERTo": "Jacobo/aristoBERTo",
            "CNR-ILC/gs-GreBerta": "bowphs/GreBerta",
            "CNR-ILC/gs-Logion": "cabrooks/LOGION-50k_wordpiece",
        }

    def get_config(self, checkpoint: str) -> dict:
        """
        Recupera la configurazione e gli iperparametri per un dato modello.
        Ritorna un dizionario flat ("appiattito") per garantire la retrocompatibilità.
        """
        resolved_checkpoint = checkpoint
        if checkpoint not in self.configs:
            for finetuned, base in self.base_model_map.items():
                if base == checkpoint:
                    resolved_checkpoint = finetuned
                    break

        if resolved_checkpoint not in self.configs:
            raise ValueError(
                f"Checkpoint '{checkpoint}' non trovato nel ModelRegistry. "
                f"Checkpoint disponibili: {list(self.configs.keys())} o i loro base models."
            )

        # Restituisce una copia shallow con gli iperparametri appiattiti nel dizionario principale
        config = self.configs[resolved_checkpoint].copy()
        hyperparams = config.pop("hyperparameters", {})
        config.update(hyperparams)
        return config


def get_model_config(checkpoint: str) -> dict:
    """
    Funzione di facciata per mantenere la retrocompatibilità con gli script esistenti.
    Utilizza il ModelRegistry Singleton per recuperare la configurazione.
    """
    return ModelRegistry().get_config(checkpoint)


# Token di accesso di HuggingFace Hub
HF_TOKEN = os.getenv("HF_TOKEN")

# variabili di configurazione per wandb
WANDB_PROJECT = "gs-suggestions"
WANDB_API_KEY = os.getenv("WANDB_API_KEY")


def wandb_login(api_key: str = WANDB_API_KEY) -> str:
    if not api_key.strip():
        return "Inserisci una API key valida."
    try:
        wandb.login(key=api_key.strip(), relogin=True)
        return "Login wandb effettuato con successo."
    except Exception as e:
        return f"Errore: {e}"
