import os
from pathlib import Path
from typing import Any

# Parametri per lo splitting dei dati nei file
CORPUS_PATHS = [
    "/home/gabriele/cltk_data/grc/corpora/First1KGreek/data/",
    "/home/gabriele/cltk_data/grc/corpora/canonical-greekL/data/",
    "/home/gabriele/cltk_data/grc/corpora/idp.data/DDB_EpiDoc_XML/",
    "/home/gabriele/cltk_data/grc/corpora/idp.data/DCLP/",
]

LIM = 50  # MAX MB per file
INDENT = 0  # indentazione

# Iperparametri per il modello
RANDOM_SEED = 42
LM_TYPE = "LIDSTONE"  # Tipo di Language Model
GAMMA = 0.001  # k-smoothing
K_PRED = 20  # Numero delle predizioni del modello 
BATCH_SIZE = 64  # Dimensione del batch per il calcolo dell'accuracy
DATA_PATH = Path("data/")  # Percorso al dataset
TEST_SIZE = 0.05  # Percentuale di dati di test
MIN_FREQ = 1  # Frequenza minima per i token dentro il modello
N = 3  # Dimensione degli ngrammi
CORPUS_NAMES = set(["DDbDP", "DCLP", "EDH", "tlg"])
LAMBDA = 0.2 #interpolazione
ALPHA = 0 #NLL score
BETA = 1 #edit distance 
DELTA = 1 #length penalty

BERT_CHECKPOINTS = ["CNR-ILC/gs-aristoBERTo", "CNR-ILC/gs-GreBerta", "CNR-ILC/gs-Logion"]

# Spazio di ricerca degli iperparametri
LM_TYPES = ["LIDSTONE", "MLE"]
GAMMAS = [0.001, 0.01, 0.1]
K_PREDICTIONS = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
MIN_FREQS = [1, 2, 3, 4]
TEST_SIZES = [0.05, 0.1, 0.2]
DIMENSIONS = [2, 3]
BATCH_SIZES = [32, 64]

# Percorsi checkpoint pre-addestrati N-gram
NGRAM_CHECKPOINTS_DIR = Path("models/ngrams/checkpoints")

# Parametri di default per il sistema DSPy RAG
RAG_INDEX_DIR = Path(os.getenv("RAG_INDEX_DIR", "models/rag/index"))
RAG_COMPILED_DIR = Path("models/rag/checkpoints")
RAG_DEFAULT_LM = os.getenv("RAG_DEFAULT_LM", "Qwen/Qwen2.5-72B-Instruct")
RAG_DEFAULT_BERT = os.getenv("RAG_DEFAULT_BERT", "CNR-ILC/gs-GreBerta")
RAG_DEFAULT_BACKEND = os.getenv("RAG_DEFAULT_BACKEND", "auto")
RAG_DEFAULT_EMBEDDER = os.getenv("RAG_DEFAULT_EMBEDDER", "CNR-ILC/gs-GreBerta")
RAG_TOP_K_PASSAGES = int(os.getenv("RAG_TOP_K_PASSAGES", "5"))
RAG_CHECKPOINTS = [RAG_DEFAULT_LM, RAG_DEFAULT_BERT]

# Catalogo di modelli LLM e BERT neuro-lessicali supportati per il sistema DSPy RAG
RAG_SUPPORTED_LMS: list[dict[str, Any]] = [
    {
        "id": "CNR-ILC/gs-GreBerta",
        "name": "gs-GreBerta + RAG (Neuro-lessicale locale)",
        "category": "BERT + RAG",
        "recommended": True,
        "description": "Modello BERT neuro-lessicale con retrieval ibrido BM25+GreBerta, attestazioni TLG e HCB Beam Search (100% in-process locale, zero dipendenze da LLM esterni)",
    },
    {
        "id": "CNR-ILC/gs-aristoBERTo",
        "name": "gs-aristoBERTo + RAG (Neuro-lessicale locale)",
        "category": "BERT + RAG",
        "recommended": False,
        "description": "Modello BERT (aristoBERTo) accostato al sistema RAG con retrieval ibrido BM25+GreBerta e attestazioni TLG (in-process locale)",
    },
    {
        "id": "CNR-ILC/gs-Logion",
        "name": "gs-Logion + RAG (Neuro-lessicale locale)",
        "category": "BERT + RAG",
        "recommended": False,
        "description": "Modello BERT (Logion) accostato al sistema RAG con retrieval ibrido BM25+GreBerta e attestazioni TLG (in-process locale)",
    },
    {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "name": "Qwen 2.5 72B Instruct (Qwen)",
        "category": "Qwen",
        "recommended": True,
        "description": "Modello 72B ad altissima precisione filologica e sintesi CoT articolata (HF Router)",
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "name": "Llama 3.3 70B Instruct (Meta)",
        "category": "Llama",
        "recommended": False,
        "description": "Modello 70B di ultima generazione Meta per completamento e ragionamento (HF Router)",
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "name": "Llama 3.1 8B Instruct (Meta)",
        "category": "Llama",
        "recommended": False,
        "description": "Modello compatto da 8B, rapido e a basso costo computazionale (HF Router)",
    },
    {
        "id": "google/gemma-3-12b-it",
        "name": "Gemma 3 12B Instruct (Google)",
        "category": "Gemma",
        "recommended": False,
        "description": "Modello multilingue bilanciato Google DeepMind da 12B (HF Router)",
    },
    {
        "id": "google/gemma-3-27b-it",
        "name": "Gemma 3 27B Instruct (Google)",
        "category": "Gemma",
        "recommended": False,
        "description": "Modello 27B Google DeepMind con forte competenza linguistica (HF Router)",
    },
    {
        "id": "google/gemma-4-31B-it",
        "name": "Gemma 4 31B Instruct (Google)",
        "category": "Gemma",
        "recommended": False,
        "description": "Modello di frontiera Gemma 4 da 31B per inferenza avanzata (HF Router)",
    },
    {
        "id": "deepseek-ai/DeepSeek-V3",
        "name": "DeepSeek V3 (DeepSeek)",
        "category": "DeepSeek",
        "recommended": False,
        "description": "Architettura Mixture-of-Experts da 671B con alta efficienza (HF Router)",
    },
    {
        "id": "gemma4:e4b",
        "name": "Gemma 4 E4B (Ollama locale)",
        "category": "Ollama",
        "recommended": False,
        "description": "Esecuzione locale offline su CPU/GPU tramite demone Ollama",
    },
]


def get_ngram_checkpoint_path(
    lm_type: str,
    n: int = N,
    gamma: float | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Restituisce il percorso del file di checkpoint per una data configurazione N-gram."""
    base = base_dir or NGRAM_CHECKPOINTS_DIR
    lm = lm_type.upper()
    if lm == "MLE":
        filename = f"ngram_mle_n{n}.pkl.zlib"
    else:
        gamma_val = gamma if gamma is not None else GAMMA
        filename = f"ngram_{lm.lower()}_n{n}_gamma{gamma_val}.pkl.zlib"
    return base / filename

