from pathlib import Path

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
RAG_INDEX_DIR = Path("checkpoints/rag_index")
RAG_COMPILED_DIR = Path("models/rag/checkpoints")
RAG_DEFAULT_LM = "Qwen/Qwen2.5-7B-Instruct"
RAG_DEFAULT_BACKEND = "auto"
RAG_DEFAULT_EMBEDDER = "CNR-ILC/gs-GreBerta"
RAG_TOP_K_PASSAGES = 5


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

