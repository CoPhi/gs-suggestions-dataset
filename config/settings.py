from pathlib import Path

# Parametri per lo splitting dei dati nei file
CORPUS_PATHS = [
    "/home/gabriele/cltk_data/grc/corpora/First1KGreek/data/",
    "/home/gabriele/cltk_data/grc/corpora/PerseusDL/canonical-greekL/data/",
    "/home/gabriele/cltk_data/grc/corpora/idp.data/DDB_EpiDoc_XML/",
    "/home/gabriele/cltk_data/grc/corpora/idp.data/DCLP/",
]

LIM = 50  # MAX MB per file
INDENT = 0  # indentazione

# Iperparametri per il modello
LM_TYPE = "LIDSTONE"  # Tipo di Language Model
GAMMA = 0.001  # k-smoothing
K_PRED = 20  # Numero delle predizioni che il modello deve fare nella funzione di accuracy (da poi confrontare con la gold label)
BATCH_SIZE = 64  # Dimensione del batch
DATA_PATH = Path("data/")  # Percorso al dataset
TEST_SIZE = 0.05  # Percentuale di dati di test
MIN_FREQ = 2  # Frequenza minima per i token dentro il modello
N = 3  # Dimensione degli ngrammi
CORPUS_NAMES = set(["DDbDP", "DCLP", "EDH"])

# Spazio di ricerca degli iperparametri
LM_TYPES = ["LIDSTONE", "MLE"]
GAMMAS = [0.001, 0.01, 0.1]
K_PREDICTIONS = [10, 20]
MIN_FREQS = [2, 3, 4]
TEST_SIZES = [0.05, 0.1]
DIMENSIONS = [2, 3]
BATCH_SIZES = [32, 64]
