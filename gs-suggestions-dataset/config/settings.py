from cltk.sentence.grc import GreekRegexSentenceTokenizer
from cltk.tokenizers.processes import GreekTokenizationProcess
from pathlib import Path

# Tokenizer per testo -> frasi
sentence_tokenizer = GreekRegexSentenceTokenizer()

# Tokenizer per frase -> tokens
tokenizer = GreekTokenizationProcess()

# Parametri di configurazione (default)

LM_TYPE = 'MLE' #Tipo di Language Model
GAMMA = 1 #k-smoothing
K_PRED = 10  # Numero delle predizioni che il modello deve fare nella funzione di accuracy (da poi confrontare con la gold label)
BATCH_SIZE = 10  # Dimensione del batch
DATA_PATH = Path("data/") #Percorso al dataset
TEST_SIZE = 0.10  # Percentuale di dati di test
N = 3  # Dimensione degli ngrammi
