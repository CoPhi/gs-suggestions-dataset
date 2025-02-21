from cltk.sentence.grc import GreekRegexSentenceTokenizer
from cltk.tokenizers.processes import GreekTokenizationProcess
from pathlib import Path

# Tokenizer testo -> frasi (str -> list[str])
sentence_tokenizer = GreekRegexSentenceTokenizer()

# Tokenizer frase -> tokens (str -> list[str])
tokenizer = GreekTokenizationProcess()

#Iperparametri
GAMMA = 0.001 #k-smoothing
K_PRED = 20  # Numero delle predizioni che il modello deve fare nella funzione di accuracy (da poi confrontare con la gold label)
BATCH_SIZE = 100  # Dimensione del batch
DATA_PATH = Path("data/") #Percorso al dataset
TEST_SIZE = 0.05  # Percentuale di dati di test
N = 3# Dimensione degli ngrammi

# Parametri di configurazione del modello
LM_TYPE = 'LIDSTONE' #Tipo di Language Model