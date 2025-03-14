from cltk.sentence.grc import GreekRegexSentenceTokenizer
from cltk.tokenizers.processes import GreekTokenizationProcess
from pathlib import Path

# Tokenizer testo -> frasi (str -> list[str])
sentence_tokenizer = GreekRegexSentenceTokenizer()

#URI per il database 
MONGO_URI = "mongodb://localhost:27017/"

#Iperparametri per il modello
LM_TYPE = 'LIDSTONE' #Tipo di Language Model
GAMMA = 0.001 #k-smoothing
K_PRED = 20  # Numero delle predizioni che il modello deve fare nella funzione di accuracy (da poi confrontare con la gold label)
BATCH_SIZE = 64  # Dimensione del batch
DATA_PATH = Path("data/") #Percorso al dataset
TEST_SIZE = 0.05  # Percentuale di dati di test
N = 3 # Dimensione degli ngrammi

#Spazio di ricerca degli iperparametri
LM_TYPES = ['LIDSTONE', 'MLE']
GAMMAS = [0.001, 0.01, 0.1]
K_PREDICTIONS = [10, 20]
TEST_SIZES = [0.05, 0.1]
DIMENSIONS = [2,3]
BATCH_SIZES = [32, 64]