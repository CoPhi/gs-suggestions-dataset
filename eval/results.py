from sklearn.model_selection import KFold
from train import load_abs, load_lm
from train.training import train_lm
from config.settings import (
    K_PRED,
    BATCH_SIZE,
    N,
    LM_TYPE,
    GAMMA,
    TEST_SIZE,
)

from metrics import get_topK_accuracy

if __name__ == "__main__":
    g_lm, test_abs = load_lm("General_model")  # carico il modello generico 
    d_lm, _ = load_lm("Domain_model")  # carico il modello specifico di dominio 
    
    print("Accuracy: ", get_topK_accuracy(g_lm, d_lm, test_abs))
