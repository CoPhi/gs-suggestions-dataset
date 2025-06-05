from sklearn.model_selection import KFold
from train import load_abs, load_lm
from train.training import train_lm
import json
from config.settings import (
    K_PREDICTIONS,
    ALPHA,
    DELTA,
    BETA,
    LAMBDA,
    N,
    LM_TYPE,
    GAMMA,
    TEST_SIZE,
)

from metrics import get_topK_accuracy

if __name__ == "__main__":
    g_lm, test_abs = load_lm("General_model")  # carico il modello generico 
    d_lm, _ = load_lm("Domain_model")  # carico il modello specifico di dominio 
    
    results = {}
    for k_pred in K_PREDICTIONS: 
        results[f"eval_top{k_pred}_acc"] = get_topK_accuracy(g_lm=g_lm,d_lm=d_lm,test_abs=test_abs, lambda_weight=LAMBDA, batch_size=1, n=N, k_pred=k_pred, alpha=ALPHA, beta=BETA, delta=DELTA)

    # Salva i risultati in un file JSON
    with open("finetuning/eval_results_ngrams.json", "w") as f:
        json.dump(results, f, indent=4)