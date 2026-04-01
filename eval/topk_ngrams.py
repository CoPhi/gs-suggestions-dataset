from sklearn.model_selection import KFold
from models.ngrams.train import load_abs, load_lm
import json
from backend.config.settings import (
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

from models.ngrams.metrics.accuracy import get_topK_accuracy
from models.ngrams.train.cleaner import load_specific_domain_abs, load_test_abs
from models.ngrams.train.training import train_lm

if __name__ == "__main__":

    train_abs = load_abs()
    domain_abs = load_specific_domain_abs(abs=train_abs)
    test_abs = load_test_abs()

    # assumiamo che gli iperparametri siano ottimizzati
    g_lm = train_lm(train_abs=train_abs, lm_type=LM_TYPE, n=N, gamma=GAMMA)
    d_lm = train_lm(train_abs=domain_abs, lm_type=LM_TYPE, n=N, gamma=GAMMA)

    results = {}
    for k_pred in K_PREDICTIONS:
        results[f"eval_top{k_pred}_acc"] = get_topK_accuracy(
            g_lm=g_lm,
            d_lm=d_lm,
            test_abs=test_abs,
            lambda_weight=LAMBDA,
            batch_size=1,
            n=N,
            k_pred=k_pred,
            alpha=ALPHA,
            beta=BETA,
            delta=DELTA,
        )

    # Salva i risultati in un file JSON
    with open(f"finetuning/eval_results_ngrams_{LM_TYPE}.json", "w") as f:
        json.dump(results, f, indent=4)
