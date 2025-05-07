from metrics.accuracy import get_topK_accuracy
from train.training import pipeline_train
from config.settings import (
    K_PREDICTIONS,
    DIMENSIONS,
    TEST_SIZES,
    GAMMAS,
    MIN_FREQS,
    BATCH_SIZE,
    K_PRED,
)
from tuning import save_results, save_results_pickle
from skopt import gp_minimize
from skopt.space import Categorical
import json

BUDGET = 10  # Budget di riferimento per condurre l'ottimizzazione

SPACE = [
    Categorical(DIMENSIONS, name="dimension"),
    Categorical(TEST_SIZES, name="test_size"),
    Categorical(GAMMAS, name="gamma"),
    Categorical(MIN_FREQS, name="min_freq"),
]


def objective_function_lidstone(params):
    n, test_size, gamma, min_freq = params
    model, val = pipeline_train(
        lm_type="LIDSTONE",
        gamma=gamma,
        min_freq=min_freq,
        n=n,
        test_size=test_size,
        budget=BUDGET,
    )
    return -get_topK_accuracy(model, val, BATCH_SIZE, n, K_PRED)


def objective_function_mle(params):
    n, test_size, _, min_freq = params
    model, val = pipeline_train(
        lm_type="MLE", min_freq=min_freq, n=n, test_size=test_size, budget=BUDGET
    )
    return -get_topK_accuracy(model, val, BATCH_SIZE, n, K_PRED)


def save_results_json(res, filename):
    """
    Salva i risultati dell'ottimizzazione in formato JSON.
    """
    with open(filename, "w") as f:
        json.dump(
            {
                "x": res.x,
                "fun": res.fun,
                "space": [dim.name for dim in SPACE],
            },
            f,
        )


def bo_gp_lidstone():
    """
    Funzione di ottimizzazione bayesiana per il modello LIDSTONE.
    """
    opt = gp_minimize(objective_function_lidstone, SPACE, random_state=42)
    save_results_json(opt, "lidstone_tuning_results.json")
    return opt


def bo_gp_mle():
    """
    Funzione di ottimizzazione bayesiana per il modello MLE.
    """
    opt = gp_minimize(objective_function_mle, SPACE, random_state=42)
    save_results_json(opt, "mle_tuning_results.json")
    return opt


if __name__ == "__main__":
    # Esegui l'ottimizzazione bayesiana per LIDSTONE
    # res_lidstone = bo_gp_lidstone()
    # print (res_lidstone)
    # Esegui l'ottimizzazione bayesiana per MLE
    res_mle = bo_gp_mle()
