from metrics.accuracy import get_topK_accuracy
from train.training import pipeline_train
from bayes_opt import BayesianOptimization
import json


def objective_function(gamma, lambda_weight):
    """
    Funzione obiettivo per ottimizzare il parametro gamma.
    """
    g_model, d_model, val = pipeline_train(lm_type="LIDSTONE", gamma=gamma, budget=50)
    return get_topK_accuracy(g_model, d_model, val, lambda_weight)


if __name__ == "__main__":
    pbounds = {"gamma": (1e-5, 1), "lambda_weight": (0, 1)}

    # Ottimizzatore
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=42,
    )

    optimizer.maximize(
        init_points=5,  # Numero di punti iniziali random
        n_iter=20,  # Numero di iterazioni di ottimizzazione
    )

    # Stampa i risultati migliori
    print("Miglior risultato:", optimizer.max)

    # Salva i risultati delle iterazioni in un file JSON
    with open("optimization_results.json", "w") as f:
        json.dump(optimizer.res, f, indent=4)
