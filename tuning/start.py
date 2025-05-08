from metrics.accuracy import get_topK_accuracy
from train.training import pipeline_train
from bayes_opt import BayesianOptimization
import json

def objective_function(gamma):
    """
    Funzione obiettivo per ottimizzare il parametro gamma.
    """
    model, val = pipeline_train(
        lm_type="LIDSTONE", gamma=gamma, budget=50
    )
    return get_topK_accuracy(model, val)

if __name__ == "__main__":
    pbounds = {
        'gamma': (1e-5, 1)  
    }

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