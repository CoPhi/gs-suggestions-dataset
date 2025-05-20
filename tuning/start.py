from metrics import perplexity, get_topK_accuracy
from train.training import train_lm
from train import load_abs, load_specific_domain_abs
from bayes_opt import BayesianOptimization
from sklearn.model_selection import ShuffleSplit
import json


def objective_function(gamma, lambda_weight, alpha, beta, delta):
    """
    Funzione obiettivo per ottimizzare il parametro gamma.
    """
    all_blocks = load_abs(budget=50)
    domain_blocks = load_specific_domain_abs(abs=all_blocks)

    
    domain_ids = set(block["id"] for block in domain_blocks)

    # Filtra i blocchi che non sono nel dominio
    non_domain_blocks = [block for block in all_blocks if block["id"] not in domain_ids]
    ss = ShuffleSplit(n_splits=10, test_size=0.05, random_state=42)
    cv_topKs = []
    for train_idx, val_idx in ss.split(non_domain_blocks):
        X_train = [non_domain_blocks[i] for i in train_idx]
        X_val = [non_domain_blocks[i] for i in val_idx]
        g_lm, d_lm = train_lm(train_abs=X_train, lm_type="LIDSTONE", gamma=gamma, n=3)
        cv_topKs.append(get_topK_accuracy(g_lm=g_lm, d_lm=d_lm, test_abs=X_val, lambda_weight=lambda_weight, alpha=alpha, beta=beta, delta=delta))
        
    avg_topK = sum(cv_topKs) / len(cv_topKs)
    return avg_topK

if __name__ == "__main__":
    pbounds = {
        "gamma": (1e-5, 1),
        "lambda_weight": (0, 1),
        "alpha": (0, 1),
        "beta": (0, 1),
        "delta": (0, 1),
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
    results = [
    {
        "params": res["params"],
        "target": res["target"]
    }
    for res in optimizer.res
    ]

    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=4)