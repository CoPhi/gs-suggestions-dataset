from backend.config.settings import K_PRED, N, RANDOM_SEED
from models.ngrams.metrics import get_topK_accuracy
from models.ngrams.train.cleaner import split_abs
from models.ngrams.train.training import train_lm
from models.ngrams.train import load_abs, load_specific_domain_abs
from bayes_opt import BayesianOptimization
import json

def objective_function_factory(dev_domain_abs, train_domain_abs, train_abs_all):
    # Pre-calcoliamo il mapping O(1) con l'id() all'esterno della funzione obiettivo 
    # così da eseguirlo un'unica volta invece di ripeterlo ad ogni iterazione di Opt
    dev_ids = {id(ab) for ab in dev_domain_abs}
    train_abs_filtered = [ab for ab in train_abs_all if id(ab) not in dev_ids]

    def objective_function(gamma, lambda_weight, alpha, beta, delta):
        """
        Funzione obiettivo per ottimizzare i parametri.
        """
        g_lm = train_lm(
            train_abs=train_abs_filtered,
            gamma=gamma,
        )
        
        d_lm = train_lm(
            train_abs=train_domain_abs,
            gamma=gamma,
        )

        return get_topK_accuracy(
            g_lm=g_lm,
            d_lm=d_lm,
            test_abs=dev_domain_abs,
            lambda_weight=lambda_weight,
            n=N,
            batch_size=1,
            k_pred=K_PRED,
            alpha=alpha,
            beta=beta,
            delta=delta,
        )
    return objective_function

if __name__ == "__main__":
    pbounds = {
        "gamma": (1e-5, 1),
        "lambda_weight": (0, 1),
        "alpha": (0, 1),
        "beta": (0, 1),
        "delta": (0, 1),
    }

    train_abs = load_abs()
    domain_abs = load_specific_domain_abs(abs=train_abs)

    train_domain_abs, dev_domain_abs = split_abs(domain_abs, test_size=0.2)

    objective_function = objective_function_factory(
        dev_domain_abs=dev_domain_abs,
        train_domain_abs=train_domain_abs,
        train_abs_all=train_abs
    )

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=RANDOM_SEED,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=20,
    )

    print("Miglior risultato:", optimizer.max)

    results = [
        {
            "params": res["params"],
            "target": res["target"]
        }
        for res in optimizer.res
    ]

    with open("optimization_results.json", "w") as f:
        json.dump(results, f, indent=4)
