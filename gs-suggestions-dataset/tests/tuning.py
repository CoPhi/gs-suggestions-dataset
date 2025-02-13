from skopt import gp_minimize
from skopt.space import Categorical
import pandas as pd
import os
from models.evaluate import accuracy, perplexity
from models.training import split_abs, load_abs, train_lm
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from skopt.utils import use_named_args
from skopt.callbacks import EarlyStopper
from skopt import Optimizer
import numpy as np

K_PREDS = [10, 15, 20]
DIMENSIONS = [2, 3]
TEST_SIZES = [0.1, 0.05]
BATCH_SIZES = [5, 15, 20]
GAMMA = [1, 10, 100]

# Spazio di ricerca per il modello MLE
space_mle = [
    Categorical(K_PREDS, name="k"),
    Categorical(DIMENSIONS, name="n"),
    Categorical(BATCH_SIZES, name="batch_size"),
    Categorical(TEST_SIZES, name="test_size"),
]

# Spazio di ricerca per il modello LIDSTONE
space_lidstone = [
    Categorical(K_PREDS, name="k"),
    Categorical(DIMENSIONS, name="n"),
    Categorical(BATCH_SIZES, name="batch_size"),
    Categorical(TEST_SIZES, name="test_size"),
    Categorical(GAMMA, name="gamma"),
]


def print_MLE_params_to_csv(params: dict) -> None:
    file_exists = os.path.isfile("MLE_results.csv")
    df = pd.DataFrame([params])
    df.to_csv("MLE_results.csv", mode="a", header=not file_exists, index=False)


def print_LIDSTONE_params_to_csv(params: dict) -> None:
    file_exists = os.path.isfile("LIDSTONE_results.csv")
    df = pd.DataFrame([params])
    df.to_csv("LIDSTONE_results.csv", mode="a", header=not file_exists, index=False)


def evaluate_mle(abs, **params) -> float:
    k = params["k"]
    n = params["n"]
    batch_size = params["batch_size"]
    test_size = params["test_size"]

    train_abs, test_abs = split_abs(abs, test_size=test_size)
    lm = train_lm(train_abs, lm_type="MLE", n=n)
    acc = accuracy(lm, test_abs, batch_size=batch_size, k_pred=k, n=n)

    params["accuracy"] = acc
    print_MLE_params_to_csv(params)

    return -acc  # Minimize negative accuracy to maximize accuracy


def evaluate_lidstone(abs, **params) -> float:
    k = params["k"]
    n = params["n"]
    batch_size = params["batch_size"]
    test_size = params["test_size"]
    gamma = params["gamma"]

    train_abs, test_abs = split_abs(abs, test_size=test_size)
    lm = train_lm(train_abs, lm_type="LIDSTONE", gamma=gamma, n=n)
    acc = accuracy(lm, test_abs, batch_size=batch_size, k_pred=k, n=n)
    pp = perplexity(lm, test_abs, n=n)

    params["accuracy"] = acc
    params["perplexity"] = pp
    print_LIDSTONE_params_to_csv(params)

    return -acc  # Minimize negative accuracy to maximize accuracy


def custom_scorer(estimator, X, y):
    acc = accuracy(estimator, X, batch_size=y["batch_size"], k_pred=y["k"], n=y["n"])
    return acc


def random_search():
    random_search = RandomizedSearchCV(
        estimator=train_lm,  # La tua funzione di addestramento del modello
        param_distributions={
            "k": K_PREDS,
            "n": DIMENSIONS,
            "batch_size": BATCH_SIZES,
            "test_size": TEST_SIZES,
            "gamma": GAMMA,
        },
        n_iter=50,  # Numero di iterazioni
        scoring=make_scorer(custom_scorer, greater_is_better=True),
        cv=3,  # Numero di fold per la cross-validation
        random_state=0,
    )

    abs = load_abs()
    for test_size in TEST_SIZES:
        train_abs, test_abs = split_abs(abs, test_size=test_size)
        random_search.fit(train_abs, test_abs)
        
    print_MLE_params_to_csv({random_search.best_params_, random_search.best_score_ })

    print(f"Best Score: {random_search.best_params_}")
    print(f"Best Parameters: { random_search.best_score_}")


def hyperband():
    abs = load_abs()

    def evaluate_mle_wrapper(params):
        param_dict = {dim.name: val for dim, val in zip(space_mle, params)}
        score = evaluate_mle(abs, **param_dict)
        print_MLE_params_to_csv(param_dict)
        return score

    def evaluate_lidstone_wrapper(params):
        param_dict = {dim.name: val for dim, val in zip(space_lidstone, params)}
        score = evaluate_lidstone(abs, **param_dict)
        print_LIDSTONE_params_to_csv(param_dict)
        return score

    opt = Optimizer(space_mle, base_estimator="GP", acq_func="EI", random_state=0)
    for i in range(50):
        next_params = opt.ask()
        score = evaluate_mle_wrapper(next_params)
        opt.tell(next_params, score)
        if EarlyStopper(10)(opt):
            break

    print(f"Best Score: {opt.get_result().x}")
    print(f"Best Parameters: {-opt.get_result().fun}")


def bayesian_optimization():
    abs = load_abs()

    def evaluate_mle_wrapper(params):
        param_dict = {dim.name: val for dim, val in zip(space_mle, params)}
        return evaluate_mle(abs, **param_dict)

    def evaluate_lidstone_wrapper(params):
        param_dict = {dim.name: val for dim, val in zip(space_lidstone, params)}
        return evaluate_lidstone(abs, **param_dict)

    with ThreadPoolExecutor() as executor:
        future_mle = executor.submit(
            gp_minimize, evaluate_mle_wrapper, space_mle, n_calls=20, random_state=0
        )
        future_lidstone = executor.submit(
            gp_minimize,
            evaluate_lidstone_wrapper,
            space_lidstone,
            n_calls=20,
            random_state=0,
        )

        res_mle = future_mle.result()
        res_lidstone = future_lidstone.result()

    best_params_mle = res_mle.x
    best_accuracy_mle = -res_mle.fun

    print(f"Best MLE Accuracy: {best_accuracy_mle}")
    print(f"Best MLE Parameters: {best_params_mle}")

    best_params_lidstone = res_lidstone.x
    best_accuracy_lidstone = -res_lidstone.fun

    print(f"Best LIDSTONE Accuracy: {best_accuracy_lidstone}")
    print(f"Best LIDSTONE Parameters: {best_params_lidstone}")
    best_params_lidstone = res_lidstone.x
    best_accuracy_lidstone = -res_lidstone.fun

    print(f"Best LIDSTONE Accuracy: {best_accuracy_lidstone}")
    print(f"Best LIDSTONE Parameters: {best_params_lidstone}")


if __name__ == "__main__":
    random_search()
