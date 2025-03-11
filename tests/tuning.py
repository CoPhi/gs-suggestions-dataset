from skopt import gp_minimize
from skopt.space import Categorical

from models.evaluate import get_topK_accuracy, perplexity
from models.training import split_abs, load_abs, train_lm
from tests.params import print_MLE_params_to_csv, print_LIDSTONE_params_to_csv
from concurrent.futures import ThreadPoolExecutor
from config.settings import K_PREDICTIONS, DIMENSIONS, TEST_SIZES, BATCH_SIZES, GAMMA

# Spazio di ricerca per il modello MLE
space_mle = [
    Categorical(K_PREDICTIONS, name="k"),
    Categorical(DIMENSIONS, name="n"),
    Categorical(BATCH_SIZES, name="batch_size"),
    Categorical(TEST_SIZES, name="test_size"),
]

# Spazio di ricerca per il modello LIDSTONE
space_lidstone = [
    Categorical(K_PREDICTIONS, name="k"),
    Categorical(DIMENSIONS, name="n"),
    Categorical(BATCH_SIZES, name="batch_size"),
    Categorical(TEST_SIZES, name="test_size"),
    Categorical(GAMMA, name="gamma"),
]

def evaluate_mle(abs, **params) -> float:
    k = params["k"]
    n = params["n"]
    batch_size = params["batch_size"]
    test_size = params["test_size"]

    train_abs, test_abs = split_abs(abs, test_size=test_size)
    lm = train_lm(train_abs, lm_type="MLE", n=n)
    acc = get_topK_accuracy(lm, test_abs, batch_size=batch_size, k_pred=k, n=n)

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
    acc = get_topK_accuracy(lm, test_abs, batch_size=batch_size, k_pred=k, n=n)
    pp = perplexity(lm, test_abs, n=n)

    params["accuracy"] = acc
    params["perplexity"] = pp
    print_LIDSTONE_params_to_csv(params)

    return -acc  # Minimize negative accuracy to maximize accuracy

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
            gp_minimize, evaluate_mle_wrapper, space_mle, n_calls=50, random_state=0
        )
    
        """future_lidstone = executor.submit(
            gp_minimize,
            evaluate_mle_wrapper,
            space_lidstone,
            n_calls=20,
            random_state=0,
        )"""
    
        res_mle = future_mle.result()
        #res_lidstone = future_lidstone.result()

    best_params_mle = res_mle.x
    best_accuracy_mle = -res_mle.fun

    print(f"Best MLE Accuracy: {best_accuracy_mle}")
    print(f"Best MLE Parameters: {best_params_mle}")
    
    """
    best_params_lidstone = res_lidstone.x
    best_accuracy_lidstone = -res_lidstone.fun

    print(f"Best LIDSTONE Accuracy: {best_accuracy_lidstone}")
    print(f"Best LIDSTONE Parameters: {best_params_lidstone}")
    """
if __name__ == "__main__":
    bayesian_optimization()
