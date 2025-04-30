from metrics.accuracy import get_topK_accuracy
from train.training import pipeline_train
from config.settings import (
    K_PREDICTIONS,
    DIMENSIONS,
    TEST_SIZES,
    GAMMAS,
    MIN_FREQS,
    BATCH_SIZE,
)
from tuning import save_results, save_results_pickle
from bohb import BOHB
import bohb.configspace as cs 

# Definizione spazi di ricerca discreti

k = cs.CategoricalHyperparameter("k", K_PREDICTIONS)
n = cs.CategoricalHyperparameter("n", DIMENSIONS)
test_size = cs.CategoricalHyperparameter("test_size", TEST_SIZES)
min_freq = cs.CategoricalHyperparameter("min_freq", MIN_FREQS)
gamma = cs.CategoricalHyperparameter("gamma", GAMMAS)
param_space_mle = cs.ConfigurationSpace([k, n, test_size, min_freq])
param_space_lidstone = cs.ConfigurationSpace([k, n, test_size, min_freq, gamma])


# Funzione obiettivo per Lidstone
def objective_function_lidstone(config, budget):
    k = config["k"]
    n = config["n"]
    test_size = config["test_size"]
    min_freq = config["min_freq"]
    gamma = config["gamma"]
    model, val = pipeline_train(
        lm_type="LIDSTONE", gamma=gamma, min_freq=min_freq, n=n, test_size=test_size, budget=budget
    )
    accuracy = get_topK_accuracy(model, val, BATCH_SIZE, n, k)
    return -accuracy


def objective_function_mle(config, budget):
    k = config["k"]
    n = config["n"]
    test_size = config["test_size"]
    min_freq = config["min_freq"]
    model, val = pipeline_train(
        lm_type="MLE", min_freq=min_freq, n=n, test_size=test_size, budget=budget
    )
    accuracy = get_topK_accuracy(model, val, BATCH_SIZE, n, k)
    return -accuracy


def BOHB_mle(max_budget=100, min_budget=10, num_proc=1):
    optimizer = BOHB(
        param_space_mle, objective_function_mle, max_budget=max_budget, min_budget=min_budget, n_proc=num_proc	
    )
    opt_log = optimizer.optimize()

    print("Best configuration found:")
    print("Best accuracy:", -opt_log.best["loss"])
    print("Best hyperparameters")
    print(opt_log.best["hyperparameter"])
    
    save_results(opt_log.logs, "MLE_results.csv", is_lidstone=False)
    save_results_pickle(opt_log, "opt_log_results_mle.pkl")

def BOHB_lidstone(max_budget=100, min_budget=10, num_proc=1):
    optimizer = BOHB(
        param_space_lidstone, objective_function_lidstone, max_budget=max_budget, min_budget=min_budget, n_proc=num_proc
    )
    opt_log = optimizer.optimize()

    print("Best configuration found:")
    print("Best accuracy:", -opt_log.best["loss"])
    print("Best hyperparameters")
    print(opt_log.best["hyperparameter"])
    
    save_results(opt_log.logs, "LIDSTONE_results.csv", is_lidstone=True)
    save_results_pickle(opt_log, "opt_log_results_lidstone.pkl")

if __name__ == "__main__":

    # Esegui BOHB per Lidstone
    BOHB_lidstone(max_budget=10, min_budget=10, num_proc=1)

    # Esegui BOHB per MLE
    #BOHB_mle(max_budget=40, min_budget=10, num_proc=1)
