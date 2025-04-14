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
def objective_function_lidstone(config):
    k = config["k"]
    n = config["n"]
    test_size = config["test_size"]
    min_freq = config["min_freq"]
    gamma = config["gamma"]
    model, val = pipeline_train(
        lm_type="LIDSTONE", gamma=gamma, min_freq=min_freq, n=n, test_size=test_size
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


def BHOB_mle(max_budget=100, min_budget=10, num_proc=1):
    optimizer = BOHB(
        param_space_mle, objective_function_mle, max_budget=max_budget, min_budget=min_budget, n_proc=num_proc	
    )  # I budget indicano le percentuali di blocchi presi dalla collezione di dati in possesso
    opt_log = optimizer.optimize()
    # Save all results to a CSV file
    print("Best configuration found:")
    print("Best accuracy:", -opt_log.best["loss"])
    print("Best hyperparameters")
    print(opt_log.best["hyperparameter"])
    
    with open("MLE_results.csv", "w") as f:
        f.write("budget, k_predictions, dimension, test_size, min_frequence, accuracy\n")
        for log in opt_log.logs:
            for budget in log:
                hyperparameters = log[budget]["hyperparameter"].to_dict()
                f.write(
                f"{budget},{hyperparameters['k']},{hyperparameters['n']},{hyperparameters['test_size']},{hyperparameters['min_freq']},{-log[budget]['loss']}\n"
            )
            
            


def BHOB_lidstone(max_budget=100, min_budget=10, num_proc=1):
    # Create the BOHB instance
    optimizer = BOHB(param_space_lidstone, objective_function_lidstone, max_budget=max_budget, min_budget=min_budget, n_proc=num_proc)
    # Run the optimization
    opt_log = optimizer.optimize()
    print("Best configuration found:")
    print("Best accuracy:", -opt_log.best["loss"])
    print("Best hyperparameters")
    print(opt_log.best["hyperparameter"])
    
    with open("LIDSTONE_results.csv", "w") as f:
        f.write("k_predictions, dimension, test_size, min_frequence, gamma, accuracy\n")
        for log in opt_log.logs:
            for budget in log:
                hyperparameters = log[budget]["hyperparameter"].to_dict()
                f.write(
                f"{hyperparameters['k']},{hyperparameters['n']},{hyperparameters['test_size']},{hyperparameters['min_freq']}, {hyperparameters['gamma']}, {-log[budget]['loss']}\n"
            )


if __name__ == "__main__":

    # Esegui BOHB per MLE
    BHOB_mle(max_budget=100, min_budget=50, num_proc=1)

    # Esegui BOHB per Lidstone
    # BHOB_lidstone()
