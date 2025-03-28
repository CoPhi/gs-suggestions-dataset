from tests.params import (
    get_best_params_LIDSTONE,
    get_best_params_MLE,
    print_LIDSTONE_params_to_csv,
    print_MLE_params_to_csv,
)

from sklearn.model_selection import KFold
from train import load_abs, load_lm
from train.training import train_lm
from config.settings import (
    K_PRED,
    BATCH_SIZE,
    N,
    LM_TYPE,
    GAMMA,
    TEST_SIZE,
)

from metrics import get_topK_accuracy, perplexity

def KFold_cross_validation(k=10):
    """
    Funzione che esegue la cross validation K-Fold per valutare il modello di linguistico su tutto il dataset.
    Addestra e valuta il modello nelle varie fold usando gli iperparametri che restituiscono la migliore accuracy, secondo l'ottimizzazione
    degli iperparametri (Bayesian Optimization).

    Args:
        k (int): Numero di fold per la cross validation. Default è 10 (10-Fold cross validation).
    """
    abs = load_abs()  # dataset
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    accuracies = []
    perplexities = []

    # Recupero degli iperparametri migliori
    if LM_TYPE == "LIDSTONE":
        best_params = get_best_params_LIDSTONE()
    else:
        best_params = get_best_params_MLE()

    for train_index, test_index in kf.split(abs):
        train_abs = [abs[i] for i in train_index]
        test_abs = [abs[i] for i in test_index]

        if LM_TYPE == "LIDSTONE":
            lm = train_lm(
                train_abs, gamma=best_params["GAMMA"], n=best_params["DIMENSION"]
            )  # carico il modello con i dati di train
        else:
            lm = train_lm(train_abs, n=best_params["DIMENSION"])

        acc = get_topK_accuracy(
            lm,
            test_abs,
            batch_size=best_params["BATCH_SIZE"],
            n=best_params["DIMENSION"],
            k_pred=best_params["K_PRED"],
        )
        accuracies.append(acc)

        if LM_TYPE == "LIDSTONE":
            pp = perplexity(lm, test_abs, n=best_params["DIMENSION"])
            perplexities.append(pp)

    avg_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average Accuracy: {avg_accuracy}")

    if LM_TYPE == "LIDSTONE":
        avg_perplexity = sum(perplexities) / len(perplexities)
        print(f"Average Perplexity: {avg_perplexity}")


if __name__ == "__main__":
    lm, test_abs = load_lm()  # carico il modello
    # KFold_cross_validation()

    print("Perplexity: ", perplexity(lm, test_abs))
    print("Accuracy: ", get_topK_accuracy(lm, test_abs))

    """acc = get_topK_accuracy(lm, test_abs)
    if LM_TYPE == "LIDSTONE":
        pp = perplexity(lm, test_abs)
        print("Perplexity: ", pp)
        print("Accuracy: ", acc)
        print_LIDSTONE_params_to_csv(
            {
                "K_PRED": K_PRED,
                "TEST_SIZE": TEST_SIZE,
                "DIMENSION": N,
                "BATCH_SIZE": BATCH_SIZE,
                "GAMMA": GAMMA,
                "ACCURACY": acc,
                "PERPLEXITY": pp,
            }
        )

    if LM_TYPE == "MLE":
        print("Accuracy: ", acc)
        print_MLE_params_to_csv(
            {
                "K_PRED": K_PRED,
                "DIMENSION": N,
                "BATCH_SIZE": BATCH_SIZE,
                "TEST_SIZE": TEST_SIZE,
                "ACCURACY": acc,
            }
        )
"""
