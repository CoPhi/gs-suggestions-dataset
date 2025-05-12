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

"""def KFold_cross_validation(k=10):
    Funzione che esegue la cross validation K-Fold per valutare il modello di linguistico su tutto il dataset.
    Addestra e valuta il modello nelle varie fold usando gli iperparametri che restituiscono la migliore accuracy, secondo l'ottimizzazione
    degli iperparametri (Bayesian Optimization).

    Args:
        k (int): Numero di fold per la cross validation. Default è 10 (10-Fold cross validation).
    
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

"""

if __name__ == "__main__":
    g_lm, test_abs = load_lm("General_model")  # carico il modello generico 
    d_lm, _ = load_lm("Domain_model")  # carico il modello specifico di dominio 
    
    # KFold_cross_validation()

    print("Perplexity: ", perplexity(g_lm,d_lm, test_abs))
    print("Accuracy: ", get_topK_accuracy(g_lm, d_lm, test_abs))
