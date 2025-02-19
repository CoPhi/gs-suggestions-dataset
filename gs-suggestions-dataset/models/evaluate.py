import re
import os
import pandas as pd
from tqdm import tqdm

from cltk.core.data_types import Doc
from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline

from concurrent.futures import ProcessPoolExecutor, as_completed

from tests.params import (
    get_best_params_LIDSTONE,
    get_best_params_MLE,
    print_LIDSTONE_params_to_csv,
    print_MLE_params_to_csv,
)
from utils.preprocess import clean_text_from_gaps, remove_punctuation, clean_supplements
from models.training import get_sentences, load_abs, load_lm, train_lm
from sklearn.model_selection import KFold
from config.settings import (
    tokenizer,
    sentence_tokenizer,
    K_PRED,
    BATCH_SIZE,
    N,
    LM_TYPE,
    GAMMA,
    TEST_SIZE,
)

def generate_without_padding(context: list[str], lm: LanguageModel, n: int) -> str:
    """
    Genera un token gestendo il caso in cui la generazione restituisca un token di padding

    :param str (context): Contesto iniziale
    :param LanguageModel (lm): Modello di linguaggio
    :param int (n): Dimensione del modello
    :return word (str): Token generato
    """

    # se incontriamo il token di fine frase è inutile continuare a generare
    if context[-1] == "</s>":
        return

    word = lm.generate(text_seed=context[(1 - n) :], num_words=1)
    if word == "<s>":
        return generate_without_padding(context + [word], lm, n)

    return word


def get_dist_freq_words_from_context(
    lm: LanguageModel, context: list[str], n=N
) -> list:
    """
    Genera la distribuzione di frequenze delle parole per un dato contesto.
    Se la distribuzione è vuota, riduce progressivamente il contesto fino a trovare una distribuzione non vuota. (Backoff)

    :param lm: Modello di linguaggio con il metodo `context_counts`
    :param context: Lista di token che rappresenta il contesto
    :param n: Dimensione del modello (numero di parole nel contesto)
    :return: Lista ordinata di tuple (parola, frequenza)
    """
    for i in range(
        n - 1, 0, -1
    ):  # Cerchiamo la distribuzione di frequenza scalando la dimensione del contesto da n-1 a 1
        dist_freq = sorted(
            lm.context_counts(lm.vocab.lookup(context[-(i):])).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        if dist_freq:  # Se troviamo una distribuzione non vuota, restituiamola
            return dist_freq

    return []  # Se nessun contesto ha una distribuzione valida, restituisci lista vuota


def get_k_words_from_context(lm: LanguageModel, context: list[str], n=N,  k=K_PRED) -> list:
    """
    Restituisce le prime k parole più probabili da una distribuzione di frequenza di parole,
    filtrando le parole per '</s>' e '<s>'.

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato per ottenere la distribuzione di frequenza delle parole.
        context (list[str]): Contesto di parole utilizzato per generare la distribuzione di frequenza.
        n (int, opzionale): Numero di parole nel contesto da considerare. Default è N.
        list: Lista delle prime k parole più probabili, escluse '</s>' e '<s>'.

    Returns:
        list: Lista di parole
    """
    words = []
    while len(words) < k:   
        
        dist_freq = get_dist_freq_words_from_context(lm, context, n)
        if not dist_freq:
            break 
            
        for word, _ in  dist_freq:
            if word not in ["</s>", "<s>"] and word not in words:
                words.append(word)
            if len(words) == k:
                break
        
        if len(words) < k:
            n-=1 #Prendo le parole restanti dalle distribuzioni con contesto meno grande
    
    return words


def get_predictions(
    lm: LanguageModel, context: list[str], suppl_words: list[str], n=N, k_pred=K_PRED
):
    """
    Genera previsioni di completamento del testo basate su un modello di linguaggio.

    Il metodo utilizza un contesto dato per ottenere una distribuzione di frequenza delle parole
    e genera previsioni per completare il testo in base alle parole più probabili.

    :param lm: Istanza del modello di linguaggio (`LanguageModel`).
    :param context: Lista di tokens che rappresenta il contesto iniziale.
    :param suppl_words: lista di stringhe (tokens) di riferimento che rappresenta la gold label e la lunghezza desiderata del completamento.
    :param n: Dimensione del modello di n-grammi (default: `N`).
    :param k_pred: Numero massimo di previsioni da generare (default: `K_PRED`).

    :return: Lista di previsioni, dove ogni previsione è una lista di parole generate.

    Il metodo segue questi passi:
    1. Recupera la distribuzione delle parole in base al contesto usando `get_dist_freq_words_from_context`.
    2. Seleziona le `k_pred` parole più probabili.
    3. Per ogni parola, estende il contesto e genera parole successive fino alla lunghezza di `suppl_words`.
    4. Restituisce una lista di liste, dove ogni lista rappresenta una sequenza generata.
    """
    predictions = []
    for word in get_k_words_from_context(
        lm, context, n=n, k=k_pred
    ):  # Prendo le prime k parole più probabili
        
        curr_context = context + [word]
        curr_pred = [word]

        if len(suppl_words) - 1 == 0:
            predictions.append(curr_pred)
        else:
            generated_sequence = []
            while len(generated_sequence) < len(suppl_words) - 1:
                next_token = generate_without_padding(curr_context, lm, n)
                if next_token is None:
                    break

                generated_sequence.append(next_token)
                curr_context.append(
                    next_token
                )  # Aggiungo l'ultimo token generato al contesto

            curr_pred.extend(generated_sequence)
            predictions.append(curr_pred)

    return predictions


def perplexity(lm: LanguageModel, test_abs: list, n=N) -> float:
    """
    Valuta un modello di linguaggio calcolando la perplessità sui dati di test.

    Args:
        lm (LanguageModel): Il modello di linguaggio da valutare.
        test_abs (list): Una lista di blocchi anonimi (abs) di test.
        n (int, opzionale): La dimensione degli n-grammi. Default è N.

    Returns:
        float: La perplessità del modello sui dati di test.

    Raises:
        ValueError: Se il modello non è stato addestrato correttamente.
        RuntimeError: Se si verifica un errore nel calcolo della perplessità.
    """

    test_ngrams, _ = padded_everygram_pipeline(n, get_sentences(test_abs))

    if not lm.vocab:
        raise ValueError("Il modello non è stato addestrato correttamente.")
    try:
        return round(lm.perplexity(flatten(test_ngrams)), 2)
    except ZeroDivisionError:
        raise RuntimeError(
            "Errore nel calcolo della perplessità. Verifica i dati di test e di addestramento."
        )


def get_context_from_test_case(test_case: str, n=N) -> str:
    """
    Forma il contesto da cui il modello linguistico fa partire la generazione della predizione.
    Prende la sottostringa del test_case che parte dall'inizio e finisce con '[', la divide in frasi e le tokenizza, pulendone lacune e punteggiatura. (Contesto a sinistra)

    Args:
        test_case (str) : caso di test del blocco anonimo, in cui è presente un supplemento da generare `[...]`
        n (optional, int): ordine degli ngrammi del modello
    """
    return list(
        flatten(
            [
                list(
                    pad_both_ends(
                        tokenizer.run(
                            input_doc=Doc(raw=remove_punctuation(sent))
                        ).tokens,
                        n=n,
                    )
                )
                for sent in sentence_tokenizer.tokenize(
                    clean_text_from_gaps(
                        re.sub(r"[^\s]+\[", "[", test_case).split("[")[0]
                    )
                )
            ]
        )
    )[: (1 - n)]


def process_batch(lm: LanguageModel, batch: list, n: int, k_pred: int):

    correct_predictions = 0
    total_predictions = 0

    for ab in batch:
        if ab["language"] != "grc":
            continue

        supplements = clean_supplements(ab["training_text"])
        if not supplements:
            continue

        for i, obj in enumerate(ab["test_cases"]):
            if i >= len(supplements):
                break

            test_case = obj["test_case"]
            context = get_context_from_test_case(test_case, n)

            try:
                predictions = get_predictions(lm, context, supplements[i], n, k_pred)
                if supplements[i] in predictions:
                    correct_predictions += 1
            except Exception as e:
                print(f"Errore nella predizione: {e}")

            total_predictions += 1

    return correct_predictions, total_predictions


def process_batch_wrapper(args):
    """Funzione wrapper per rendere il multiprocessing compatibile con pickle."""
    lm, batch, n, k_pred = args
    return process_batch(lm, batch, n, k_pred)


def accuracy_parallel(lm, test_abs, batch_size=BATCH_SIZE, n=N, k_pred=K_PRED):
    correct_predictions = 0
    total_predictions = 0

    batches = [
        (lm, test_abs[i : i + batch_size], n, k_pred)
        for i in range(0, len(test_abs), batch_size)
    ]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_batch_wrapper, batch) for batch in batches]

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Calcolo accuracy",
            unit="ab",
            leave=False,
            disable=True,
        ):
            batch_correct, batch_total = future.result()
            correct_predictions += batch_correct
            total_predictions += batch_total

    return (
        round((correct_predictions / total_predictions) * 100, 2)
        if total_predictions
        else 0.0
    )


def accuracy(
    lm: LanguageModel, test_abs: list, batch_size=BATCH_SIZE, n=N, k_pred=K_PRED
) -> float:
    """
    Calcola l'accuratezza del modello di linguaggio su un dataset di test.

    Args:
        lm (LanguageModel): Il modello di linguaggio da valutare.
        test_abs (list): Lista di blocchi anonimi di testo per il test.
        batch_size (int, opzionale): La dimensione del batch per l'elaborazione. Default è `BATCH_SIZE`.
        n (int, opzionale): Dimensioni degli ngrammi del modello linguistico. Default è `N`.
        k_pred (int, opzionale): Il numero di predizioni da generare per ogni contesto. Default è `K_PRED`.

    Returns:
        float: L'accuratezza del modello espressa in percentuale.
    """
    correct_predictions = 0
    total_predictions = 0

    for start in tqdm(
        range(0, len(test_abs), batch_size),
        desc="Calcolo accuracy",
        unit="ab",
        leave=False,
    ):
        batch = test_abs[start : start + batch_size]  # batch di blocchi anonimi
        for ab in batch:
            if ab["language"] == "grc":
                supplements = clean_supplements(
                    ab["training_text"]
                )  # supplements puliti
                if not supplements:
                    continue  # non ci sono blocchi da predire

                for i, obj in enumerate(ab["test_cases"]):
                    # Per ogni test_case presente nel blocco faccio k-predizioni: se almeno una delle k predizioni è corretta, incremento il contatore delle predizioni corrette
                    # Assumo che i supplements siano ordinati con i test_cases
                    if i >= len(supplements):
                        break

                    test_case = obj["test_case"]
                    context = get_context_from_test_case(test_case, n)
                    predictions = get_predictions(
                        lm, context, supplements[i], n, k_pred
                    )

                    if supplements[i] in predictions:
                        correct_predictions += 1

                    total_predictions += 1

    return round((correct_predictions / total_predictions) * 100, 2)


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

        acc = accuracy(
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

    acc = accuracy(lm, test_abs)
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
