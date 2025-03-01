import re
from tqdm import tqdm

from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline

from tests.params import (
    get_best_params_LIDSTONE,
    get_best_params_MLE,
    print_LIDSTONE_params_to_csv,
    print_MLE_params_to_csv,
)

from utils.preprocess import clean_text_from_gaps, remove_punctuation, clean_supplements, get_tokens_from_clean_text
from models.training import get_sentences, load_abs, load_lm, train_lm
from sklearn.model_selection import KFold
from config.settings import (
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
    if word == "</s>" or word == '<UNK>':
        return

    return word

def generate_sequence(context: list[str], lm: LanguageModel, n: int, length: int) -> list[str]:
    """
    Genera una sequenza di token di una lunghezza specificata a partire da un contesto dato.

    Args:
        context (list[str]): Contesto iniziale per la generazione.
        lm (LanguageModel): Modello di linguaggio utilizzato per la generazione.
        n (int): Dimensione del modello di n-grammi.
        length (int): Lunghezza desiderata della sequenza generata.

    Returns:
        list[str]: Sequenza di token generata.
    """
    generated_sequence = []
    while len(generated_sequence) < length:
        next_token = generate_without_padding(context, lm, n)
        if next_token is None:
            break
        generated_sequence.append(next_token)
        context.append(next_token)
        
    return generated_sequence     

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
    return sorted(
         lm.context_counts(lm.vocab.lookup(context[-(n-1):])).items(),
         key=lambda x: x[1],
         reverse=True,
     )

def get_next_word_from_dist_freqs(lm: LanguageModel, context: list[str], words: list[str], n=N) -> str:
    """
    Restituisce la prossima parola più probabile da una distribuzione di frequenza sulle parole dato un contesto, filtrando se essa non è già presente nella lista di parole words.
    Words contiene una lista di parole che sono già state viste e non devono essere considerate.

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato per ottenere la distribuzione di frequenza delle parole.
        context (list[str]): Contesto di parole utilizzato per generare la distribuzione di frequenza.
        words (list[str]): Lista di parole già generate.
        n (int, opzionale): Numero di parole nel contesto da considerare. Default è N.

    Returns:
        str: La prossima parola più probabile che non è già presente nella lista di parole words.
    """
    while n > 0:
        dist_freq_words_from_context = get_dist_freq_words_from_context(lm, context, n)
        for word, _ in dist_freq_words_from_context:
            if word not in ["</s>", "<s>", '<UNK>'] and word not in words:
                return word
        n -= 1

    return None  # Caso in cui non trovo parole nuove nelle distribuzioni di frequenze del contesto
            
def get_K_predictions(
    lm: LanguageModel, context: list[str], len_suppl_words: int, n=N, k_pred=K_PRED
):
    """
    Genera previsioni di completamento del testo basate su un modello di linguaggio.

    Il metodo utilizza un contesto dato per ottenere le k parole più probabili, dopodiché estende il contesto
    e genera previsioni per completare il testo in base alle parole più probabili.

    :param lm: Istanza del modello di linguaggio (`LanguageModel`).
    :param context: Lista di tokens che rappresenta il contesto iniziale.
    :param suppl_words: lista di stringhe (tokens) di riferimento che rappresenta la gold label e la lunghezza desiderata del completamento.
    :param n: Dimensione del modello di n-grammi (default: `N`).
    :param k_pred: Numero massimo di previsioni da generare (default: `K_PRED`).

    :return: Lista di previsioni, dove ogni previsione è una lista di parole generate.

    Il metodo segue questi passi:
    1. Recupera le k parole più probabili, attraverso `get_k_words_from_context`.
    2. Per ogni parola, estende il contesto e genera parole successive fino alla lunghezza di `suppl_words`.
    3. Restituisce una lista di liste, dove ogni lista rappresenta una sequenza generata.
    """
    predictions = []
    dist_words = []  # Parole trovate nella distribuzione
    max_iterations = k_pred * 2 # Limite massimo di iterazioni
    iterations = 0

    while len(predictions) < k_pred:
        iterations += 1
        word = get_next_word_from_dist_freqs(lm, context, dist_words, n)
        
        if word is None:
            break 
        
        dist_words.append(word) #la inserisco nella lista delle parole trovate per non considerarla più nella generazioni successive
        
        if len_suppl_words == 1:
            if [word] not in predictions:
                predictions.append([word])
                continue

        curr_pred = [word]
        generated_sequence = generate_sequence(context + [word], lm, n, len_suppl_words - 1)
        curr_pred.extend(generated_sequence)

        if len(curr_pred) == len_suppl_words:
            if curr_pred not in predictions:    
                predictions.append(curr_pred)
                continue
            
        if iterations >= max_iterations: # Se non riesco a generare k-predizioni di lunghezza corretta, esco
            break
        
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

def get_context(text: str, n=N) -> list[str]: 
    """
        Ritorna una lista di token da passare al modello liguistico per l'inferenza
        Args: 
            text (str): Testo da tokenizzare
        Returns:
            list[str]: Lista di token, rappresenta il contesto da cui partire per l'inferenza del modello
    """
    return list(
        flatten(
                list(
                    pad_both_ends(
                        get_tokens_from_clean_text(
                            remove_punctuation(
                                sent
                                )
                        ),
                        n=n,
                    )
                )
                for sent in sentence_tokenizer.tokenize(
                    clean_text_from_gaps(text)
                    )
                )
        )[: (1 - n)]


def get_context_from_test_case(test_case: str, n=N) -> list[str]:
    """
    Forma il contesto da cui il modello linguistico fa partire la generazione della predizione.
    Prende la sottostringa del test_case che parte dall'inizio e finisce con '[', la divide in frasi e le tokenizza, pulendone lacune e punteggiatura. (Contesto a sinistra)
    Questo metodo viene usato per generare il contesto dal test_case da cui partire per la generazione della predizione nel metodo `accuracy()`.

    Args:
        test_case (str) : caso di test del blocco anonimo, in cui è presente un supplemento da generare `[...]`
        n (optional, int): ordine degli ngrammi del modello
        
    Returns: 
        list[str]: contesto da cui partire per generare la predizione
    """
    return list(
        flatten(
            [
                list(
                    pad_both_ends(
                        get_tokens_from_clean_text(
                            remove_punctuation(
                                sent
                                )
                        ),
                        n=n,
                    )
                )
                for sent in sentence_tokenizer.tokenize(
                    clean_text_from_gaps(
                        re.sub(r"[^\s]+\[", "[", test_case).split("[")[0] #Si prende il contesto a sinistra della parentesi `[`
                    )
                )
            ]
        )
    )[: (1 - n)]   

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
        desc="Accuracy",
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
                    # Assumo che i supplements siano ordinati con i test_cases
                    if i >= len(supplements):
                        break

                    test_case = obj["test_case"]
                    context = get_context_from_test_case(test_case, n)
                    predictions = get_K_predictions(
                        lm, context, len(supplements[i]), n, k_pred
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
