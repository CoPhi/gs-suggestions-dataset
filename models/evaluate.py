import heapq
import re
from math import inf, log
from tqdm import tqdm
from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.metrics.distance import edit_distance

from tests.params import (
    get_best_params_LIDSTONE,
    get_best_params_MLE,
    print_LIDSTONE_params_to_csv,
    print_MLE_params_to_csv,
)

from utils.preprocess import (
    clean_text_from_gaps,
    remove_punctuation,
    clean_supplements,
    get_tokens_from_clean_text,
)
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
    if word == "</s>" or word == "<UNK>":
        return

    return word


def generate_sequence(
    context: list[str], lm: LanguageModel, n: int, length: int
) -> list[str]:
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
        lm.context_counts(lm.vocab.lookup(context[-(n - 1) :])).items(),
        key=lambda x: x[1],
        reverse=True,
    )


def get_next_word_from_dist_freqs(
    lm: LanguageModel, context: list[str], words: set, n=N
) -> str:
    """
    Restituisce la prossima parola più probabile da una distribuzione di frequenza sulle parole dato un contesto, filtrando se essa non è già presente nella lista di parole words.
    Words contiene una lista di parole che sono già state viste e non devono essere considerate.

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato per ottenere la distribuzione di frequenza delle parole.
        context (list[str]): Contesto di parole utilizzato per generare la distribuzione di frequenza.
        words (set): Insieme di parole già generate.
        n (int, opzionale): Numero di parole nel contesto da considerare. Default è N.

    Returns:
        str: La prossima parola più probabile che non è già presente nella lista di parole words.
    """
    while n > 0:
        dist_freq_words_from_context = get_dist_freq_words_from_context(lm, context, n)
        for word, _ in dist_freq_words_from_context:
            if word not in ["</s>", "<s>", "<UNK>"] and word not in words:
                return word
        n -= 1

    return None  # Caso in cui non trovo parole nuove nelle distribuzioni di frequenze del contesto


def get_pred_sorted_by_edit_distance(
    predictions: list[list[str]], gold_label: str
) -> list[str]:
    """
    Ordina una lista di previsioni in base alla edit distance rispetto a una etichetta di riferimento.
    Si usa la distanza di Levenstein implementata da NLTK

    Args:
        predictions (list[list[str]]): Una lista di previsioni, dove ogni previsione è una lista di `token`.
        gold_label (str): La stringa di riferimento con cui confrontare le previsioni.

    Returns:
        list[str]: La lista di previsioni ordinate in base alla edit distance rispetto all'etichetta di riferimento.
    """
    return sorted(
        predictions,
        key=lambda x: edit_distance(" ".join(x), gold_label),
    )


def get_K_predictions(
    lm: LanguageModel, context: list[str], suppl_words: list[str], n=N, k_pred=K_PRED
) -> list[list[str]]:
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
    1. Recupera le k parole più probabili, attraverso `get_words_from_context`.
    2. Per ogni parola, estende il contesto e genera parole successive fino alla lunghezza di `suppl_words`.
    3. Restituisce una lista di liste, dove ogni lista rappresenta una sequenza generata.
    """
    predictions = []
    dist_words = []  # Parole trovate nella distribuzione
    max_iterations = k_pred * 5  # Limite massimo di iterazioni
    iterations = 0

    while True:
        iterations += 1
        word = get_next_word_from_dist_freqs(lm, context, dist_words, n)

        if word is None:
            break

        dist_words.append(
            word
        )  # la inserisco nella lista delle parole trovate per non considerarla più nella generazioni successive

        if len(suppl_words) == 1:
            if [word] not in predictions:
                predictions.append([word])

        curr_pred = [word]
        generated_sequence = generate_sequence(
            context + [word], lm, n, len(suppl_words) - 1
        )
        curr_pred.extend(generated_sequence)

        if len(curr_pred) == len(suppl_words):
            if curr_pred not in predictions:
                predictions.append(curr_pred)

        if (
            iterations >= max_iterations
        ):  # Se non riesco a generare k-predizioni di lunghezza corretta, esco
            break

    return get_pred_sorted_by_edit_distance(predictions, " ".join(suppl_words))[
        :k_pred
    ]  # Ritorno le k-predizioni basate sull'euristica edit distance


def get_best_K_predictions_from_context(
    lm: LanguageModel,
    context: list[str],
    len_suppl: int = None,
    suppl_words: list[str] = None,
    n: int = N,
    k_pred: int = K_PRED,
    beam_size: int = K_PRED * 4,
    alpha: float = 0,
    beta: float = 1,
    mod: str = "acc",
) -> list[list[str]]:
    """
    Calcola le migliori K predizioni basate sulla probabilità dell'intera sequenza generata (contesto + generazione) utilizzando un algoritmo di ricerca locale (local beam search).
    Gli stati dell'albero sono la generazione corrente: i k stati iniziali vengono generati dal contesto passato per parametro.
    La funzione di perdita è espressa in termini della probabilità di prevedere una sequenza, assegnata dal modello: `1 - p(seq)`, aggiungendo un punteggio che riguarda la edit distance con le altre predizioni.
    I parametri alpha e beta sono i pesi dei punteggi della loss e della edit distance, e sono regolabili.
    Args:
        lm (LanguageModel): Il modello di linguaggio utilizzato per calcolare le probabilità delle sequenze.
        context (list[str]): Il contesto iniziale da cui partire per generare le predizioni.
        len_suppl (int): La lunghezza massima della sequenza generata.
        n (int, opzionale): Il numero di parole da considerare per la generazione. Default è N.
        k_pred (int, opzionale): Il numero di predizioni da mantenere nel beam. Default è K_PRED.
        beam_size (int, opzionale): La dimensione massima del beam. Default è K_PRED*5.
        alpha (float, opzionale): Peso del punteggio della loss nella funzione di perdita. Default è 1.
        beta (float, opzionale): Peso del punteggio della edit distance nella funzione di perdita. Default è 0.5.
        mod (str, opzionale): Modalità di calcolo delle predizioni. Default è "acc".
    Returns:
        list[list[str]]: Una lista delle migliori K sequenze generate, ordinate per combinazione migliore tra loss ed edit distance.
    """

    def loss(
        lm: LanguageModel, context: tuple, generated_seq: list[str], lm_type=LM_TYPE
    ) -> float:
        """
        Funzione di perdita usata nella local beam search.
        """

        def compute_log_prob(word: str, context: tuple) -> float:
            """
            Calcola la log-probabilità di una parola data il contesto.
            Restituisce -inf se la probabilità è zero.
            """
            prob = lm.score(word, context)
            return log(prob) if prob > 0 else -inf

        def update_context(context: tuple, word: str) -> tuple:
            """
            Aggiorna il contesto includendo la parola generata.
            """
            return context[1:] + (word,)

        total_log_prob = 0
        for word in generated_seq:
            total_log_prob += compute_log_prob(word, context)
            context = update_context(context, word)

        if lm_type == "MLE":
            return (
                -total_log_prob if total_log_prob != -inf else 100
            )  # Si aggiunge un valore di fallback nel caso in cui le parole della sequenza non siano presenti nel modello
            
        return -total_log_prob

    def get_words_from_context(
        lm: LanguageModel, context: list[str], boundary: int, n: int = N
    ) -> list[str]:
        """
        Funzione con cui espandiamo le K parole possibili successive da uno stato
        """
        k_words = []
        dist_words = (
            set()
        )  # Parole già viste nelle distribuzioni di parole date dal contesto

        while len(k_words) < boundary:
            next_w = get_next_word_from_dist_freqs(lm, context, dist_words, n)
            if next_w is None:
                return k_words  # Non ci sono più parole disponibili

            dist_words.add(next_w)
            k_words.append(next_w)

        return k_words

    def avg_edit_distance(seq: str, others: list[str]) -> float:
        """
        Calcola la distanza di Levensthein media tra una sequenza e le altre presenti nel beam
        """
        return sum(
            edit_distance(" ".join(seq), " ".join(other))
            for other in others
            if other != seq
        ) / max(len(others) - 1, 1)

    def get_best_candidates_from_beam(
        beam: list[tuple[list[str], float]],
    ) -> list[list[str]]:
        """
        Restituisce i candidati ordinati per combinazione migliore tra loss ed edit distance con gli altri candidati
        """
        if not beam:
            return []

        sorted_candidates = sorted(beam, key=lambda x: x[1])
        string_candidates = [candidate[0] for candidate in sorted_candidates]

        if mod == "infer":
            distances = {
                tuple(seq): avg_edit_distance(seq, string_candidates)
                for seq in string_candidates
            }
        elif mod == "acc":
            if not suppl_words:
                raise ValueError("Define the gold label")
            distances = {
                tuple(seq): edit_distance(" ".join(seq), " ".join(suppl_words))
                for seq in string_candidates
            }
        else:
            raise ValueError("Cannot value distances: mod is not well defined")

        ranking_candidates = sorted(
            [
                (
                    candidate[0],
                    (alpha * candidate[1] + beta * distances[tuple(candidate[0])]),
                )
                for candidate in sorted_candidates
            ],
            key=lambda x: x[1],
        )
        
        return [candidate[0] for candidate in ranking_candidates][:k_pred]

    def get_successors(
        lm: LanguageModel, context: list[str], candidate: tuple[list[str], float]
    ) -> list[tuple[list[str], float]]:
        successors = []
        nexts = get_words_from_context(lm, context + candidate[0], beam_size)
        for w in nexts:
            successors.append(
                (
                    candidate[0] + [w],
                    loss(
                        lm,
                        lm.vocab.lookup(context[(1 - n) :]),
                        candidate[0] + [w],
                    ),
                )
            )
        return successors

    def local_beam_search(lm: LanguageModel, context: list[str]) -> list[list[str]]:

        beam = []
        # Inizializzazione K stati iniziali
        states = get_words_from_context(lm, context, beam_size)
        for s in states:
            heapq.heappush(
                beam, ([s], loss(lm, lm.vocab.lookup((context)[(1 - n) :]), [s]))
            )

        while True:
            successors = []
            # Generiamo tutti i successori degli stati presenti nel beam
            for candidate in beam:
                if (
                    len(candidate[0]) >= len_suppl
                ):  # Termina se la lunghezza della sequenza raggiunge len_suppl : le altre sequenze del beam hanno la solita lunghezza
                    continue

                successors.extend(get_successors(lm, context, candidate))

            if not successors:
                break  # Nessun successore

            beam = heapq.nsmallest(beam_size, successors, key=lambda x: x[1])

        return get_best_candidates_from_beam(beam)

    return local_beam_search(lm, context)


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
                    get_tokens_from_clean_text(remove_punctuation(sent)),
                    n=n,
                )
            )
            for sent in sentence_tokenizer.tokenize(clean_text_from_gaps(text))
        )
    )[: (1 - n)]


def get_context_from_test_case(test_case: str, n=N) -> list[str]:
    """
    Forma il contesto da cui il modello linguistico fa partire la generazione della predizione.
    Prende la sottostringa del test_case che parte dall'inizio e finisce con '[', la divide in frasi e le tokenizza, pulendone lacune e punteggiatura. (Contesto a sinistra)
    Questo metodo viene usato per generare il contesto dal test_case da cui partire per la generazione della predizione nel metodo `get_topK_accuracy()`.

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
                        get_tokens_from_clean_text(remove_punctuation(sent)),
                        n=n,
                    )
                )
                for sent in sentence_tokenizer.tokenize(
                    clean_text_from_gaps(
                        re.sub(r"[^\s]+\[", "[", test_case).split("[")[
                            0
                        ]  # Si prende il contesto a sinistra della parentesi `[`
                    )
                )
            ]
        )
    )[: (1 - n)]


def get_topK_accuracy(
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

                    if not supplements[i]: #Supplementi non presenti nel testo 
                        continue
                    
                    if '<UNK>' in supplements[i]: #significa che sono presenti dei token 'None' nei supplementi e non posso fare inferenza
                        continue

                    context = get_context_from_test_case(obj["test_case"], n)
                    
                    predictions = get_best_K_predictions_from_context(
                        lm=lm,
                        context=context,
                        len_suppl=len(supplements[i]),
                        suppl_words=supplements[i],
                        n=n,
                        k_pred=k_pred,
                        mod="acc",
                    )

                    # print (len(supplements[i]), ':', predictions, ": ", len(predictions),'\n')

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
