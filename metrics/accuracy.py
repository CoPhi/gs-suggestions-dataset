import heapq
import re
import unicodedata
from math import inf, log
from typing import Optional
from tqdm import tqdm
from metrics import FALLBACK_LOSS
from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.metrics.distance import edit_distance

from utils.preprocess import (
    clean_text_from_gaps,
    remove_punctuation,
    clean_supplements,
    get_head_supplement,
    get_tail_supplement,
    get_tokens_from_clean_text,
    test_case_contains_lacuna,
)

from metrics import sentence_tokenizer
from config.settings import (
    K_PRED,
    BATCH_SIZE,
    N,
    LM_TYPE,
)


def get_dist_freq_words_from_context(
    lm: LanguageModel,
    context: list[str],
    n=N,
    head: Optional[str] = None,
    tail: Optional[str] = None,
) -> list[str]:
    """
    Genera la distribuzione di frequenze delle parole per un dato contesto.

    Args:
        lm (LanguageModel): Modello di linguaggio con il metodo `context_counts`.
        context (list[str]): Lista di token che rappresenta il contesto.
        n (int): Dimensione del modello (numero di parole nel contesto).
        head (Optional[str]): Se specificato, filtra solo le parole che iniziano con questa sottostringa.
        tail (Optional[str]): Se specificato, filtra solo le parole che finiscono con questa sottostringa.

    Returns:
        list[str]: Lista di parole ordinate in base alla testa, coda e frequenza.
    """

    def filter_words(df):
        """Filtra parole non valide come token speciali."""
        return [w for w, _ in df if w not in ["</s>", "<s>", "<UNK>"]]

    def sort_and_filter(df, condition, key_func):
        """Ordina e filtra le parole in base a una condizione e una funzione di ordinamento."""
        target = sorted(
            [(w, f) for w, f in df if condition(w)],
            key=key_func,
        )
        remaining = sorted(
            [(w, f) for w, f in df if w not in dict(target)],
            key=lambda x: x[1],
            reverse=True,
        )
        return filter_words(target + remaining)

    if n <= 1:
        return []

    df = lm.context_counts(lm.vocab.lookup(context[-(n - 1) :])).items()

    if head and tail:
        return sort_and_filter(
            df,
            lambda w: len(w) >= (len(head) + len(tail)),
            lambda x: edit_distance(x[0][: len(head)], head)
            + edit_distance(x[0][-len(tail) :], tail),
        )
    elif head:
        return sort_and_filter(
            df,
            lambda w: len(w) >= len(head),
            lambda x: edit_distance(x[0][: len(head)], head),
        )
    elif tail:
        return sort_and_filter(
            df,
            lambda w: len(w) >= len(tail),
            lambda x: edit_distance(x[0][-len(tail) :], tail),
        )
    else:
        # Ordina solo in base alla frequenza se non ci sono informazioni su testa o coda
        return filter_words(
            sorted(
                df,
                key=lambda x: x[1],
                reverse=True,
            )
        )


def get_words_from_context(
    lm: LanguageModel,
    context: list[str],
    boundary: int,
    head: Optional[str] = None,
    tail: Optional[str] = None,
    n: int = N,
) -> list[str]:
    """
    Funzione con cui espandiamo le parole possibili successive da uno stato (il numero massimo consentito è rappresentato da `boundary`).
    Se la distribuzione è vuota, riduce progressivamente il contesto fino a trovare una distribuzione non vuota. (Backoff)

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato.
        context (list[str]): Contesto di parole.
        boundary (int): Numero massimo di parole da generare.
        head (str , opzionale): Testa del supplemento, utile per far virare la ricerca verso token più rappresentativi
        tail (str , opzionale): Coda del suggerimento
        n (int, opzionale): Numero di parole nel contesto. Default è N.

    Returns:
        list[str]: Lista di parole generate.
    """
    tokens = []
    dist_words = (
        set()
    )  # Parole già viste nelle distribuzioni di parole date dal contesto

    # print (f"Contesto: {context}")
    # Se il contesto è vuoto, esco

    while len(tokens) < boundary:
        # Se non ci sono più parole disponibili, esco
        if n <= 1:
            break

        for next_w in get_dist_freq_words_from_context(lm, context, n, head, tail):

            if next_w in dist_words:
                continue

            dist_words.add(next_w)  # La aggiungo alla lista delle parole già viste
            tokens.append(next_w)
        n -= 1

    return tokens


def loss(
    lm: LanguageModel, context: tuple, generated_seq: list[str], lm_type=LM_TYPE
) -> float:
    """
    Funzione di perdita usata nella local beam search.
    Tale implementazione fa riferimento alla metrica NLL (Negative Log Likelihood) normalizzata per la lunghezza della frase complessiva (contesto + generazione).

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato.
        context (tuple): Contesto di parole.
        generated_seq (list[str]): Sequenza di parole generata.
        lm_type (str, opzionale): Tipo di modello di linguaggio. Default è LM_TYPE.

    Returns:
        float: Valore della funzione di perdita.
    """

    def compute_log_prob(word: str, context: tuple) -> float:
        """
        Calcola la log-probabilità di una parola dato il contesto.
        Restituisce -inf se la probabilità è zero.

        Args:
            word (str): Parola di cui calcolare la probabilità.
            context (tuple): Contesto di parole.

        Returns:
            float: Log-probabilità della parola.
        """
        prob = lm.score(word, context)
        return log(prob) if prob > 0 else -inf

    def update_context(context: tuple, word: str) -> tuple:
        """
        Aggiorna il contesto includendo la parola generata.

        Args:
            context (tuple): Contesto di parole.
            word (str): Parola da aggiungere al contesto.

        Returns:
            tuple: Nuovo contesto aggiornato.
        """
        return context[1:] + (word,)

    total_log_prob = 0
    for word in generated_seq:
        total_log_prob += compute_log_prob(word, context)
        context = update_context(context, word)

    if lm_type == "MLE":
        return (
            -(total_log_prob / len(generated_seq))
            if total_log_prob != -inf
            else FALLBACK_LOSS
        )  # Si aggiunge un valore di fallback nel caso in cui le parole della sequenza non siano presenti nel modello

    return -total_log_prob


"""
def pmi(context: tuple, word: str, lm: LanguageModel, alpha: float) -> float:
    
    Calcola la PMI (Pointwise Mutual Information) tra una parola e un contesto (PMI condizionata).

    Args:
        context (tuple): Contesto di parole.
        word (str): Parola da analizzare.
        lm (LanguageModel): Modello di linguaggio utilizzato.

    Returns:
        float: Valore della PMI.
    
    return log(
        lm.score((word,) + context) / (lm.score(word) * (pow(lm.score(context), alpha)))
    )  # PMI condizionata


def ppmi(context: tuple, word: str, lm: LanguageModel, alpha: float = 0.75) -> float:
    
    Calcola la PPMI (Positive Pointwise Mutual Information) tra una parola e un contesto.
    Alpha rappresenta il peso che normalizza le probabilità delle parole più rare con quelle più frequenti.

    Args:
        context (tuple): Contesto di parole.
        word (str): Parola da analizzare.
        lm (LanguageModel): Modello di linguaggio utilizzato.

    Returns:
        float: Valore della PPMI.
    
    return max(0, pmi(context, word, lm, alpha))  # PPMI = max(0, PMI)
"""


def avg_edit_distance(seq: str, others: list[str]) -> float:
    """
    Calcola la distanza di Levensthein media tra una sequenza e le altre presenti nel beam.

    Args:
        seq (str): Sequenza di parole.
        others (list[str]): Altre sequenze di parole.

    Returns:
        float: Distanza media di Levensthein.
    """
    return sum(
        edit_distance(" ".join(seq), " ".join(other))
        for other in others
        if other != seq
    ) / max(len(others) - 1, 1)


def get_best_candidates_from_beam(
    beam: list[tuple[list[str], float]],
    suppl_words: list[str] = None,
    mod: str = "acc",
    alpha: float = 0,
    beta: float = 1,
    k_pred: int = K_PRED,
) -> list[list[str]]:
    """
    Restituisce i candidati ordinati per combinazione migliore tra loss ed edit distance con gli altri candidati.

    Args:
        beam (list[tuple[list[str], float]]): Lista di candidati con le rispettive perdite.
        suppl_words (list[str], opzionale): Parole supplementari. Default è None.
        mod (str, opzionale): Modalità di calcolo delle distanze. Default è "acc".
        alpha (float, opzionale): Peso del punteggio della loss. Default è 0.
        beta (float, opzionale): Peso del punteggio della edit distance. Default è 1.
        k_pred (int, opzionale): Numero di predizioni da generare. Default è K_PRED.

    Returns:
        list[list[str]]: Lista delle migliori sequenze generate.
    """
    if not beam:
        return []

    sorted_candidates = sorted(beam, key=lambda x: x[1])
    string_candidates = [candidate[0] for candidate in sorted_candidates]

    if mod == "infer":
        distances = {tuple(seq): score for seq, score in sorted_candidates}
    elif mod == "acc":
        if not suppl_words:
            raise ValueError("Definire la gold label")
        distances = {
            tuple(seq): edit_distance(" ".join(seq), " ".join(suppl_words))
            for seq in string_candidates
        }
    else:
        raise ValueError("Modalità non definita correttamente")

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
    lm: LanguageModel,
    lm_type: str,
    context: list[str],
    candidate: tuple[list[str], float],
    beam_size: int = K_PRED * 2,
    len_suppl_words: int = 0,
    head: Optional[str] = None,
    tail: Optional[str] = None,
    n: int = N,
) -> list[tuple[list[str], float]]:
    """
    Genera i successori di un candidato nel beam.

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato.
        context (list[str]): Contesto di parole.
        candidate (tuple[list[str], float]): Candidato corrente.
        beam_size (int, opzionale): Dimensione del beam. Default è K_PRED * 4.
        n (int, opzionale): Numero di parole nel contesto. Default è N.

    Returns:
        list[tuple[list[str], float]]: Lista di successori generati.
    """
    successors = []
    nexts = get_words_from_context(lm, context + candidate[0], beam_size, None, tail, n)
    for w in nexts:
        successors.append(
            (
                candidate[0] + [w],
                score_candidate(
                    lm,
                    lm_type,
                    lm.vocab.lookup(context[(1 - n) :]),
                    candidate[0] + [w],
                    head,
                    tail,
                    len_suppl_words,
                ),
            )
        )
    return successors


def score_candidate(
    lm: LanguageModel,
    lm_type: str,
    context: tuple,
    candidate: list[str],
    head: Optional[str] = None,
    tail: Optional[str] = None,
    len_suppl_words: int = 0,
) -> float:
    """
    Calcola il punteggio di un candidato in base alla loss e alla edit distance con la testa e la coda del supplemento.

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato.
        context (list[str]): Contesto di parole.
        candidate (tuple): Candidato da valutare.
        head (Optional[str], opzionale): Testa del supplemento. Default è None.
        tail (Optional[str], opzionale): Coda del supplemento. Default è None.
        len_suppl_words (int, opzionale): Lunghezza del supplemento. Default è 0.

    Returns:
        float: Punteggio calcolato per il candidato.
    """

    def calculate_edit_distance(segment: str, reference: str) -> float:
        return edit_distance(segment, reference)

    def apply_length_penalty(
        candidate: list[str],
        head: Optional[str],
        tail: Optional[str],
        len_suppl_words: int,
        weight: float = 0.75,
    ) -> float:
        """
        Applica una penalità basata sulla lunghezza del candidato rispetto alla testa e alla coda del supplemento.

        Args:
            candidate (list[str]): Lista di token del candidato.
            head (Optional[str]): Testa del supplemento.
            tail (Optional[str]): Coda del supplemento.
            len_suppl_words (int): Lunghezza del supplemento.
            weight (float, opzionale): Peso della penalità. Default è 0.75.

        Returns:
            float: Fattore di penalità da moltiplicare allo score.
        """
        penalty = 0
        if len_suppl_words == 1:
            if head and tail:
                if len(candidate[0]) < (len(head) + len(tail)):
                    penalty += ((len(head) + len(tail)) - len(candidate[0])) / weight
            elif head:
                if len(candidate[0]) < len(head):
                    penalty += (len(head) - len(candidate[0])) / weight
            elif tail:
                if len(candidate[0]) < len(tail):
                    penalty += (len(tail) - len(candidate[0])) / weight
        return penalty

    loss_score = loss(lm, context, candidate, lm_type)

    score = loss_score
    if head and len(candidate) >= 1:
        first_token = candidate[0]
        score += calculate_edit_distance(first_token[: len(head)], head)
    if tail and len(candidate) == len_suppl_words and candidate:
        last_token = candidate[-1]
        score += calculate_edit_distance(last_token[-len(tail) :], tail)

    # Applica la penalità allo score
    if len_suppl_words == 1: 
        score += apply_length_penalty(candidate, head, tail, len_suppl_words, weight=1.0)

    return score

def local_beam_search(
    lm: LanguageModel,
    context: list[str],
    lm_type: str = LM_TYPE,
    beam_size: int = K_PRED * 2,
    n: int = N,
    len_suppl_words: int = 0,
    suppl_words: list[str] = None,
    head_suppl: Optional[str] = None,
    tail_suppl: Optional[str] = None,
    mod: str = "acc",
    alpha: float = 0,
    beta: float = 1,
    k_pred: int = K_PRED,
) -> list[list[str]]:
    """
    Esegue una ricerca locale con beam search per generare le migliori K predizioni.

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato.
        context (list[str]): Contesto iniziale.
        beam_size (int, opzionale): Dimensione del beam. Default è K_PRED * 4.
        n (int, opzionale): Numero di parole nel contesto. Default è N.
        suppl_words (list[str], opzionale): Parole supplementari. Default è None.
        mod (str, opzionale): Modalità di calcolo delle predizioni. Default è "acc".
        alpha (float, opzionale): Peso del punteggio della loss. Default è 0.
        beta (float, opzionale): Peso del punteggio della edit distance. Default è 1.
        k_pred (int, opzionale): Numero di predizioni da generare. Default è K_PRED.

    Returns:
        list[list[str]]: Lista delle migliori K sequenze generate.
    """
    beam = []
    # Inizializzazione K stati iniziali
    if len_suppl_words == 1:
        states = get_words_from_context(
            lm, context, beam_size, head_suppl, tail_suppl, n
        )
        # print("Testa: ", head_suppl)
        # print("Coda: ", tail_suppl)
        # print(f"Stati iniziali con sequenza di token da generare pari a 1: {states}")

        for s in states:
            heapq.heappush(
                beam,
                (
                    [s],
                    score_candidate(
                        lm,
                        lm_type,
                        lm.vocab.lookup(context[(1 - n) :]),
                        [s],
                        head_suppl,
                        tail_suppl,
                        len_suppl_words,
                    ),
                ),
            )

        return get_best_candidates_from_beam(
            beam, suppl_words, mod, alpha, beta, k_pred
        )

    else:
        states = get_words_from_context(lm, context, beam_size, head_suppl, None, n)
        # print(f"Stati iniziali con sequenza di token da generare > 1: {states}")
        # print("Testa: ", head_suppl)

        for s in states:
            heapq.heappush(
                beam,
                (
                    [s],
                    score_candidate(
                        lm,
                        lm_type,
                        lm.vocab.lookup(context[(1 - n) :]),
                        [s],
                        head_suppl,
                        tail_suppl,
                        len_suppl_words,
                    ),
                ),
            )
        while True:
            successors = []
            # Generiamo tutti i successori degli stati presenti nel beam
            for candidate in beam:
                if (
                    len(candidate[0]) >= len_suppl_words
                ):  # Termina se la lunghezza della sequenza raggiunge `len_suppl_words` : le altre sequenze del beam hanno la solita lunghezza
                    continue

                if len(candidate[0]) == len_suppl_words - 1:
                    successors.extend(
                        get_successors(
                            lm,
                            lm_type,
                            context,
                            candidate,
                            beam_size,
                            len_suppl_words,
                            head_suppl,
                            tail_suppl,
                            n,
                        )
                    )
            if not successors:
                break  # Nessun successore

            beam = heapq.nsmallest(beam_size, successors, key=lambda x: x[1])

        return get_best_candidates_from_beam(
            beam, suppl_words, mod, alpha, beta, k_pred
        )


def get_best_K_predictions_from_context(
    lm: LanguageModel,
    context: list[str],
    lm_type: str = LM_TYPE,
    len_suppl_words: int = 0,
    suppl_words: list[str] = None,
    head_suppl: Optional[str] = None,
    tail_suppl: Optional[str] = None,
    n: int = N,
    k_pred: int = K_PRED,
    beam_size: int = K_PRED * 2,
    mod: str = "acc",
    alpha: float = 0,
    beta: float = 1,
) -> list[list[str]]:
    """
    Calcola le migliori K predizioni basate sulla probabilità dell'intera sequenza generata (contesto + generazione) utilizzando un algoritmo di ricerca locale (local beam search).

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato.
        context (list[str]): Contesto iniziale.
        suppl_words (list[str], opzionale): Parole supplementari. Default è None.
        n (int, opzionale): Numero di parole nel contesto. Default è N.
        k_pred (int, opzionale): Numero di predizioni da generare. Default è K_PRED.
        beam_size (int, opzionale): Dimensione del beam. Default è K_PRED * 4.
        mod (str, opzionale): Modalità di calcolo delle predizioni. Default è "acc".
        alpha (float, opzionale): Peso del punteggio della loss. Default è 0.
        beta (float, opzionale): Peso del punteggio della edit distance. Default è 1.

    Returns:
        list[list[str]]: Lista delle migliori K sequenze generate.
    """
    return local_beam_search(
        lm,
        context,
        lm_type,
        beam_size,
        n,
        len_suppl_words,
        suppl_words,
        head_suppl,
        tail_suppl,
        mod,
        alpha,
        beta,
        k_pred,
    )


def get_context_from_test_case(
    test_case: str, n=N, case_folding: bool = True
) -> tuple[list[str], str, str]:
    """
    Forma il contesto da cui il modello linguistico fa partire la generazione della predizione.
    Prende la sottostringa del test_case che parte dall'inizio e finisce con '[', la divide in frasi e le tokenizza, pulendone lacune e punteggiatura. (Contesto a sinistra)
    Questo metodo viene usato per generare il contesto dal test_case da cui partire per la generazione della predizioni nel metodo `get_topK_accuracy()`.

    Args:
        test_case (str) : caso di test del blocco anonimo, in cui è presente un supplemento da generare `[...]`
        n (optional, int): ordine degli ngrammi del modello

    Returns:
        tupla : (list[str], str, str)
            - list[str]: lista di token che rappresentano il contesto per l'inferenza del modello linguistico.
            - str: testa del supplemento
            - str: coda del supplemento
    Note:
        - Si assume che il testo passato per parametro denoti un test_case, perciò la stringa deve essere composta da una lacuna, rappresenta da `[...]`.
        - I token risultanti nel contesto vengono imbottiti e troncati alla dimensione specificata degli n-grammi.
    """

    if not test_case_contains_lacuna(test_case):
        raise ValueError(
            "Il test case non contiene una lacuna (Assicurati di mantenere una singola lacuna dentro il testo)"
        )

    context = list(
        flatten(
            list(
                pad_both_ends(
                    get_tokens_from_clean_text(remove_punctuation(sent)),
                    n=n,
                )
            )
            for sent in sentence_tokenizer.tokenize(
                clean_text_from_gaps(
                    re.sub(r"[^\s]+\[", "[", test_case).split("[")[0],
                    case_folding=case_folding,  # Si prende il contesto a sinistra della parentesi `[`
                )
            )
        )
    )[: (1 - n)]

    return (context, get_head_supplement(test_case), get_tail_supplement(test_case))


def check_supplement(supplement: list[str]) -> bool:
    """
    Verifica se la lista di supplementi è valida per la valutazione nella topK accuracy.

    Questa funzione controlla se la lista di supplementi (token) fornita è vuota o
    contiene la stringa "<UNK>". Se una di queste condizioni è vera,
    la funzione restituisce False, altrimenti restituisce True.

    Args:
        supplement (list[str]): Una lista di tokens (stringhe) che rappresentano i supplementi.

    Returns:
        bool: True se la lista è valida, False altrimenti.
    """
    if (
        not supplement or len(supplement) > 1 or "<UNK>" in supplement
    ):  # Supplementi non presenti nel testo
        return False
    return True


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

                    if i >= len(supplements):
                        break

                    if not check_supplement(supplements[i]):
                        continue

                    context, head_suppl, tail_suppl = get_context_from_test_case(
                        obj["test_case"], n
                    )

                    predictions = get_best_K_predictions_from_context(
                        lm=lm,
                        context=context,
                        len_suppl_words=len(supplements[i]),
                        suppl_words=supplements[i],
                        head_suppl=head_suppl,
                        tail_suppl=tail_suppl,
                        n=n,
                        k_pred=k_pred,
                        beam_size=k_pred * 4,
                        mod="acc",
                    )

                    # print("testa del supplemento: ", head_suppl)
                    # print("coda del supplemento: ", tail_suppl)
                    # print("predizioni: ", predictions)

                    if supplements[i] in predictions:
                        correct_predictions += 1

                    total_predictions += 1

    return round((correct_predictions / total_predictions) * 100, 2)
