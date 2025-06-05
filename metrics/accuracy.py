import heapq
import re
from math import inf, log
from typing import Counter, Optional
from tqdm import tqdm
from metrics import FALLBACK_LOSS, MIN_BEAM_SIZE, MAX_BEAM_SIZE
from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import pad_both_ends, flatten
from nltk.metrics.distance import edit_distance
from nltk.lm.util import log_base2
from functools import lru_cache
import gc

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
from config.settings import K_PRED, BATCH_SIZE, N, LM_TYPE, LAMBDA, ALPHA, BETA, DELTA


@lru_cache(maxsize=None)
def cached_edit_distance(a, b):
    """
    Versione cachabile di `edit_distance`
    """
    return edit_distance(a, b)


def filter_words(df):
    """Filtra parole non valide nella distribuzione di frequenza, come i token speciali."""
    return [w for w, _ in df if w not in ["</s>", "<s>", "<UNK>"]]


def sort_and_filter_array(arr, condition, key_func):
    """Ordina e filtra gli elementi di un array in base a una condizione e una funzione di ordinamento."""
    annotated = [(item, key_func(item)) for item in arr]
    target = [x for x in annotated if condition(x[0])]
    remaining = [x for x in annotated if not condition(x[0])]

    target_sorted = sorted(target, key=lambda x: x[1])
    remaining_sorted = sorted(remaining, key=lambda x: x[1])
    return [x[0] for x in target_sorted + remaining_sorted]


def get_sorted_filtered_array(arr, head, tail, len_lacuna):
    """Ordina e filtra gli elementi di un array in base a testa e coda."""
    head_len = len(head) if head else 0
    tail_len = len(tail) if tail else 0

    if head and tail:
        return sort_and_filter_array(
            arr,
            lambda item: (len(item) == (head_len + tail_len + len_lacuna)) and ((item.startswith(head) and item.endswith(tail)) or (item.startswith(head)) or item.endswith(tail)),
            lambda x: cached_edit_distance(x[:head_len], head)
            + cached_edit_distance(x[-tail_len:], tail),
        )
    elif head:
        return sort_and_filter_array(
            arr,
            lambda item: (len(item) >= head_len + len_lacuna) and item.startswith(head),
            lambda x: cached_edit_distance(x[:head_len], head),
        )
    elif tail:
        return sort_and_filter_array(
            arr,
            lambda item: (len(item) >= tail_len + len_lacuna) and item.endswith(tail),
            lambda x: cached_edit_distance(x[-tail_len:], tail),
        )
    else:
        # Ordina solo in base alla lunghezza se non ci sono informazioni su testa o coda
        return sort_and_filter_array(
            arr,
            lambda item: len(item) == len_lacuna,
            lambda x: abs(len(x) - len_lacuna),
        )


def sort_and_filter(df, condition, key_func):
    """Ordina e filtra le parole in base a una condizione e una funzione di ordinamento."""
    target = []
    remaining = []

    for w, f in df:
        if condition(w):
            target.append((w, f))
        else:
            remaining.append((w, f))

    target_sorted = sorted(target, key=key_func)
    remaining_sorted = sorted(remaining, key=key_func, reverse=True)

    return filter_words(target_sorted + remaining_sorted)


def get_sorted_filtered_words(df, head, tail, len_lacuna):
    """Ordina e filtra le parole in base a testa e coda."""
    head_len = len(head) if head else 0
    tail_len = len(tail) if tail else 0
    
    if head and tail:
        return sort_and_filter(
            df,
            lambda w: (len(w) == (head_len + tail_len + len_lacuna)) and ((w.startswith(head) and w.endswith(tail)) or (w.startswith(head)) or w.endswith(tail)),
            lambda x: cached_edit_distance(x[0][: head_len], head)
            + cached_edit_distance(x[0][-tail_len :], tail),
        )
    elif head:
        return sort_and_filter(
            df,
            lambda w: (len(w) >= head_len + len_lacuna) and w.startswith(head),
            lambda x: cached_edit_distance(x[0][: head_len], head),
        )
    elif tail:
        return sort_and_filter(
            df,
            lambda w: (len(w) >= tail_len + len_lacuna) and w.endswith(tail),
            lambda x: cached_edit_distance(x[0][-tail_len :], tail),
        )
    else:
        # Ordina solo in base alla frequenza se non ci sono informazioni su testa o coda
        return sort_and_filter(
            df,
            lambda w: len(w) == len_lacuna,
            lambda x: x[1],
        )


def get_dist_freq_words_from_context(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    context: list[str],
    len_lacuna: int,
    n=N,
    head: Optional[str] = None,
    tail: Optional[str] = None,
) -> list[str]:
    """
    Genera la distribuzione di frequenze delle parole per un dato contesto.

    Args:
        g_lm (LanguageModel): Modello di linguaggio globale utilizzato.
        d_lm (LanguageModel): Modello di linguaggio di dominio utilizzato.
        context (list[str]): Lista di token che rappresenta il contesto.
        n (int): Dimensione del modello (numero di parole nel contesto).
        head (Optional[str]): Se specificato, filtra solo le parole che iniziano con questa sottostringa.
        tail (Optional[str]): Se specificato, filtra solo le parole che finiscono con questa sottostringa.

    Returns:
        list[str]: Lista di parole ordinate in base alla testa, coda e frequenza.
    """

    if n <= 1:
        return []

    g_df = Counter(g_lm.context_counts(g_lm.vocab.lookup(context[-(n - 1) :])))
    d_df = Counter(d_lm.context_counts(d_lm.vocab.lookup(context[-(n - 1) :])))
    merged = g_df + d_df
    return get_sorted_filtered_words(merged.items(), head, tail, len_lacuna)


def get_words_from_context(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    context: list[str],
    boundary: int,
    len_lacuna: int,
    head: Optional[str] = None,
    tail: Optional[str] = None,
    n: int = N,
) -> list[str]:
    """
    Funzione con cui espandiamo le parole possibili successive da uno stato (il numero massimo consentito è rappresentato da `boundary`).
    Se la distribuzione è vuota, riduce progressivamente il contesto fino a trovare una distribuzione non vuota. (Backoff)

    Args:
        g_lm (LanguageModel): Modello di linguaggio globale utilizzato.
        d_lm (LanguageModel): Modello di linguaggio di dominio utilizzato.
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

    while len(tokens) < boundary:
        # Se non ci sono più parole disponibili, esco
        if n <= 1:
            break

        for next_w in get_dist_freq_words_from_context(
            g_lm, d_lm, context, len_lacuna, n, head, tail
        ):

            if next_w in dist_words:
                continue

            dist_words.add(next_w)  # La aggiungo alla lista delle parole già viste
            tokens.append(next_w)
        n -= 1

    return get_sorted_filtered_array(tokens, head, tail, len_lacuna)


def interpolated_log_score(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    word: str,
    context: list[str],
    lambda_weight: float,
) -> float:
    """
    Calcola la log-probabilità di una parola dato il contesto, utilizzando
    l'interpolazione lineare delle probabilità di due modelli di linguaggio.
    Restituisce -inf se la probabilità è zero.

    Args:
        g_lm (LanguageModel): Modello di linguaggio globale.
        d_lm (LanguageModel): Modello di linguaggio specifico di dominio.
        word (str): Parola di cui calcolare la probabilità.
        context (tuple): Contesto di parole precedenti.
        lambda_weight (float): Peso per l'interpolazione lineare.

    Returns:
        float: Log-probabilità della parola.
    """

    return log_base2(
        lambda_weight * d_lm.score(word, d_lm.vocab.lookup(context))
        + (1 - lambda_weight) * g_lm.score(word,  g_lm.vocab.lookup(context))
    )


def nll_score(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    lambda_weight: float,
    context: list[str],
    generated_seq: list[str],
    lm_type=LM_TYPE,
) -> float:
    """
    Funzione di scoring usata nella local beam search.
    Tale implementazione fa riferimento alla metrica NLL (Negative Log Likelihood) normalizzata per la lunghezza della frase complessiva (contesto + generazione).

    Args:
        g_lm (LanguageModel): Modello di linguaggio globale utilizzato.
        d_lm (LanguageModel): Modello di linguaggio di dominio utilizzato.
        lambda_weight (float): Iperparametro da determinare per l'interpolazione
        context (tuple): Contesto di parole.
        generated_seq (list[str]): Sequenza di parole generata.
        lm_type (str, opzionale): Tipo di modello di linguaggio. Default è LM_TYPE.

    Returns:
        float: Valore della funzione di perdita.
    """

    def update_context(context: list[str], word: str) -> list[str]:
        """
        Aggiorna il contesto includendo la parola generata.

        Args:
            context (list): Contesto di parole.
            word (str): Parola da aggiungere al contesto.

        Returns:
            list: Nuovo contesto aggiornato.
        """
        return [*context[1:], word]

    total_log_prob = 0
    for word in generated_seq:
        total_log_prob += interpolated_log_score(
            g_lm, d_lm, word, context, lambda_weight
        )
        context = update_context(context, word)

    if lm_type == "MLE":
        return (
            -(total_log_prob / len(generated_seq))
            if total_log_prob != -inf
            else FALLBACK_LOSS
        )  # Si aggiunge un valore di fallback nel caso in cui le parole della sequenza non siano presenti nel modello

    return -total_log_prob


def get_best_candidates_from_beam(
    g_lm: LanguageModel,
    d_lm: LanguageModel, 
    context: tuple[str],
    beam: list[tuple[list[str], float]],
    len_suppl_words: int,
    suppl_words: list[str] = None,
    lambda_weight: float = LAMBDA,
    k_pred: int = K_PRED,
    mod="acc",	
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
    
    if len_suppl_words > 1: 
        raise ValueError("Ranking for `len_suppl_words > 1` not implemented yet")

    if mod=="acc": 
        gold_label = " ".join(suppl_words)
        sorted_candidates = sorted([candidate[0] for candidate in beam], key=lambda x: cached_edit_distance(" ".join(x), gold_label))
        return [tkn for tkn in sorted_candidates][:k_pred]
    else: 
        sorted_candidates = sorted(beam, key=lambda x: x[1])
        string_candidates = [candidate[0] for candidate in sorted_candidates]
        return [tkn for tkn in string_candidates][:k_pred]


def get_successors(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    lambda_weight: float,
    len_lacuna: int,
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
        g_lm (LanguageModel): Modello di linguaggio globale utilizzato.
        g_lm (LanguageModel): Modello di linguaggio di dominio utilizzato.
        context (list[str]): Contesto di parole.
        candidate (tuple[list[str], float]): Candidato corrente.
        beam_size (int, opzionale): Dimensione del beam. Default è K_PRED * 4.
        n (int, opzionale): Numero di parole nel contesto. Default è N.

    Returns:
        list[tuple[list[str], float]]: Lista di successori generati.
    """
    successors = []
    nexts = get_words_from_context(
        g_lm, d_lm, context + candidate[0], beam_size, None, tail, n
    )
    for w in nexts:
        successors.append(
            (
                candidate[0] + [w],
                score_candidate(
                    g_lm,
                    d_lm,
                    lambda_weight,
                    len_lacuna,
                    lm_type,
                    context[(1 - n):],
                    candidate[0] + [w],
                    head,
                    tail,
                    len_suppl_words,
                ),
            )
        )
    return successors

def apply_length_penalty(
        candidate: list[str],
        head: Optional[str],
        tail: Optional[str],
        len_lacuna: int,
        len_suppl_words: int,
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
        if len_suppl_words != 1:
            return 0

        target_length = len_lacuna
        if head:
            target_length += len(head)
        if tail:
            target_length += len(tail)

        candidate_length = len(candidate[0])
        length_difference = abs(candidate_length - target_length)

        return length_difference
    
def score_candidate(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    lambda_weight: float,
    len_lacuna: int,
    lm_type: str,
    context: list[str],
    candidate: list[str],
    head: Optional[str] = None,
    tail: Optional[str] = None,
    len_suppl_words: int = 0,
    alpha: float = ALPHA,
    beta: float = BETA,
    delta: float = DELTA, 
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

    nll = nll_score(g_lm, d_lm, lambda_weight, context, candidate, lm_type)
    edit_dist_head = 0
    edit_dist_tail = 0

    if head and len(candidate) >= 1:
        first_token = candidate[0]
        edit_dist_head += cached_edit_distance(first_token[: len(head)], head)
    if tail and len(candidate) == len_suppl_words and candidate:
        last_token = candidate[-1]
        edit_dist_tail += cached_edit_distance(last_token[-len(tail) :], tail)

    
    length_pen = 0
    if len_suppl_words == 1:
        length_pen = apply_length_penalty(candidate, head, tail, len_lacuna, len_suppl_words)

    return alpha * nll + beta * (edit_dist_head + edit_dist_tail) + delta * length_pen


def local_beam_search(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    context: list[str],
    len_lacuna: int = 0,
    lambda_weight: float = LAMBDA,
    lm_type: str = LM_TYPE,
    beam_size: int = K_PRED * 4,
    n: int = N,
    len_suppl_words: int = 0,
    suppl_words: list[str] = None,
    head_suppl: Optional[str] = None,
    tail_suppl: Optional[str] = None,
    mod: str = "acc",
    alpha: float = ALPHA,
    beta: float = BETA,
    delta: float = DELTA,
    k_pred: int = K_PRED,
) -> list[list[str]]:
    """
    Esegue una ricerca locale utilizzando l'algoritmo di beam search per generare
    le migliori K predizioni basate su un contesto iniziale e su modelli di linguaggio.

        g_lm (LanguageModel): Modello di linguaggio globale utilizzato per il calcolo delle probabilità.
        d_lm (LanguageModel): Modello di linguaggio di dominio utilizzato per il calcolo delle probabilità.
        context (list[str]): Lista di token che rappresenta il contesto iniziale.
        len_lacuna (int): Lunghezza della lacuna da riempire.
        lambda_weight (float, opzionale): Peso del modello di dominio rispetto al modello globale. Default è LAMBDA.
        lm_type (str, opzionale): Tipo di modello di linguaggio utilizzato. Default è LM_TYPE.
        beam_size (int, opzionale): Dimensione del beam, ovvero il numero massimo di stati mantenuti ad ogni iterazione. Default è K_PRED * 2.
        n (int, opzionale): Numero di token considerati nel contesto. Default è N.
        len_suppl_words (int, opzionale): Numero di parole supplementari da generare. Default è 0.
        suppl_words (list[str], opzionale): Lista di parole supplementari da considerare. Default è None.
        head_suppl (Optional[str], opzionale): Token opzionale da utilizzare come prefisso per le parole supplementari. Default è None.
        tail_suppl (Optional[str], opzionale): Token opzionale da utilizzare come suffisso per le parole supplementari. Default è None.
        mod (str, opzionale): Modalità di calcolo delle predizioni (ad esempio, "acc" per accuratezza). Default è "acc".
        alpha (float, opzionale): Peso del punteggio dello score assegnato dall'algoritmo di ricerca nel calcolo del ranking. Default è 1.
        beta (float, opzionale): Peso del punteggio della edit distance nel calcolo del ranking. Default è 1.
        delta: float, opzionale): Peso del punteggio della `length penalty` nel calcolo del ranking. Default è 1.
        k_pred (int, opzionale): Numero di predizioni finali da generare. Default è K_PRED.

        list[list[str]]: Una lista contenente le migliori K sequenze generate, ordinate in base al punteggio.
    """
    beam = []
    # Inizializzazione K stati iniziali
    if len_suppl_words == 1:
        states = get_words_from_context(
            g_lm, d_lm, context, beam_size, len_lacuna, head_suppl, tail_suppl, n
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
                        g_lm,
                        d_lm,
                        lambda_weight,
                        len_lacuna,
                        lm_type,
                        g_lm.vocab.lookup(context[(1 - n) :]),
                        [s],
                        head_suppl,
                        tail_suppl,
                        len_suppl_words,
                        alpha,
                        beta,
                        delta
                    ),
                ),
            )

        return get_best_candidates_from_beam(
            g_lm, d_lm, context, beam, len_suppl_words, suppl_words, lambda_weight, k_pred, mod
        )

    else:
        states = get_words_from_context(
            g_lm, d_lm, context, beam_size, 0, head_suppl, None, n
        )

        for s in states:
            heapq.heappush(
                beam,
                (
                    [s],
                    score_candidate(
                        g_lm,
                        d_lm,
                        lambda_weight,
                        0,  # si inibisce l'informazione sulla lunghezza della lacuna
                        lm_type,
                        g_lm.vocab.lookup(context[(1 - n) :]),
                        [s],
                        head_suppl,
                        tail_suppl,
                        len_suppl_words,
                        alpha,
                        beta,
                        delta
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
                            g_lm,
                            d_lm,
                            lambda_weight,
                            0,
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
            g_lm, d_lm, context, beam, len_suppl_words, lambda_weight, k_pred
        )


def get_best_K_predictions_from_context(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    context: list[str],
    len_lacuna: int,
    lambda_weight: float = LAMBDA,
    lm_type: str = LM_TYPE,
    len_suppl_words: int = 0,
    suppl_words: list[str] = None,
    head_suppl: Optional[str] = None,
    tail_suppl: Optional[str] = None,
    n: int = N,
    k_pred: int = K_PRED,
    beam_size: int = K_PRED * 4,
    mod: str = "acc",
    alpha: float = ALPHA,
    beta: float = BETA,
    delta: float = DELTA,
) -> list[list[str]]:
    """
    Calcola le migliori K predizioni basate sulla probabilità dell'intera sequenza generata (contesto + generazione) utilizzando un algoritmo di ricerca locale (local beam search).

    Args:
        g_lm (LanguageModel): Modello di linguaggio globale utilizzato.
        d_lm (LanguageModel): Modello di linguaggio di dominio utilizzato.
        context (list[str]): Contesto iniziale.
        len_lacuna (int): Lunghezza della lacuna (numero di frasi).
        lambda_weight (float, opzionale): Iperparametro per la definizione dei punteggi tramite interpolazione lineare. Default è `LAMBDA`.
        lm_type (str, opzionale): Tipo di modello di linguaggio. Default è `LM_TYPE`.
        len_suppl_words (int, opzionale): Lunghezza del supplemento. Default è 0.
        suppl_words (list[str], opzionale): Parole supplementari. Default è None.
        head_suppl (Optional[str], opzionale): Testa del supplemento. Default è None.
        tail_suppl (Optional[str], opzionale): Coda del supplemento. Default è None.
        n (int, opzionale): Numero di parole nel contesto. Default è `N`.
        k_pred (int, opzionale): Numero di predizioni da generare. Default è `K_PRED`.
        beam_size (int, opzionale): Dimensione del beam. Default è `K_PRED * 2`.
        mod (str, opzionale): Modalità di calcolo delle predizioni. Default è "acc".
        alpha (float, opzionale): Peso del punteggio della loss. Default è 0.
        beta (float, opzionale): Peso del punteggio della distanza di edit. Default è 1.
        delta (float, opzionale): Peso del punteggio della `length penalty`. Default è 1.

    Returns:
        list[list[str]]: Lista delle migliori K sequenze generate.
    """
    return local_beam_search(
        g_lm=g_lm,
        d_lm=d_lm,
        context=context,
        len_lacuna=len_lacuna,
        lambda_weight=lambda_weight,
        lm_type=lm_type,
        beam_size=beam_size,
        n=n,
        len_suppl_words=len_suppl_words,
        suppl_words=suppl_words,
        head_suppl=head_suppl,
        tail_suppl=tail_suppl,
        mod=mod,
        alpha=alpha,
        beta=beta,
        delta=delta, 
        k_pred=k_pred,
    )


def get_context_from_test_case(
    test_case: str, n=N, case_folding: bool = True
) -> tuple[list[str], str, str, int]:
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

    match = test_case_contains_lacuna(test_case)
    if not match:
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

    return (
        context,
        get_head_supplement(test_case),
        get_tail_supplement(test_case),
        match.count("."),
    )


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


def get_beam_size(
    k_predictions: int,
    beam_multiplier: int,
) -> int:
    """
    Calcola la dimensione del beam da utilizzare in base al numero di predizioni desiderate e a un moltiplicatore.

    La funzione restituisce il valore minimo tra il massimo tra (k_predictions * beam_multiplier) e MIN_BEAM_SIZE, e MAX_BEAM_SIZE.
    Questo garantisce che la dimensione del beam sia sempre compresa tra MIN_BEAM_SIZE e MAX_BEAM_SIZE.

    Args:
        k_predictions (int): Numero di predizioni desiderate.
        beam_multiplier (int): Moltiplicatore per calcolare la dimensione del beam.

    Returns:
        int: Dimensione del beam calcolata e limitata tra MIN_BEAM_SIZE e MAX_BEAM_SIZE.
    """
    return min(max(k_predictions * beam_multiplier, MIN_BEAM_SIZE), MAX_BEAM_SIZE)


def get_topK_accuracy(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    test_abs: list,
    lambda_weight: float = LAMBDA,
    batch_size=BATCH_SIZE,
    n=N,
    k_pred=K_PRED,
    alpha: float = ALPHA,
    beta: float = BETA,
    delta: float = DELTA,
) -> float:
    """
    Calcola la topK accuracy del modello a n-grammi su un insieme di blocchi anonimi di test.

    Args:
        g_lm (LanguageModel): Il modello di linguaggio generico, addestrato su tutto il dataset.
        d_lm (LanguageModel): Il modello di linguaggio di dominio, addestrato su un dominio specifico.
        test_abs (list): Lista di blocchi anonimi di testo per il test.
        lambda_weight (float): Peso dell'interpolazione lineare, da regolare tramite EM.
        batch_size (int, opzionale): La dimensione del batch per l'elaborazione. Default è `BATCH_SIZE`.
        n (int, opzionale): Dimensioni degli ngrammi del modello linguistico. Default è `N`.
        k_pred (int, opzionale): Il numero di predizioni da generare per ogni contesto. Default è `K_PRED`.
        alpha (float, opzionale): Peso del punteggio della loss. Default è 0.
        beta (float, opzionale): Peso del punteggio della edit distance. Default è 1.   
        delta (float, opzionale): Peso del punteggio della `length penalty`. Default è 1.   

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

                    suppl_seq, suppl_len_lacuna = supplements[i]

                    if not check_supplement(suppl_seq):
                        continue

                    context, head_suppl, tail_suppl, _ = get_context_from_test_case(
                        obj["test_case"], n
                    )

                    predictions = get_best_K_predictions_from_context(
                        g_lm=g_lm,
                        d_lm=d_lm,
                        context=context,
                        len_lacuna=suppl_len_lacuna,
                        lambda_weight=lambda_weight,
                        len_suppl_words=len(suppl_seq),
                        suppl_words=suppl_seq,
                        head_suppl=head_suppl,
                        tail_suppl=tail_suppl,
                        n=n,
                        k_pred=k_pred,
                        beam_size=get_beam_size(k_pred, 4),
                        mod="acc",
                        alpha=alpha,
                        beta=beta,
                        delta=delta,
                    )
                    
                    if suppl_seq in predictions:
                        correct_predictions += 1

                    total_predictions += 1
        
    return round((correct_predictions / total_predictions), 5)
