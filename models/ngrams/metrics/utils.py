"""
Utility per l'inferenza e la valutazione dei modelli a n-grammi:
- distribuzione di frequenza delle parole candidate dato un contesto
- scoring tramite NLL interpolata e penalità di lunghezza/edit distance
- local beam search per la generazione delle top-K predizioni
- estrazione del contesto da un test case MAAT
"""

import heapq
import re
from functools import lru_cache
from math import inf
from typing import Optional

from nltk.lm.models import LanguageModel
from nltk.lm.preprocessing import flatten, pad_both_ends
from nltk.lm.util import log_base2
from nltk.metrics.distance import edit_distance
from typing import Counter

from backend.config.settings import ALPHA, BETA, DELTA, K_PRED, LAMBDA, LM_TYPE, N
from backend.core.preprocess import (
    clean_text_from_gaps,
    get_head_supplement,
    get_tail_supplement,
    get_tokens_from_clean_text,
    remove_punctuation,
    test_case_contains_lacuna,
)
from models.ngrams.metrics import (
    FALLBACK_LOSS,
    MAX_BEAM_SIZE,
    MIN_BEAM_SIZE,
    _SPECIAL_TOKENS,
    sentence_tokenizer,
)


# ---------------------------------------------------------------------------
# Edit distance con cache
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def cached_edit_distance(a: str, b: str) -> int:
    """Versione con cache di `nltk.metrics.distance.edit_distance`."""
    return edit_distance(a, b)


# ---------------------------------------------------------------------------
# Filtering e ordinamento della distribuzione di frequenza
# ---------------------------------------------------------------------------


def filter_words(df: list[tuple[str, int]]) -> list[str]:
    """Rimuove i token speciali dalla distribuzione di frequenza."""
    return [w for w, _ in df if w not in _SPECIAL_TOKENS]


def sort_and_filter(
    df: list[tuple[str, int]],
    condition: callable,
    key_func: callable,
) -> list[str]:
    """
    Separa le parole di ``df`` in due gruppi (quelle che soddisfano ``condition``
    e le rimanenti), ordina ciascun gruppo con ``key_func`` e restituisce la
    lista concatenata, filtrata dai token speciali.

    Args:
        df: Distribuzione di frequenza — lista di tuple (parola, frequenza).
        condition: Predicato su una stringa; ``True`` → gruppo *target*.
        key_func: Chiave di ordinamento applicata a ogni tupla (parola, frequenza).

    Returns:
        Lista di parole filtrate e ordinate (target prima, rimanenti dopo).
    """
    target, remaining = [], []
    for item in df:
        (target if condition(item[0]) else remaining).append(item)

    return filter_words(sorted(target, key=key_func) + sorted(remaining, key=key_func))


def get_sorted_filtered_words(
    df: list[tuple[str, int]],
    head: Optional[str],
    tail: Optional[str],
    len_lacuna: int,
) -> list[str]:
    """
    Ordina e filtra le parole candidate in base a testa, coda e lunghezza della lacuna.

    Args:
        df: Distribuzione di frequenza — lista di tuple (parola, frequenza).
        head: Prefisso atteso del supplemento (può essere ``None``).
        tail: Suffisso atteso del supplemento (può essere ``None``).
        len_lacuna: Numero di caratteri mancanti nella lacuna.

    Returns:
        Lista di parole ordinate per distanza di edit dalla testa/coda.
    """
    head_len = len(head) if head else 0
    tail_len = len(tail) if tail else 0

    if head and tail:
        expected_len = head_len + tail_len + len_lacuna
        return sort_and_filter(
            df,
            lambda w: (
                len(w) == expected_len and (w.startswith(head) or w.endswith(tail))
            ),
            lambda x: (
                cached_edit_distance(x[0][:head_len], head)
                + cached_edit_distance(x[0][-tail_len:], tail)
            ),
        )

    if head:
        return sort_and_filter(
            df,
            lambda w: len(w) == head_len + len_lacuna and w.startswith(head),
            lambda x: cached_edit_distance(x[0][:head_len], head),
        )

    if tail:
        return sort_and_filter(
            df,
            lambda w: len(w) == tail_len + len_lacuna and w.endswith(tail),
            lambda x: cached_edit_distance(x[0][-tail_len:], tail),
        )

    # Nessuna informazione su testa/coda: ordina per frequenza
    return sort_and_filter(
        df,
        lambda w: len(w) == len_lacuna,
        lambda x: x[1],
    )


# ---------------------------------------------------------------------------
# Distribuzione di frequenza dal contesto
# ---------------------------------------------------------------------------


def get_dist_freq_words_from_context(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    context: list[str],
    len_lacuna: int,
    n: int = N,
    head: Optional[str] = None,
    tail: Optional[str] = None,
) -> list[str]:
    """
    Genera la distribuzione di frequenza delle parole candidate per il contesto dato,
    combinando il modello globale e quello di dominio.

    Args:
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        context: Lista di token che rappresenta il contesto.
        len_lacuna: Lunghezza della lacuna in caratteri.
        n: Dimensione degli n-grammi. Default è N.
        head: Prefisso atteso del supplemento. Default è ``None``.
        tail: Suffisso atteso del supplemento. Default è ``None``.

    Returns:
        Lista di parole ordinate per testa, coda e frequenza.
    """
    if n <= 1:
        return []

    ctx_slice = context[-(n - 1) :]
    g_df = Counter(g_lm.context_counts(g_lm.vocab.lookup(ctx_slice)))
    d_df = Counter(d_lm.context_counts(d_lm.vocab.lookup(ctx_slice)))
    merged = g_df + d_df
    return get_sorted_filtered_words(list(merged.items()), head, tail, len_lacuna)


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
    Espande le parole candidate successive a partire dal contesto, con backoff
    progressivo sul contesto se la distribuzione è vuota.

    Args:
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        context: Lista di token che rappresenta il contesto corrente.
        boundary: Numero massimo di parole da raccogliere.
        len_lacuna: Lunghezza della lacuna in caratteri.
        head: Prefisso atteso del supplemento. Default è ``None``.
        tail: Suffisso atteso del supplemento. Default è ``None``.
        n: Ordine degli n-grammi. Default è N.

    Returns:
        Lista di (al più ``boundary``) parole candidate distinte.
    """
    tokens: list[str] = []
    seen: set[str] = set()

    while len(tokens) < boundary:
        if n <= 1:
            break

        for word in get_dist_freq_words_from_context(
            g_lm, d_lm, context, len_lacuna, n, head, tail
        ):
            if word not in seen:
                seen.add(word)
                tokens.append(word)

        n -= 1

    return tokens


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def interpolated_log_score(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    word: str,
    context: list[str],
    lambda_weight: float,
) -> float:
    """
    Log-probabilità (base 2) di ``word`` dato ``context``, calcolata tramite
    interpolazione lineare dei due modelli.

    Args:
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        word: Parola di cui calcolare la probabilità.
        context: Contesto di parole precedenti.
        lambda_weight: Peso del modello di dominio nell'interpolazione.

    Returns:
        Log-probabilità della parola; ``-inf`` se la probabilità è zero.
    """
    p = lambda_weight * d_lm.score(word, d_lm.vocab.lookup(context)) + (
        1 - lambda_weight
    ) * g_lm.score(word, g_lm.vocab.lookup(context))
    return log_base2(p)


def nll_score(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    lambda_weight: float,
    context: list[str],
    generated_seq: list[str],
    lm_type: str = LM_TYPE,
) -> float:
    """
    NLL (Negative Log Likelihood) normalizzata per la lunghezza della sequenza
    generata, usata come funzione di scoring nella beam search.

    Per i modelli MLE la NLL è normalizzata per la lunghezza; per gli altri
    modelli viene restituita la NLL grezza. In caso di probabilità zero viene
    restituito ``FALLBACK_LOSS``.

    Args:
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        lambda_weight: Peso del modello di dominio nell'interpolazione.
        context: Contesto di parole precedenti.
        generated_seq: Sequenza di parole generate.
        lm_type: Tipo di modello (``"MLE"`` o altro). Default è LM_TYPE.

    Returns:
        Valore della funzione di perdita (più basso = migliore).
    """
    total_log_prob = 0.0
    ctx = list(context)

    for word in generated_seq:
        total_log_prob += interpolated_log_score(g_lm, d_lm, word, ctx, lambda_weight)
        ctx = [*ctx[1:], word]  # aggiorna il contesto sliding-window

    if lm_type == "MLE":
        return (
            -(total_log_prob / len(generated_seq))
            if total_log_prob != -inf
            else FALLBACK_LOSS
        )

    return -total_log_prob


def apply_length_penalty(
    candidate: list[str],
    head: Optional[str],
    tail: Optional[str],
    len_lacuna: int,
    len_suppl_words: int,
) -> float:
    """
    Penalità basata sulla differenza di lunghezza tra il candidato e il target
    (testa + lacuna + coda). Applicata solo quando ``len_suppl_words == 1``.

    Args:
        candidate: Lista di token del candidato.
        head: Prefisso atteso del supplemento.
        tail: Suffisso atteso del supplemento.
        len_lacuna: Numero di caratteri mancanti nella lacuna.
        len_suppl_words: Numero di parole del supplemento.

    Returns:
        Differenza assoluta di lunghezza (0 se ``len_suppl_words != 1``).
    """
    if len_suppl_words != 1:
        return 0.0

    target_length = len_lacuna + (len(head) if head else 0) + (len(tail) if tail else 0)
    return float(abs(len(candidate[0]) - target_length))


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
    Punteggio composito di un candidato: combinazione pesata di NLL, edit
    distance dalla testa/coda e penalità di lunghezza.

    Args:
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        lambda_weight: Peso del modello di dominio nell'interpolazione.
        len_lacuna: Lunghezza della lacuna in caratteri.
        lm_type: Tipo di modello (``"MLE"`` o altro).
        context: Contesto di parole precedenti.
        candidate: Sequenza di token candidata.
        head: Prefisso atteso del supplemento. Default è ``None``.
        tail: Suffisso atteso del supplemento. Default è ``None``.
        len_suppl_words: Numero di parole del supplemento. Default è 0.
        alpha: Peso della NLL. Default è ALPHA.
        beta: Peso dell'edit distance. Default è BETA.
        delta: Peso della length penalty. Default è DELTA.

    Returns:
        Punteggio scalare del candidato (più basso = migliore).
    """
    nll = (
        nll_score(g_lm, d_lm, lambda_weight, context, candidate, lm_type)
        if alpha > 0
        else 0.0
    )

    edit_dist_head = (
        cached_edit_distance(candidate[0][: len(head)], head)
        if head and candidate
        else 0
    )
    edit_dist_tail = (
        cached_edit_distance(candidate[-1][-len(tail) :], tail)
        if tail and len(candidate) == len_suppl_words and candidate
        else 0
    )

    length_pen = apply_length_penalty(
        candidate, head, tail, len_lacuna, len_suppl_words
    )

    return (
        (alpha * nll)
        + (beta * (edit_dist_head + edit_dist_tail))
        + (delta * length_pen)
    )


# ---------------------------------------------------------------------------
# Beam search
# ---------------------------------------------------------------------------


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
    Genera i successori di un candidato nel beam estendendo la sequenza di
    un token alla volta.

    Args:
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        lambda_weight: Peso del modello di dominio nell'interpolazione.
        len_lacuna: Lunghezza della lacuna in caratteri.
        lm_type: Tipo di modello.
        context: Contesto iniziale.
        candidate: Tupla (sequenza corrente, score).
        beam_size: Numero massimo di successori da esplorare. Default è K_PRED * 2.
        len_suppl_words: Numero di parole del supplemento. Default è 0.
        head: Prefisso atteso. Default è ``None``.
        tail: Suffisso atteso. Default è ``None``.
        n: Ordine degli n-grammi. Default è N.

    Returns:
        Lista di tuple (sequenza estesa, score) per ogni successore.
    """
    extended_context = context + candidate[0]
    nexts = get_words_from_context(
        g_lm, d_lm, extended_context, beam_size, None, tail, n
    )

    return [
        (
            candidate[0] + [w],
            score_candidate(
                g_lm,
                d_lm,
                lambda_weight,
                len_lacuna,
                lm_type,
                context[(1 - n) :],
                candidate[0] + [w],
                head,
                tail,
                len_suppl_words,
            ),
        )
        for w in nexts
    ]


def get_best_candidates_from_beam(
    beam: list[tuple[list[str], float]],
    len_suppl_words: int,
    suppl_words: Optional[list[str]] = None,
    k_pred: int = K_PRED,
    mod: str = "acc",
) -> list[list[str]]:
    """
    Estrae le migliori ``k_pred`` sequenze dal beam.

    In modalità ``"acc"`` i candidati vengono riordinati per edit distance
    dalla gold label (usata solo in fase di valutazione). Altrimenti vengono
    restituiti nell'ordine in cui compaiono nel beam.

    Args:
        beam: Lista di tuple (sequenza, score).
        len_suppl_words: Numero di parole del supplemento.
        suppl_words: Gold label (lista di token). Default è ``None``.
        k_pred: Numero di predizioni da restituire. Default è K_PRED.
        mod: ``"acc"`` per ordinamento per edit distance, altro per ordine beam.

    Returns:
        Lista delle migliori ``k_pred`` sequenze.

    Raises:
        ValueError: Se ``len_suppl_words > 1`` (ranking multi-parola non implementato).
    """
    if not beam:
        return []
    if len_suppl_words > 1:
        raise ValueError("Ranking for `len_suppl_words > 1` not implemented yet")

    if mod == "acc":
        gold = " ".join(suppl_words)
        candidates = sorted(
            [c[0] for c in beam],
            key=lambda x: cached_edit_distance(" ".join(x), gold),
        )
    else:
        candidates = [c[0] for c in beam]

    return candidates[:k_pred]


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
    suppl_words: Optional[list[str]] = None,
    head_suppl: Optional[str] = None,
    tail_suppl: Optional[str] = None,
    mod: str = "acc",
    alpha: float = ALPHA,
    beta: float = BETA,
    delta: float = DELTA,
    k_pred: int = K_PRED,
) -> list[list[str]]:
    """
    Local beam search per generare le migliori K predizioni dato un contesto.

    Per supplementi di una sola parola (``len_suppl_words == 1``) esegue una
    ricerca shallow sui token immediatamente successivi al contesto.
    Per supplementi multi-parola espande iterativamente il beam fino a
    raggiungere la lunghezza target.

    Args:
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        context: Contesto iniziale (lista di token).
        len_lacuna: Lunghezza della lacuna in caratteri. Default è 0.
        lambda_weight: Peso del modello di dominio. Default è LAMBDA.
        lm_type: Tipo di modello. Default è LM_TYPE.
        beam_size: Dimensione massima del beam. Default è K_PRED * 4.
        n: Ordine degli n-grammi. Default è N.
        len_suppl_words: Numero di parole del supplemento. Default è 0.
        suppl_words: Gold label per il re-ranking in modalità ``"acc"``. Default è ``None``.
        head_suppl: Prefisso atteso del supplemento. Default è ``None``.
        tail_suppl: Suffisso atteso del supplemento. Default è ``None``.
        mod: Modalità di estrazione (``"acc"`` o altro). Default è ``"acc"``.
        alpha: Peso NLL. Default è ALPHA.
        beta: Peso edit distance. Default è BETA.
        delta: Peso length penalty. Default è DELTA.
        k_pred: Numero di predizioni finali. Default è K_PRED.

    Returns:
        Lista delle migliori K sequenze generate, ordinate per punteggio.
    """
    beam: list[tuple[list[str], float]] = []

    _score = lambda cand: score_candidate(
        g_lm=g_lm,
        d_lm=d_lm,
        lambda_weight=lambda_weight,
        len_lacuna=len_lacuna,
        lm_type=lm_type,
        context=context,
        candidate=cand,
        head=head_suppl,
        tail=tail_suppl,
        len_suppl_words=len_suppl_words,
        alpha=alpha,
        beta=beta,
        delta=delta,
    )

    if len_suppl_words == 1:
        states = get_words_from_context(
            g_lm, d_lm, context, beam_size, len_lacuna, head_suppl, tail_suppl, n
        )
        for s in states:
            heapq.heappush(beam, ([s], _score([s])))
        beam = heapq.nsmallest(beam_size, beam, key=lambda x: x[1])

    else:
        states = get_words_from_context(
            g_lm, d_lm, context, beam_size, 0, head_suppl, None, n
        )
        for s in states:
            score = score_candidate(
                g_lm,
                d_lm,
                lambda_weight,
                0,  # lunghezza lacuna ignorata per sequenze multi-parola
                lm_type,
                list(g_lm.vocab.lookup(context[(1 - n) :])),
                [s],
                head_suppl,
                tail_suppl,
                len_suppl_words,
                alpha,
                beta,
                delta,
            )
            heapq.heappush(beam, ([s], score))

        while True:
            successors = [
                succ
                for candidate in beam
                if len(candidate[0]) < len_suppl_words
                and len(candidate[0]) == len_suppl_words - 1
                for succ in get_successors(
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
            ]
            if not successors:
                break
            beam = heapq.nsmallest(beam_size, successors, key=lambda x: x[1])

    return get_best_candidates_from_beam(
        beam=beam,
        len_suppl_words=len_suppl_words,
        suppl_words=suppl_words,
        k_pred=k_pred,
        mod=mod,
    )

def get_best_K_predictions_from_context(
    g_lm: LanguageModel,
    d_lm: LanguageModel,
    context: list[str],
    len_lacuna: int,
    lambda_weight: float = LAMBDA,
    lm_type: str = LM_TYPE,
    len_suppl_words: int = 0,
    suppl_words: Optional[list[str]] = None,
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
    Calcola le migliori K predizioni per la lacuna data, usando la local beam
    search con interpolazione lineare tra modello globale e di dominio.

    Questo wrapper espone la stessa interfaccia di ``local_beam_search`` per
    compatibilità con ``accuracy.py``.

    Args:
        g_lm: Modello linguistico globale.
        d_lm: Modello linguistico di dominio.
        context: Contesto iniziale (lista di token).
        len_lacuna: Lunghezza della lacuna in caratteri.
        lambda_weight: Peso del modello di dominio. Default è LAMBDA.
        lm_type: Tipo di modello. Default è LM_TYPE.
        len_suppl_words: Numero di parole del supplemento. Default è 0.
        suppl_words: Gold label per il re-ranking. Default è ``None``.
        head_suppl: Prefisso atteso. Default è ``None``.
        tail_suppl: Suffisso atteso. Default è ``None``.
        n: Ordine degli n-grammi. Default è N.
        k_pred: Numero di predizioni da restituire. Default è K_PRED.
        beam_size: Dimensione del beam. Default è K_PRED * 4.
        mod: Modalità di estrazione. Default è ``"acc"``.
        alpha: Peso NLL. Default è ALPHA.
        beta: Peso edit distance. Default è BETA.
        delta: Peso length penalty. Default è DELTA.

    Returns:
        Lista delle migliori K sequenze generate.
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


# ---------------------------------------------------------------------------
# Utilità per i test cases MAAT
# ---------------------------------------------------------------------------


def get_context_from_test_case(
    test_case: str,
    n: int = N,
    case_folding: bool = True,
) -> tuple[list[str], Optional[str], Optional[str], int]:
    """
    Estrae contesto, testa, coda e lunghezza della lacuna da un test case MAAT.

    Il contesto è la sottosequenza di token a sinistra della lacuna ``[...]``,
    imbottita e troncata alla dimensione degli n-grammi.

    Args:
        test_case: Testo del test case contenente esattamente una lacuna ``[...]``.
        n: Ordine degli n-grammi del modello. Default è N.
        case_folding: Se ``True``, applica la normalizzazione maiuscolo/diacritici.

    Returns:
        Tupla ``(context, head, tail, lacuna_len)`` dove:
        - ``context``: lista di token per l'inferenza;
        - ``head``: prefisso della lacuna (può essere ``None``);
        - ``tail``: suffisso della lacuna (può essere ``None``);
        - ``lacuna_len``: numero di punti nella lacuna (stima dei caratteri mancanti).

    Raises:
        ValueError: Se il test case non contiene esattamente una lacuna.
    """
    match = test_case_contains_lacuna(test_case)
    if not match:
        raise ValueError(
            "Il test case non contiene una lacuna. "
            "Assicurati che sia presente esattamente una sequenza `[...]`."
        )

    # Testo a sinistra della `[` — rimuove tutto ciò che precede la `[` stessa
    left_of_lacuna = re.sub(r"[^\s]+\[", "[", test_case).split("[")[0]
    cleaned = clean_text_from_gaps(left_of_lacuna, case_folding=case_folding)

    context = list(
        flatten(
            pad_both_ends(get_tokens_from_clean_text(remove_punctuation(sent)), n=n)
            for sent in sentence_tokenizer.tokenize(cleaned)
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
    Verifica se il supplemento è idoneo alla valutazione nella top-K accuracy.

    Un supplemento è valido se: non è vuoto, contiene esattamente una parola
    e non include il token ``<UNK>``.

    Args:
        supplement: Lista di token del supplemento estratto dal training text.

    Returns:
        ``True`` se il supplemento è valido, ``False`` altrimenti.
    """
    return bool(supplement) and len(supplement) == 1 and "" not in supplement


def get_beam_size(k_predictions: int, beam_multiplier: int) -> int:
    """
    Calcola la dimensione del beam come ``clamp(k_predictions * beam_multiplier,
    MIN_BEAM_SIZE, MAX_BEAM_SIZE)``.

    Args:
        k_predictions: Numero di predizioni desiderate.
        beam_multiplier: Fattore moltiplicativo.

    Returns:
        Dimensione del beam compresa in ``[MIN_BEAM_SIZE, MAX_BEAM_SIZE]``.
    """
    return min(max(k_predictions * beam_multiplier, MIN_BEAM_SIZE), MAX_BEAM_SIZE)
