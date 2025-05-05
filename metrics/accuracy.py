import heapq
import re
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
    startswith: Optional[str] = None,
    endswith: Optional[str] = None,
) -> list[tuple[str, int]]:
    """
    Genera la distribuzione di frequenze delle parole per un dato contesto.
    Se la distribuzione è vuota, riduce progressivamente il contesto fino a trovare una distribuzione non vuota. (Backoff)

    :param lm: Modello di linguaggio con il metodo `context_counts`
    :param context: Lista di token che rappresenta il contesto
    :param n: Dimensione del modello (numero di parole nel contesto)
    :param startswith: Se specificato, filtra solo le parole che iniziano con questa sottostringa
    :param endswith: Se specificato, filtra solo le parole che finiscono con questa sottostringa
    :return: Lista ordinata di tuple (parola, frequenza)
    """
    freq_dist = lm.context_counts(lm.vocab.lookup(context[-(n - 1):]))
    context_counts = freq_dist.most_common(len(freq_dist)) 
    
    if startswith and endswith: 
       return sorted(context_counts, key=lambda x: x[0].startswith(startswith) and x[0].endswith(endswith), reverse=True) 
    elif startswith:
        return sorted(context_counts, key=lambda x: x[0].startswith(startswith), reverse=True)
    elif endswith:
        return sorted(context_counts, key=lambda x: x[0].endswith(endswith), reverse=True)
        
    return context_counts

def get_next_word_from_dist_freqs(
    lm: LanguageModel,
    context: list[str],
    words: set,
    head: Optional[str] = None,
    tail: Optional[str] = None,
    n=N,
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
        dist_freq_words_from_context = get_dist_freq_words_from_context(lm, context, n, head, tail)
        for word, _ in dist_freq_words_from_context:
            if word not in ["</s>", "<s>", "<UNK>"] and word not in words:
                return word
        n -= 1

    return None  # Caso in cui non trovo parole nuove nelle distribuzioni di frequenze del contesto


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

    Args:
        lm (LanguageModel): Modello di linguaggio utilizzato.
        context (list[str]): Contesto di parole.
        boundary (int): Numero massimo di parole da generare.
        n (int, opzionale): Numero di parole nel contesto. Default è N.

    Returns:
        list[str]: Lista di parole generate.
    """
    k_words = []
    dist_words = (
        set()
    )  # Parole già viste nelle distribuzioni di parole date dal contesto

    while len(k_words) < boundary:
        next_w = get_next_word_from_dist_freqs(lm, context, dist_words, head, tail, n)
        if next_w is None:
            return k_words  # Non ci sono più parole disponibili

        dist_words.add(next_w)
        k_words.append(next_w)

    return k_words


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
        distances = {
            tuple(seq): avg_edit_distance(seq, string_candidates)
            for seq in string_candidates
        }
    elif mod == "acc":
        if not suppl_words:
            raise ValueError("Definire il gold label")
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
    context: list[str],
    candidate: tuple[list[str], float],
    beam_size: int = K_PRED * 4,
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
    nexts = get_words_from_context(lm, context + candidate[0], beam_size, None, tail,  n)
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


def local_beam_search(
    lm: LanguageModel,
    context: list[str],
    lm_type: str = LM_TYPE,
    beam_size: int = K_PRED * 4,
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
    if len_suppl_words <= 1:  
        states = get_words_from_context(lm, context, beam_size, head_suppl, tail_suppl, n)
    else:
        states = get_words_from_context(lm, context, beam_size, head_suppl, None, n)
        
    for s in states:
        heapq.heappush(
            beam, ([s], loss(lm, lm.vocab.lookup((context)[(1 - n) :]), [s], lm_type))
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
                successors.extend(get_successors(lm, context, candidate, beam_size, tail_suppl, n))
            else: 
                successors.extend(get_successors(lm, context, candidate, beam_size, None, n))
        if not successors:
            break  # Nessun successore
        beam = heapq.nsmallest(beam_size, successors, key=lambda x: x[1])
    return get_best_candidates_from_beam(beam, suppl_words, mod, alpha, beta, k_pred)


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
    beam_size: int = K_PRED * 4,
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


def get_context(text: str, n=N, case_folding: bool = True) -> list[str]:
    """
    Ritorna una lista di token da utilizzare come contesto per l'inferenza del modello linguistico.

    Args:
        text (str): Testo di input da tokenizzare e processare.
        n (int, opzionale): Dimensione degli n-grammi da generare. Default è N.
        case_folding (bool, opzionale): Indica se convertire il testo in minuscolo durante il processo. Default è True.

    Returns:
        list[str]: Lista di token che rappresentano il contesto per l'inferenza del modello linguistico.

    Note:
        - Se il testo di input contiene parentesi quadre ("[") denota un test_case, perciò si delega la formazione del contesto alla funzione `get_context_from_test_case`.
        - Altrimenti, il testo viene pulito, suddiviso in frasi e ulteriormente processato per generare i token.
        - I token risultanti vengono imbottiti su entrambi i lati e troncati alla dimensione specificata degli n-grammi.
    """
    if "[" in text:
        return get_context_from_test_case(text, n, case_folding)

    return list(
        flatten(
            list(
                pad_both_ends(
                    get_tokens_from_clean_text(remove_punctuation(sent)),
                    n=n,
                )
            )
            for sent in sentence_tokenizer.tokenize(
                clean_text_from_gaps(text, case_folding)
            )
        )
    )[: (1 - n)]


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
        list[str]: contesto da cui partire per generare la predizione
    """
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
    Verifica se la lista di supplementi è valida.

    Questa funzione controlla se la lista di supplementi (token) fornita è vuota o
    contiene la stringa "<UNK>". Se una di queste condizioni è vera,
    la funzione restituisce False, altrimenti restituisce True.

    Args:
        supplement (list[str]): Una lista di tokens (stringhe) che rappresentano i supplementi.

    Returns:
        bool: True se la lista è valida, False altrimenti.
    """
    if not supplement or "<UNK>" in supplement:  # Supplementi non presenti nel testo
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
                    # Assumo che i supplements siano ordinati con i test_cases
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

                    if supplements[i] in predictions:
                        correct_predictions += 1

                    total_predictions += 1

    return round((correct_predictions / total_predictions) * 100, 2)
