from nltk.lm.models import LanguageModel
from tqdm import tqdm
from config.settings import K_PRED, BATCH_SIZE, N, LAMBDA, ALPHA, BETA, DELTA
from predictions.ngrams import check_supplement, get_beam_size, get_best_K_predictions_from_context, get_context_from_test_case
from utils.preprocess import clean_supplements

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

                # non ci sono blocchi da predire
                if not supplements:
                    continue  

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
