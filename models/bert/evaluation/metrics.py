import numpy as np
import torch
from transformers import PreTrainedTokenizer

from bert_score import BERTScorer

from backend.core.preprocess import normalize_greek
from packages.hcb_infilling.hcb_infilling.metrics import (
    score_batch,
    compute_bertscore,
    default_scorer,
)

# Scorer lazy-singleton per evitare di ricaricare il modello ad ogni invocazione.
# Viene inizializzato la prima volta che si chiama evaluate_bertscore_text.
_text_scorer: BERTScorer | None = None


def _get_text_scorer() -> BERTScorer:
    """Restituisce un'istanza condivisa di BERTScorer (lazy init)."""
    global _text_scorer
    if _text_scorer is None:
        _text_scorer = BERTScorer(lang="el", rescale_with_baseline=False)
    return _text_scorer


def evaluate_topK_text(
    predictions_text: list[list[tuple[str, float]]],
    gold_labels: list[str] | list[list[str]],
) -> dict[str, float]:
    """
    Calcola le metriche top-K confrontando le stringhe normalizzate (lowercase)
    dei suggerimenti con la gold label.

    Args:
        predictions_text: batch di suggerimenti in formato testo.
            Ogni elemento è una lista di tuple (suggerimento_str, score)
            ordinata per score decrescente, come restituito da fill_mask
            con return_raw=False.
        gold_labels: batch di gold label in formato stringa.

    Returns:
        Dizionario con metriche top1, top3, top5, top10, top20 (percentuali).
    """
    count = 0
    # Numero massimo di suggerimenti per caso (beam size)
    max_k = max((len(preds) for preds in predictions_text), default=10)
    num_correct = np.zeros(max_k)

    for preds, gold in zip(predictions_text, gold_labels):
        if isinstance(gold, list):
            gold = " ".join(gold)

        # Normalizzazione coerente per gold e suggestions:
        # entrambi devono passare per normalize_greek con gli stessi parametri.
        gold_norm = (
            normalize_greek(
                text=gold,
                case_folding="fold",
                strip_diacritics_flag=True,
            )
            .replace(" ", "")
            .strip()
        )

        count += 1

        for rank, (suggestion, _score) in enumerate(preds):
            sugg_norm = (
                normalize_greek(
                    text=suggestion,
                    case_folding="fold",
                    strip_diacritics_flag=True,
                )
                .replace(" ", "")
                .strip()
            )

            if sugg_norm == gold_norm:
                num_correct[rank] += 1
                break  # conta solo il primo match

    if count == 0:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0, "top20": 0.0}

    cumulative = np.cumsum(num_correct)
    topk_metrics = {}
    for k in [1, 3, 5, 10, 20]:
        idx = min(k - 1, len(cumulative) - 1)
        topk_metrics[f"top{k}"] = (cumulative[idx] / count) * 100.0

    return topk_metrics


def evaluate_bertscore_text(
    predictions_text: list[list[tuple[str, float]]],
    gold_labels: list[str] | list[list[str]],
    scorer: BERTScorer | None = None,
) -> dict[str, float]:
    """
    Calcola BERTscore tra il suggerimento top-1 (decodificato) e la gold label.
    Legacy wrapper per evaluate_bertscore_topk_text.
    """
    res = evaluate_bertscore_topk_text(
        predictions_text, gold_labels, k_values=[1], scorer=scorer
    )
    return {
        "bertscore_precision": res.get("bertscore_precision_top1", 0.0),
        "bertscore_recall": res.get("bertscore_recall_top1", 0.0),
        "bertscore_f1": res.get("bertscore_f1_top1", 0.0),
    }


def evaluate_bertscore_topk_text(
    predictions_text: list[list[tuple[str, float]]],
    gold_labels: list[str] | list[list[str]],
    k_values: list[int] = [1, 3, 5, 10],
    scorer: BERTScorer | None = None,
) -> dict[str, float]:
    """
    Calcola il BERTscore@K (valore massimo tra i primi K suggerimenti)
    per diversi valori di K, ottimizzando le chiamate al modello.

    Args:
        predictions_text: batch di suggerimenti (lista di liste di tuple).
        gold_labels: batch di gold labels.
        k_values: lista di valori K da calcolare.
        scorer: istanza BERTScorer.

    Returns:
        Dizionario con precision, recall e f1 per ogni K.
    """
    if scorer is None:
        scorer = _get_text_scorer()

    max_k = max(k_values)
    all_cands: list[str] = []
    all_refs: list[str] = []
    # Mappa: (sample_idx, rank) -> index in all_cands
    mapping: dict[tuple[int, int], int] = {}

    for i, (preds, gold) in enumerate(zip(predictions_text, gold_labels)):
        if not preds:
            continue

        if isinstance(gold, list):
            gold = " ".join(gold)

        gold_norm = normalize_greek(
            text=gold, case_folding="fold", strip_diacritics_flag=True
        )

        for rank, (suggestion, _) in enumerate(preds[:max_k]):
            sugg_norm = normalize_greek(
                text=suggestion, case_folding="fold", strip_diacritics_flag=True
            )
            mapping[(i, rank)] = len(all_cands)
            all_cands.append(sugg_norm)
            all_refs.append(gold_norm)

    if not all_cands:
        return {f"bertscore_f1_top{k}": 0.0 for k in k_values}

    # Calcolo in un unico batch massivo
    P, R, F1 = scorer.score(all_cands, all_refs)

    num_samples = len(predictions_text)
    # Matrici per contenere i punteggi (N_samples x max_k)
    # Usiamo -1.0 per indicare l'assenza di un suggerimento
    scores_p = np.full((num_samples, max_k), -1.0)
    scores_r = np.full((num_samples, max_k), -1.0)
    scores_f1 = np.full((num_samples, max_k), -1.0)

    for (i, rank), flat_idx in mapping.items():
        scores_p[i, rank] = P[flat_idx].item()
        scores_r[i, rank] = R[flat_idx].item()
        scores_f1[i, rank] = F1[flat_idx].item()

    metrics = {}
    for k in k_values:
        # Per ogni sample, prendiamo il massimo tra i primi k suggerimenti disponibili
        # Usiamo nanmax e poi convertiamo i NaN in 0 se un sample non ha proprio suggerimenti
        with np.errstate(all="ignore"):
            # Slice fino a k
            slice_p = scores_p[:, :k]
            slice_r = scores_r[:, :k]
            slice_f1 = scores_f1[:, :k]

            # Maschera per i campioni che hanno almeno un suggerimento in questo range
            has_preds = np.any(slice_f1 != -1.0, axis=1)

            if np.any(has_preds):
                # Calcoliamo il massimo ignorando i -1.0
                # Invece di nanmax, usiamo np.where per ignorare i -1.0
                k_max_p = np.max(np.where(slice_p != -1.0, slice_p, -np.inf), axis=1)
                k_max_r = np.max(np.where(slice_r != -1.0, slice_r, -np.inf), axis=1)
                k_max_f1 = np.max(np.where(slice_f1 != -1.0, slice_f1, -np.inf), axis=1)

                metrics[f"bertscore_precision_top{k}"] = (
                    k_max_p[has_preds].mean() * 100.0
                )
                metrics[f"bertscore_recall_top{k}"] = k_max_r[has_preds].mean() * 100.0
                metrics[f"bertscore_f1_top{k}"] = k_max_f1[has_preds].mean() * 100.0
            else:
                metrics[f"bertscore_precision_top{k}"] = 0.0
                metrics[f"bertscore_recall_top{k}"] = 0.0
                metrics[f"bertscore_f1_top{k}"] = 0.0

    return metrics


def evaluate_topK(
    predictions_hcb_format: list[list[list[int | float]]],
    true_ids: list[list[int]],
    tokenizer: PreTrainedTokenizer,
) -> dict[str, float]:
    """
    Calcola le metriche top-K (Top-1, Top-3, Top-5, Top-10) per un batch.

    Args:
        predictions_hcb_format: batch di predizioni output di decode_modified_*.
            Formato: [
                [
                    [prob1, token1_1, token1_2, ...],
                    [prob2, token2_1, token2_2, ...]
                ], ...
            ]
        true_ids: batch di veri token ids della lacuna. Formato: [[id_1, id_2], ...]
        tokenizer: Il tokenizzatore (necessario per verificare pad_token_id).

    Returns:
        Dizionario con metriche top1, top3, top5, top10.
    """
    # Mappa i veri id a liste di int (se non lo sono già) in quanto la comparazione in score_batch
    # fa `if true_ids in sorted_suggestions`
    true_ids_list = []
    for ids in true_ids:
        if hasattr(ids, "tolist"):
            true_ids_list.append(ids.tolist())
        else:
            true_ids_list.append(list(ids))

    count, num_correct_ranks = score_batch(
        suggestions_batch=predictions_hcb_format,
        true_ids_batch=true_ids_list,
        tokenizer=tokenizer,
        method="topk",
    )

    if count == 0:
        return {"top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0}

    # cumulative_correct contiene i matches per ogni rank
    cumulative_correct = np.cumsum(num_correct_ranks)

    topk_metrics = {}
    for k in [1, 3, 5, 10]:
        idx = min(k - 1, len(cumulative_correct) - 1)
        topk_metrics[f"top{k}"] = (cumulative_correct[idx] / count) * 100.0

    return topk_metrics


def evaluate_bertscore_custom(
    predictions_hcb_format: list[list[list[int | float]]],
    true_ids: list[list[int]],
    masked_inputs: list[list[int]],
    masked_positions: list[int] | torch.Tensor,
    tokenizer: PreTrainedTokenizer,
    scorer=default_scorer,
) -> dict[str, float]:
    """
    Calcola la metrica BERTscore utilizzando `compute_bertscore` di hcb_infilling.

    Args:
        predictions_hcb_format: output di decode_modified_* contenente le suggestions
        true_ids: batch di veri token ids
        masked_inputs: il tensore o lista di input IDs con maschere iniziali
        masked_positions: un indice o maschera booleana delle posizioni coperte da maschera (comuni o array)
        tokenizer: Tokenizer
        scorer: Istanza di BERTScorer (default usa lang="en" da hcb_infilling)

    Returns:
        Dizionario con precision, recall e f1 mediati sul batch.
    """
    masked_inputs_tensor = (
        torch.tensor(masked_inputs)
        if not isinstance(masked_inputs, torch.Tensor)
        else masked_inputs.clone()
    )

    return compute_bertscore(
        suggestions=predictions_hcb_format,
        true_ids_batch=true_ids,
        masked_inputs_batch=masked_inputs_tensor,
        masked_positions=masked_positions,
        tokenizer=tokenizer,
        scorer=scorer,
    )
