import numpy as np
import torch
from transformers import PreTrainedTokenizer

from packages.hcb_infilling.hcb_infilling.metrics import (
    score_batch,
    compute_bertscore,
    default_scorer
)

def evaluate_topK(
    predictions_hcb_format: list[list[list[int | float]]], 
    true_ids: list[list[int]], 
    tokenizer: PreTrainedTokenizer
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
    # Mappa i veri id a liste (se non lo sono già) in quanto la comparazione in score_batch
    # fa `if true_ids in sorted_suggestions`
    true_ids_list = [list(ids) for ids in true_ids]

    count, num_correct_ranks = score_batch(
        suggestions_batch=predictions_hcb_format,
        true_ids_batch=true_ids_list,
        tokenizer=tokenizer,
        method='topk'
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
    scorer=default_scorer
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
    masked_inputs_tensor = torch.tensor(masked_inputs) if not isinstance(masked_inputs, torch.Tensor) else masked_inputs.clone()
    
    return compute_bertscore(
        suggestions=predictions_hcb_format,
        true_ids_batch=true_ids,
        masked_inputs_batch=masked_inputs_tensor,
        masked_positions=masked_positions,
        tokenizer=tokenizer,
        scorer=scorer
    )
