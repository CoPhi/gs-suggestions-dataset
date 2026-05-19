import re
import numpy as np
import torch
from bert_score import BERTScorer

from backend.core.preprocess import normalize_greek, remove_punctuation
from models.bert.finetuning import get_model_config

_scorers: dict[str, BERTScorer] = {}

# Modelli con baseline pre-calcolata nella libreria bert-score.
# Per tutti gli altri (es. pranaydeeps/Ancient-Greek-BERT) rescale_with_baseline=False.
_MODELS_WITH_BASELINE = {
    "bert-base-uncased",
    "bert-base-cased",
    "bert-large-uncased",
    "bert-large-cased",
    "roberta-base",
    "roberta-large",
    "xlm-roberta-base",
    "xlm-roberta-large",
    "xlnet-base-cased",
    "xlnet-large-cased",
    "microsoft/deberta-xlarge-mnli",
    "microsoft/deberta-large-mnli",
}

# Numero di layer da usare per modelli non registrati in bert_score.model2layers.
# pranaydeeps/Ancient-Greek-BERT è un BERT-base (12 layer); usiamo layer 9
# come da best practice bert_score per BERT-base (L=9 massimizza correlazione umana).
_MODEL_NUM_LAYERS: dict[str, int] = {
    "pranaydeeps/Ancient-Greek-BERT": 9,
}


def get_scoring_model_for_training(training_checkpoint: str) -> str:
    """
    Seleziona un modello di scoring diverso da quello in training per evitare bias.
    """
    models = {
        "cnr-ilc/gs-greberta": "pranaydeeps/Ancient-Greek-BERT",
        "cnr-ilc/gs-aristoberto": "pranaydeeps/Ancient-Greek-BERT",
        "cnr-ilc/gs-logion": "pranaydeeps/Ancient-Greek-BERT",
    }

    key = training_checkpoint.lower()
    for k, v in models.items():
        if k in key:
            return v

    return "pranaydeeps/Ancient-Greek-BERT"


def _get_contextual_scorer(model_name: str) -> BERTScorer:
    global _scorers
    if model_name not in _scorers:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _scorers[model_name] = BERTScorer(
            model_type=model_name,
            lang="el",
            device=device,
            num_layers=_MODEL_NUM_LAYERS.get(model_name),
            rescale_with_baseline=model_name in _MODELS_WITH_BASELINE,
        )
    return _scorers[model_name]


def reset_scorer_cache() -> None:
    """
    Svuota la cache degli scorer. Da chiamare all'inizio di ogni run di training
    per evitare che istanze vecchie vengano riutilizzate in ambienti long-running.
    """
    global _scorers
    _scorers.clear()


def reconstruct_context(
    context_with_gap: str, suggestion: str, window_size: int = 1
) -> str:
    """
    Sostituisce la lacuna [....] con il suggerimento e taglia il contesto
    per mantenere un massimo di 'window_size' parole a destra e a sinistra.
    Per il momento si mantiene una sola parola di contesto in tutte e due le direzioni,
    per evitare impennate nelle valutazioni delle metriche BERTscore.
    """
    pattern = r"\[\.+\]"

    reconstructed = re.sub(pattern, lambda _: suggestion, context_with_gap)
    
    if window_size <= 0:
        return reconstructed

    words = reconstructed.split()
    if len(words) <= window_size * 2:
        return reconstructed

    # Troviamo la posizione della lacuna per centrare la finestra
    match = re.search(pattern, context_with_gap)
    if not match:
        return reconstructed

    # Contiamo quante parole ci sono a sinistra della lacuna nel contesto originale
    left_context = context_with_gap[: match.start()]
    left_words_count = len(left_context.split())

    # Calcoliamo gli indici di taglio della finestra
    start_idx = max(0, left_words_count - window_size)

    # Se il suggerimento contiene spazi (più parole), aggiustiamo l'offset destro
    suggestion_len = len(suggestion.split()) if suggestion.strip() else 1
    end_idx = min(len(words), left_words_count + suggestion_len + window_size)

    return " ".join(words[start_idx:end_idx])


def evaluate_topK_text(
    predictions_text: list[list[tuple[str, float]]],
    gold_labels: list[str] | list[list[str]],
) -> dict[str, float]:
    """
    Calcola le metriche top-K confrontando le stringhe normalizzate (lowercase)
    dei suggerimenti con la gold label, neutralizzando artefatti di tokenizzazione.
    """
    count = 0
    max_k = max((len(preds) for preds in predictions_text), default=10)
    num_correct = np.zeros(max_k)

    def _strict_sanitize(text: str) -> str:
        # Pulisce la stringa da spazi e caratteri invisibili (es. zero-width space)
        return re.sub(r'[\s\u200B-\u200D\uFEFF]', '', text).strip()

    for preds, gold in zip(predictions_text, gold_labels):
        if isinstance(gold, list):
            gold = " ".join(gold)

        # Normalizzazione Gold
        gold_norm = normalize_greek(
            text=gold,
            case_folding="fold",
            strip_diacritics_flag=True,
        )
        gold_norm = _strict_sanitize(gold_norm)

        count += 1

        for rank, (suggestion, _score) in enumerate(preds):
            
            sugg_norm = normalize_greek(
                text=suggestion,
                case_folding="fold",
                strip_diacritics_flag=True,
            )
            sugg_norm = _strict_sanitize(sugg_norm)

            if sugg_norm == gold_norm:
                num_correct[rank] += 1
                break

    if count == 0:
        return {"top1": 0.0, "top5": 0.0, "top10": 0.0, "top20": 0.0}

    cumulative = np.cumsum(num_correct)
    topk_metrics = {}
    for k in [1, 5, 10, 20]:
        idx = min(k - 1, len(cumulative) - 1)
        topk_metrics[f"top{k}"] = (cumulative[idx] / count) * 100.0

    return topk_metrics

def evaluate_bertscore_text(
    predictions_text: list[list[tuple[str, float]]],
    gold_labels: list[str] | list[list[str]],
    scorer: BERTScorer | None = None,
) -> dict[str, float]:
    """
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
    contexts: list[str] | None = None,
    k_values: list[int] = [1, 3, 5, 10],
    scorer: BERTScorer | None = None,
    checkpoint: str | None = None,
) -> dict[str, float]:
    """
    Calcola il BERTscore@K (valore massimo tra i primi K suggerimenti)
    per diversi valori di K, ottimizzando le chiamate al modello.
    """
    if scorer is None:
        scoring_model = get_scoring_model_for_training(checkpoint or "default")
        scorer = _get_contextual_scorer(scoring_model)

    max_k = max(k_values)
    all_cands: list[str] = []
    all_refs: list[str] = []

    # Mappa: (sample_idx, rank) -> index in all_cands
    mapping: dict[tuple[int, int], int] = {}

    config = get_model_config(checkpoint) if checkpoint else {}

    for i, (preds, gold) in enumerate(zip(predictions_text, gold_labels)):
        if not preds:
            continue

        context = contexts[i] if contexts is not None else None

        if isinstance(gold, list):
            gold = " ".join(gold)

        gold_norm = normalize_greek(
            text=gold,
            case_folding="fold",
            strip_diacritics_flag=config.get("strip_diacritics"),
        )

        if config.get("remove_punct"):
            gold_norm = remove_punctuation(gold_norm)

        gold_norm = gold_norm.strip()

        for rank, (suggestion, _) in enumerate(preds[:max_k]):
            flat_idx = len(all_cands)

            if context:
                cand_sent = reconstruct_context(context, suggestion).strip()
                ref_sent = reconstruct_context(context, gold_norm).strip()
            else:
                cand_sent = suggestion.strip()
                ref_sent = gold_norm

            mapping[(i, rank)] = flat_idx
            all_cands.append(cand_sent)
            all_refs.append(ref_sent)

    if not all_cands:
        return {f"bertscore_f1_top{k}": 0.0 for k in k_values}

    P, R, F1 = scorer.score(all_cands, all_refs)

    num_samples = len(predictions_text)
    scores_p = np.full((num_samples, max_k), -1.0)
    scores_r = np.full((num_samples, max_k), -1.0)
    scores_f1 = np.full((num_samples, max_k), -1.0)

    for (i, rank), flat_idx in mapping.items():
        scores_p[i, rank] = P[flat_idx].item()
        scores_r[i, rank] = R[flat_idx].item()
        scores_f1[i, rank] = F1[flat_idx].item()

    metrics = {}
    for k in k_values:
        with np.errstate(all="ignore"):
            slice_f1 = scores_f1[:, :k]
            has_preds = np.any(slice_f1 != -1.0, axis=1)

            if np.any(has_preds):
                k_max_f1 = np.max(np.where(slice_f1 != -1.0, slice_f1, -np.inf), axis=1)
                metrics[f"bertscore_f1_top{k}"] = k_max_f1[has_preds].mean() * 100.0

                k_mean_f1 = np.nanmean(
                    np.where(slice_f1 != -1.0, slice_f1, np.nan), axis=1
                )
                metrics[f"bertscore_f1_top{k}_mean"] = (
                    np.nanmean(k_mean_f1[has_preds]) * 100.0
                )
            else:
                metrics[f"bertscore_f1_top{k}"] = 0.0
                metrics[f"bertscore_f1_top{k}_mean"] = 0.0

    return metrics


def evaluate_contextual_similarity(
    candidate_embeddings: list[torch.Tensor],
    gold_embedding: torch.Tensor,
) -> list[float]:
    """
    Calcola la similarità coseno tra ciascun candidato (già aggregato in un vettore 1D
    tramite mean-pooling) e la gold label.
    Ottimizzato per elaborazione vettoriale in parallelo (singolo operatore CUDA).
    
    Args:
        candidate_embeddings: Lista di tensori [hidden_dim] generati da get_contextual_embeddings
        gold_embedding: Tensore [hidden_dim] della gold label
        
    Returns:
        Lista di similarità coseno [-1, 1] (float).
    """
    import torch.nn.functional as F
    import torch
    
    if not candidate_embeddings:
        return []
        
    # Riconduciamo tutti i candidati a vettori 1D [hidden_dim] (fallback di sicurezza)
    pooled_candidates = [
        torch.mean(emb, dim=0) if emb.dim() != 1 else emb 
        for emb in candidate_embeddings
    ]
    
    # Impiliamo in un unico tensore 2D: [num_candidates, hidden_dim]
    candidates_tensor = torch.stack(pooled_candidates) 
    gold_unsqueezed = gold_embedding.unsqueeze(0) # Shape: [1, hidden_dim]
    
    # Calcolo in parallelo su GPU tramite un unico kernel CUDA
    similarities = F.cosine_similarity(candidates_tensor, gold_unsqueezed, dim=1)
    
    return similarities.cpu().tolist()


def evaluate_cosine_similarity_topk(
    similarities_list: list[list[float]],
    k_values: list[int] = [1, 5, 10, 20],
) -> dict[str, float]:
    """
    Calcola la Cosine Similarity @K (massima e media) per diversi valori di K.
    
    Args:
        similarities_list: Lista contenente, per ogni sample, la lista delle 
                           similarità coseno tra i candidati e la gold label.
        k_values: Lista di valori K per cui calcolare le metriche.
        
    Returns:
        Dizionario con metriche 'cos_sim_topK_max' e 'cos_sim_topK_mean' (in %).
    """
    import numpy as np
    
    if not similarities_list:
        return {f"cos_sim_top{k}_{metric}": 0.0 for k in k_values for metric in ["max", "mean"]}
        
    max_k = max(k_values)
    num_samples = len(similarities_list)
    
    # Riempiamo una matrice (num_samples, max_k) con NaN per gestire array sbilanciati
    scores = np.full((num_samples, max_k), np.nan)
    
    for i, sims in enumerate(similarities_list):
        k_limit = min(len(sims), max_k)
        if k_limit > 0:
            scores[i, :k_limit] = sims[:k_limit]
            
    metrics = {}
    for k in k_values:
        # Punteggi top-K per il K corrente
        slice_k = scores[:, :k]
        
        with np.errstate(all="ignore"):
            # Righe valide (campioni che hanno almeno una similarità tra i primi K)
            valid_rows = ~np.isnan(slice_k).all(axis=1)
            
            if np.any(valid_rows):
                # Max @K (ignora i NaN)
                k_max = np.nanmax(slice_k[valid_rows], axis=1)
                metrics[f"cos_sim_top{k}_max"] = k_max.mean() * 100.0
                
                # Mean @K
                k_mean = np.nanmean(slice_k[valid_rows], axis=1)
                metrics[f"cos_sim_top{k}_mean"] = k_mean.mean() * 100.0
            else:
                metrics[f"cos_sim_top{k}_max"] = 0.0
                metrics[f"cos_sim_top{k}_mean"] = 0.0
                
    return metrics


