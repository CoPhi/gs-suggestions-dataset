import numpy as np
import torch
from transformers import PreTrainedTokenizer

from bert_score import BERTScorer

from backend.core.preprocess import normalize_greek, remove_punctuation
from models.bert.finetuning import get_model_config
from packages.hcb_infilling.hcb_infilling.metrics import (
    score_batch,
)

_scorers: dict[str, BERTScorer] = {}


def get_scoring_model_for_training(training_checkpoint: str) -> str:
    """
    Seleziona un modello di scoring diverso da quello in training per evitare bias.
    """
    training_checkpoint = training_checkpoint.lower()

    # Mappa dei modelli disponibili, forzo tutti i modelli ad usare ancient-greek-bert per accentrare la valutazione usando il solito modello "terzo"
    models = {
        "CNR-ILC/gs-GreBerta": "pranaydeeps/Ancient-Greek-BERT",
        "CNR-ILC/gs-aristoBERTo": "pranaydeeps/Ancient-Greek-BERT",
        "CNR-ILC/gs-Logion": "pranaydeeps/Ancient-Greek-BERT",
    }

    if "CNR-ILC/gs-GreBerta" in training_checkpoint:
        return models["CNR-ILC/gs-GreBerta"]
    elif "CNR-ILC/gs-aristoBERTo" in training_checkpoint:
        return models["CNR-ILC/gs-aristoBERTo"]
    elif "CNR-ILC/gs-Logion" in training_checkpoint:
        return models["CNR-ILC/gs-Logion"]
    else:
        return models["CNR-ILC/gs-GreBerta"]


def _get_contextual_scorer(model_name: str) -> BERTScorer:
    global _scorers
    if model_name not in _scorers:
        _scorers[model_name] = BERTScorer(
            model_type=model_name, lang="el", rescale_with_baseline=True
        )
    return _scorers[model_name]


def reconstruct_context(
    context_with_gap: str, suggestion: str, window_size: int = 15
) -> str:
    """
    Sostituisce la lacuna [....] con il suggerimento e taglia il contesto.
    """

    pattern = r"\[\.+\]"
    reconstructed = re.sub(pattern, suggestion, context_with_gap)

    if window_size <= 0:
        return reconstructed

    words = reconstructed.split()
    if len(words) <= window_size * 2:
        return reconstructed

    return reconstructed


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
    contexts: list[str] | None = None,
    k_values: list[int] = [1, 3, 5, 10],
    scorer: BERTScorer | None = None,
    checkpoint: str | None = None,
) -> dict[str, float]:
    """
    Calcola il BERTscore@K (valore massimo tra i primi K suggerimenti)
    per diversi valori di K, ottimizzando le chiamate al modello.

    Args:
        predictions_text: batch di suggerimenti (lista di liste di tuple).
        gold_labels: batch di gold labels.
        k_values: lista di valori K da calcolare.
        scorer: istanza BERTScorer.
        checkpoint: percorso del checkpoint del modello BERT.

    Returns:
        Dizionario con precision, recall e f1 per ogni K.
    """
    if scorer is None:
        scoring_model = get_scoring_model_for_training(checkpoint or "default")
        scorer = _get_contextual_scorer(scoring_model)

    max_k = max(k_values)
    all_cands: list[str] = []
    all_refs: list[str] = []

    mapping: dict[tuple[int, int], int] = (
        {}
    )  # Mappa: (sample_idx, rank) -> index in all_cands

    config = get_model_config(checkpoint) if checkpoint else {}

    for i, (preds, gold, context) in enumerate(
        zip(predictions_text, gold_labels, contexts or [None] * len(gold_labels))
    ):
        if not preds:
            continue

        if isinstance(gold, list):
            gold = " ".join(gold)

        gold_norm = normalize_greek(
            text=gold,
            case_folding="fold",
            strip_diacritics_flag=config.get("strip_diacritics"),
        )

        if config.get("remove_punct"):
            gold_norm = remove_punctuation(gold_norm)

        gold_norm = gold_norm.replace(" ", "").strip()

        for rank, (suggestion, _) in enumerate(preds[:max_k]):
            if context:
                cand_sent = reconstruct_context(context, suggestion).strip()
                ref_sent = reconstruct_context(context, gold_norm).strip()
                all_cands.append(cand_sent)
                all_refs.append(ref_sent)
            else:
                sugg_norm = (
                    suggestion.strip()
                )  # i suggerimenti escono già normalizzati da fill_mask, ma facciamo un'ulteriore pulizia di spazi
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
        with np.errstate(all="ignore"):
            slice_f1 = scores_f1[:, :k]
            has_preds = np.any(slice_f1 != -1.0, axis=1)

            if np.any(has_preds):
                # --- MAX (Il migliore tra i primi K) ---
                k_max_f1 = np.max(np.where(slice_f1 != -1.0, slice_f1, -np.inf), axis=1)
                metrics[f"bertscore_f1_top{k}"] = k_max_f1[has_preds].mean() * 100.0

                # --- MEAN (Qualità media dei primi K) ---
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