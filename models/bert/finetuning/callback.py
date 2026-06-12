import traceback

from transformers import TrainerCallback

from models.bert.inference.predict import fill_mask, get_contextual_embeddings
from models.bert.evaluation.metrics import (
    evaluate_topK_text,
    evaluate_bertscore_topk_text,
    evaluate_cosine_similarity_topk,
    evaluate_contextual_similarity,
)
from backend.core.preprocess import normalize_greek
from models.bert.finetuning import get_model_config

import wandb


class CustomEvaluationCallback(TrainerCallback):
    """
    Callback personalizzato per calcolare le metriche TopK e BERTscore
    durante la fase di eval.
    Invece di valutare l'intero corpus, valutiamo un sottoinsieme (pool) di casi reali annotati ad ogni ciclo di on_evaluate.

    Il confronto tra suggerimenti e gold label avviene in modalità normalizzata
    (lowercase, spazi rimossi) per garantire invarianza al casing del modello.
    """

    def __init__(self, dev_cases_pool, tokenizer, checkpoint: str, max_eval_cases=50):
        super().__init__()
        self.dev_cases_pool = dev_cases_pool[:max_eval_cases]
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint

    def on_evaluate(self, args, state, control, model, **kwargs):
        """
        Esegue la validazione sul pool di casi di test.
        Calcola TopK e BERTscore@K (massimo tra i primi K suggerimenti).
        """

        model.eval()

        if len(self.dev_cases_pool) == 0:
            return

        predictions_text = []
        gold_labels = []
        contexts = []

        for case in self.dev_cases_pool:
            try:
                suggestions = fill_mask(
                    text=case.x,
                    checkpoint=self.checkpoint,
                    n_chars=case.gap_length,
                    model=model,
                    tokenizer=self.tokenizer,
                    K=20,
                    beam_size=50,
                    method="modified_best_to_worst",
                    return_raw=False,
                )

                predictions_text.append(suggestions)
                gold_labels.append(case.y)
                contexts.append(case.x)
            except Exception as e:
                print(f"[Evaluation Error] fill_mask ha generato un'eccezione: {e}")
                print(f"[Evaluation Error] Case: {case}")
                continue

        if not predictions_text:
            return

        config = get_model_config(self.checkpoint)
        is_strip = config.get("strip_diacritics", True)
        case_fold = config.get("case_folding", "fold")

        normalized_gold_labels = []
        for case_y in gold_labels:
            if isinstance(case_y, list):
                norm_y = [
                    normalize_greek(
                        y, case_folding=case_fold, strip_diacritics_flag=is_strip
                    )
                    for y in case_y
                ]
            else:
                norm_y = normalize_greek(
                    case_y, case_folding=case_fold, strip_diacritics_flag=is_strip
                )
            normalized_gold_labels.append(norm_y)

        # Calcolo Top-K EM
        try:
            topk_metrics = evaluate_topK_text(
                predictions_text=predictions_text, gold_labels=normalized_gold_labels
            )
        except Exception as e:
            print(f"[Evaluation Error] evaluate_topK_text fallito: {e}")
            topk_metrics = {}

        # BERTscore@K (massimo tra i primi K)
        try:
            bertscore_metrics = evaluate_bertscore_topk_text(
                predictions_text=predictions_text,
                gold_labels=normalized_gold_labels,
                contexts=contexts,
                k_values=[1, 5, 10, 20],
                checkpoint=self.checkpoint,
            )
        except Exception as e:
            print(
                f"[Evaluation Error] evaluate_bertscore_topk_text ha generato un'eccezione: {e}"
            )
            print("[Evaluation Error] Traceback completo:")
            print(traceback.format_exc())
            bertscore_metrics = {}

        # Cosine Similarity @K
        all_similarities = []
        for i, case in enumerate(self.dev_cases_pool):
            if i < len(predictions_text) and predictions_text[i]:
                cand_texts = [s[0] for s in predictions_text[i]]
                gold_text = " ".join(case.y) if isinstance(case.y, list) else case.y
                all_texts_to_embed = cand_texts + [gold_text]

                try:
                    embs = get_contextual_embeddings(
                        text_with_gap=case.x,
                        candidates=all_texts_to_embed,
                        model=model,
                        tokenizer=self.tokenizer,
                        checkpoint=self.checkpoint,
                    )
                    gold_emb = embs[-1]
                    cand_embs = embs[:-1]

                    similarities = evaluate_contextual_similarity(cand_embs, gold_emb)
                    all_similarities.append(similarities)
                except Exception as e:
                    print(
                        f"[Evaluation Error] Cosine Similarity fallita per case {case}: {e}"
                    )
                    all_similarities.append([])
            else:
                all_similarities.append([])

        try:
            cos_sim_metrics = evaluate_cosine_similarity_topk(
                similarities_list=all_similarities, k_values=[1, 5, 10, 20]
            )
        except Exception as e:
            print(f"[Evaluation Error] evaluate_cosine_similarity_topk fallito: {e}")
            cos_sim_metrics = {}

        # Unione dei risultati e calcolo metrica composita
        all_metrics = {**topk_metrics, **bertscore_metrics, **cos_sim_metrics}

        top1_em = all_metrics.get("top1", 0)
        cossim_max_top1 = all_metrics.get("cos_sim_top1_max", 0)
        all_metrics["composite_score"] = (top1_em + cossim_max_top1) / 2

        # Aggiornamento dello stato del Trainer
        eval_callback_logs = {f"eval_{k}": v for k, v in all_metrics.items()}
        state.log_history[-1].update(eval_callback_logs)

        # Stampa a terminale in formato tabellare
        c_score = all_metrics.get("composite_score", 0.0)

        print("\n" + "=" * 80)
        print(
            f" EVALUATION | Epoch: {state.epoch:<5} | Step: {state.global_step:<6} | Composite Score: {c_score:.2f}%"
        )
        print("=" * 80)
        print(f"{'Metric':<25} | {'@1':<10} | {'@5':<10} | {'@10':<10} | {'@20':<10}")
        print("-" * 80)

        rows = [
            ("Exact Match", "top1", "top5", "top10", "top20"),
            (
                "BERTScore F1 (Max)",
                "bertscore_f1_top1",
                "bertscore_f1_top5",
                "bertscore_f1_top10",
                "bertscore_f1_top20",
            ),
            (
                "BERTScore F1 (Mean)",
                "bertscore_f1_top1_mean",
                "bertscore_f1_top5_mean",
                "bertscore_f1_top10_mean",
                "bertscore_f1_top20_mean",
            ),
            (
                "CosSim (Max)",
                "cos_sim_top1_max",
                "cos_sim_top5_max",
                "cos_sim_top10_max",
                "cos_sim_top20_max",
            ),
            (
                "CosSim (Mean)",
                "cos_sim_top1_mean",
                "cos_sim_top5_mean",
                "cos_sim_top10_mean",
                "cos_sim_top20_mean",
            ),
        ]

        for name, k1, k5, k10, k20 in rows:
            v1 = all_metrics.get(k1, 0.0)
            v5 = all_metrics.get(k5, 0.0)
            v10 = all_metrics.get(k10, 0.0)
            v20 = all_metrics.get(k20, 0.0)
            print(
                f"{name:<25} | {v1:>8.2f}% | {v5:>8.2f}% | {v10:>8.2f}% | {v20:>8.2f}%"
            )

        print("=" * 80 + "\n")

        # log su wandb
        logs = {f"eval/{k}": v for k, v in all_metrics.items()}
        logs["eval_composite_score"] = all_metrics["composite_score"]
        logs["train/global_step"] = state.global_step
        logs["epoch"] = state.epoch

        # Log delle previsioni sotto forma di tabella
        if wandb.run is not None:
            columns = ["Epoch", "Step", "Context", "Gold Label", "Top Predictions"]
            data = []

            # Logghiamo ad esempio i primi 15 casi
            for i, case in enumerate(self.dev_cases_pool[:15]):
                gold = ", ".join(case.y) if isinstance(case.y, list) else case.y
                # Formattiamo i primi 5 suggerimenti con i relativi punteggi
                top_preds = " | ".join(
                    [f"'{s}' ({score:.2f})" for s, score in predictions_text[i][:5]]
                )
                data.append([state.epoch, state.global_step, case.x, gold, top_preds])

            # Log della tabella per lo step corrente
            table = wandb.Table(columns=columns, data=data)
            logs["eval/predictions_table"] = table
            wandb.log(logs)
