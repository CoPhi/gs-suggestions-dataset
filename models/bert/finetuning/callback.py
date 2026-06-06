import traceback

from transformers import TrainerCallback

from models.bert.inference.predict import fill_mask
from models.bert.evaluation.metrics import (
    evaluate_topK_text,
    evaluate_bertscore_topk_text,
)

import wandb


class HCBEvaluationCallback(TrainerCallback):
    """
    Callback personalizzato per calcolare le metriche TopK e BERTscore tramite HCB
    durante la fase di eval.
    Invece di valutare l'intero corpus con HCB (che rallenterebbe enormemente il training),
    valutiamo un sottoinsieme (pool) di casi reali annotati ad ogni ciclo di on_evaluate.

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
        Esegue la validazione HCB sul pool di casi di test.
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
                    beam_size=20,
                    method="modified_best_to_worst",
                    return_raw=False,
                )

                predictions_text.append(suggestions)
                gold_labels.append(case.y)
                contexts.append(case.x)
            except Exception as e:
                print(f"[HCB Error] fill_mask ha generato un'eccezione: {e}")
                print(f"[HCB Error] Case: {case}")
                continue

        if not predictions_text:
            return

        # Calcolo Top-K EM
        try:
            topk_metrics = evaluate_topK_text(
                predictions_text=predictions_text, gold_labels=gold_labels
            )
        except Exception as e:
            print(f"[HCB Error] evaluate_topK_text fallito: {e}")
            topk_metrics = {}

        # BERTscore@K (massimo tra i primi K)
        try:
            bertscore_metrics = evaluate_bertscore_topk_text(
                predictions_text=predictions_text,
                gold_labels=gold_labels,
                contexts=contexts,
                k_values=[1, 5, 10, 20],
                checkpoint=self.checkpoint,
            )
        except Exception as e:
            print(
                f"[HCB Error] evaluate_bertscore_topk_text ha generato un'eccezione: {e}"
            )
            print("[HCB Error] Traceback completo:")
            print(traceback.format_exc())
            bertscore_metrics = {}

        # Unione dei risultati
        all_metrics = {**topk_metrics, **bertscore_metrics}

        # Aggiornamento dello stato del Trainer
        hcb_logs = {f"eval_{k}": v for k, v in all_metrics.items()}
        state.log_history[-1].update(hcb_logs)

        print(
            f"[HCB Val] "
            f"Top-1 EM: {all_metrics.get('top1', 0):.2f}% | "
            f"Top-5 EM: {all_metrics.get('top5', 0):.2f}% | "
            f"Top-10 EM: {all_metrics.get('top10', 0):.2f}% | "
            f"Top-20 EM: {all_metrics.get('top20', 0):.2f}% | "
            f"BS-F1@1: {bertscore_metrics.get('bertscore_f1_top1', 0):.2f}% (mean: {bertscore_metrics.get('bertscore_f1_top1_mean', 0):.2f}%)| "
            f"BS-F1@5: {bertscore_metrics.get('bertscore_f1_top5', 0):.2f}% (mean: {bertscore_metrics.get('bertscore_f1_top5_mean', 0):.2f}%) | "
            f"BS-F1@10: {bertscore_metrics.get('bertscore_f1_top10', 0):.2f}% (mean: {bertscore_metrics.get('bertscore_f1_top10_mean', 0):.2f}%) | "
            f"BS-F1@20: {bertscore_metrics.get('bertscore_f1_top20', 0):.2f}% (mean: {bertscore_metrics.get('bertscore_f1_top20_mean', 0):.2f}%)"
        )

        # log su wandb
        logs = {f"eval/{k}": v for k, v in all_metrics.items()}
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
