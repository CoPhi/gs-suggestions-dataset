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
        # Limitiamo il pool per non far durare ore ogni validazione
        self.dev_cases_pool = dev_cases_pool[:max_eval_cases]
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint

    def on_evaluate(self, args, state, control, model, **kwargs):
        """
        Esegue la validazione HCB sul pool di casi di test.
        Calcola TopK e BERTscore@K (massimo tra i primi K suggerimenti).
        """

        # Assicuriamoci che il modello sia in eval mode
        model.eval()

        if len(self.dev_cases_pool) == 0:
            return

        predictions_text = []  # list[list[tuple[str, float]]]
        gold_labels = []  # list[str]

        for case in self.dev_cases_pool:
            try:
                # fill_mask con return_raw=False restituisce tuple (str, score)
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
            except Exception as e:
                print(f"[HCB Error] fill_mask ha generato un'eccezione: {e}")
                print(f"[HCB Error] Case: {case}")
                continue

        if not predictions_text:
            return

        # --- TopK (confronto normalizzato lowercase) ---
        top_k_metrics = evaluate_topK_text(
            predictions_text=predictions_text,
            gold_labels=gold_labels,
        )

        # --- BERTscore@K (massimo tra i primi K) ---
        bertscore_metrics = evaluate_bertscore_topk_text(
            predictions_text=predictions_text,
            gold_labels=gold_labels,
            k_values=[1, 3, 5, 10],
        )

        # Unisci tutte le metriche
        all_metrics = {**top_k_metrics, **bertscore_metrics}

        # LOG nel Trainer state (visibile in log_history)
        hcb_logs = {f"eval_hcb_{k}": v for k, v in all_metrics.items()}
        state.log_history[-1].update(hcb_logs)

        # Stampa su CLI
        print(
            f"[HCB Val] "
            f"Top1: {top_k_metrics.get('top1', 0):.2f}% | "
            f"Top5: {top_k_metrics.get('top5', 0):.2f}% | "
            f"BS-F1@1: {bertscore_metrics.get('bertscore_f1_top1', 0):.2f}% | "
            f"BS-F1@5: {bertscore_metrics.get('bertscore_f1_top5', 0):.2f}% | "
            f"BS-F1@10: {bertscore_metrics.get('bertscore_f1_top10', 0):.2f}%"
        )

        # Log su wandb
        logs = {f"eval/hcb_{k}": v for k, v in all_metrics.items()}
        logs["train/global_step"] = state.global_step
        logs["epoch"] = state.epoch
        if wandb.run is not None:
            wandb.log(logs)
