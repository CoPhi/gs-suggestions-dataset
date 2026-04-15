from transformers import TrainerCallback

from models.bert.dataset.dev_set import build_dev_set
from models.bert.inference.predict import fill_mask
from models.bert.evaluation.metrics import (
    evaluate_topK_text,
    evaluate_bertscore_text,
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
    def __init__(self, dev_cases_pool, tokenizer, max_eval_cases=50):
        super().__init__()
        # Limitiamo il pool per non far durare ore ogni validazione
        self.dev_cases_pool = dev_cases_pool[:max_eval_cases]
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model, **kwargs):
        """
        Esegue la validazione HCB sul pool di casi di test.
        Calcola TopK (confronto normalizzato) e BERTscore (top-1 vs gold).
        """

        # Assicuriamoci che il modello sia in eval mode
        model.eval()

        if len(self.dev_cases_pool) == 0:
            return

        predictions_text = []  # list[list[tuple[str, float]]]
        gold_labels = []       # list[str]

        for case in self.dev_cases_pool:
            try:
                # fill_mask con return_raw=False restituisce tuple (str, score)
                suggestions = fill_mask(
                    text=case.x,
                    n_chars=case.gap_length,
                    model=model,
                    tokenizer=self.tokenizer,
                    K=10,
                    beam_size=10,
                    method="modified_best_to_worst",
                    return_raw=False,
                    case_folding=False,  # il testo uscirà dal tokenizer senza case folding forzato
                )

                predictions_text.append(suggestions)
                gold_labels.append(case.y)
            except Exception:
                pass  # skip se c'è un errore formativo su qualche lacuna particolare

        if not predictions_text:
            return

        # --- TopK (confronto normalizzato lowercase) ---
        top_k_metrics = evaluate_topK_text(
            predictions_text=predictions_text,
            gold_labels=gold_labels,
        )

        # --- BERTscore (top-1 vs gold label) ---
        bertscore_metrics = evaluate_bertscore_text(
            predictions_text=predictions_text,
            gold_labels=gold_labels,
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
            f"BERTscore F1: {bertscore_metrics.get('bertscore_f1', 0):.2f}%"
        )

        # Log su wandb se abilitato
        logs = {f"eval_hcb_{k}": v for k, v in all_metrics.items()}
        logs["step"] = state.global_step
        if wandb.run is not None:
            wandb.log(logs)
