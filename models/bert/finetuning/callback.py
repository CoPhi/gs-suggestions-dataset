from transformers import TrainerCallback

from models.bert.dataset.dev_set import build_dev_set
from models.bert.inference.predict import fill_mask
from models.bert.evaluation.metrics import evaluate_topK

import wandb

class HCBEvaluationCallback(TrainerCallback):
    """
    Callback personalizzato per calcolare le metriche TopK tramite HCB durante la fase di eval.
    Invece di valutare l'intero corpus con HCB (che rallenterebbe enormemente il training),
    valutiamo un sottoinsieme (pool) di casi reali annotati ad ogni ciclo di on_evaluate.
    """
    def __init__(self, dev_cases_pool, tokenizer, max_eval_cases=50):
        super().__init__()
        # Limitiamo il pool per non far durare ore ogni validazione
        self.dev_cases_pool = dev_cases_pool[:max_eval_cases]
        self.tokenizer = tokenizer

    def on_evaluate(self, args, state, control, model, **kwargs):
        """
        Esegue la validazione HCB sul pool di casi di test.
        """

        # Assicuriamoci che il modello sia in eval mode
        model.eval()
        device = next(model.parameters()).device
        
        predictions_ids = []
        true_ids_batch = []
        
        if len(self.dev_cases_pool) == 0:
            return

        for case in self.dev_cases_pool:
            try:
                suggestions = fill_mask(
                    case=case,
                    tokenizer=self.tokenizer,
                    model=model,
                    num_suggestions=10,
                    mask_token=self.tokenizer.mask_token or "[MASK]",
                    method="modified_best_to_worst",
                    device=device
                )
                
                y_tokens_ids = self.tokenizer(case.y, add_special_tokens=False)["input_ids"]
                
                beam_preds = []
                for s in suggestions:
                    beam_preds.append([s[0]] + s[1])
                    
                predictions_ids.append(beam_preds)
                true_ids_batch.append(y_tokens_ids)
            except Exception as e:
                pass # skip se c'è un errore formativo su qualche lacuna particolare

        if predictions_ids:
            top_k_metrics = evaluate_topK(
                predictions_hcb_format=predictions_ids,
                true_ids=true_ids_batch,
                tokenizer=self.tokenizer
            )
            
            # Log su wandb se abilitato e stampiamo su CLI
            print(f"[HCB Val] Top1: {top_k_metrics.get('top1', 0):.2f}% | Top5: {top_k_metrics.get('top5', 0):.2f}%")
            
            logs = { f"eval_hcb_{k}": v for k, v in top_k_metrics.items() }
            logs["step"] = state.global_step
            if wandb.run is not None:
                wandb.log(logs)
