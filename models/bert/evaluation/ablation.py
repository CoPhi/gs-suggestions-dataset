import os
import re
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer

from models.bert.dataset.dev_set import DevCase
from models.bert.inference.predict import fill_mask
from models.bert.evaluation.metrics import evaluate_topK

try:
    from bert_score import BERTScorer
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

def build_dev_cases(eval_dataset, max_cases=100):
    """Estrae i casi di test con lacune piccole (1-5 caratteri)."""
    dev_cases = []
    # Seleziona lo split 'test' se presente, sennò 'dev'
    split_name = "test" if "test" in eval_dataset else "dev"
    
    for row in eval_dataset[split_name].to_list():
        if 1 <= row["gap_length"] <= 5:
            # Assicuriamoci che 'x' contenga effettivamente il placeholder
            if re.search(r'\[\.+\]', row["x"]):
                dev_cases.append(
                    DevCase(
                        x=row["x"],
                        y=row["y"],
                        gap_length=row["gap_length"],
                        corpus_id=row["corpus_id"],
                        file_id=row["file_id"]
                    )
                )
            if len(dev_cases) >= max_cases:
                break
    return dev_cases

def truncate_context(text: str, tokenizer, window_size: int) -> str:
    """
    Tronca il testo lasciando al massimo `window_size` sub-word tokens
    a sinistra e a destra della lacuna (rappresentata come `[.....]`).
    Se window_size è -1, non fa alcun cropping.
    """
    if window_size < 0:
        return text
        
    match = re.search(r'\[\.+\]', text)
    if not match:
        return text
    
    start_idx = match.start()
    end_idx = match.end()
    
    left_context = text[:start_idx]
    gap_placeholder = text[start_idx:end_idx]
    right_context = text[end_idx:]
    
    # Nessun special token così il testo si ricollega fedelmente
    left_tokens = tokenizer.encode(left_context, add_special_tokens=False)
    right_tokens = tokenizer.encode(right_context, add_special_tokens=False)
    
    if window_size > 0:
        left_tokens = left_tokens[-window_size:]
        right_tokens = right_tokens[:window_size]
        
    left_truncated = tokenizer.decode(left_tokens, skip_special_tokens=True)
    right_truncated = tokenizer.decode(right_tokens, skip_special_tokens=True)
    
    return f"{left_truncated} {gap_placeholder} {right_truncated}".strip()

def main():
    print("Inizializzazione ablation study...")
    model_name = "bowphs/GreBerta"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512
    
    print("Caricamento dataset di valutazione...")
    eval_dataset = load_dataset("CNR-ILC/gs-maat-eval")
    test_cases = build_dev_cases(eval_dataset, max_cases=50) # Modifica per più recs
    print(f"Istanze di test caricate: {len(test_cases)}")
    
    if HAS_BERTSCORE:
        print("Inizializzazione BERTScorer (GreBerta / multilingua)...")
        # Usiamo il modello greco/multilingua stesso come baseline per bertscore
        scorer = BERTScorer(model_type=model_name, num_layers=4, device=device)
    else:
        print("bert_score non installato, la valutazione BERTScore verrà skippata.")
        scorer = None

    windows_to_test = [256, 128, 64, 32, 16, 8]
    ablation_results = {}
    
    for w in windows_to_test:
        print(f"\n--- Valutazione con Window Size: {w} (L/R) ---")
        
        predictions_ids = []
        true_ids_batch = []
        
        preds_text = []
        refs_text = []
        
        for idx, case in enumerate(test_cases):
            # 1. Troncamento del testo
            truncated_x = truncate_context(case.x, tokenizer, w)
            
            # 2. Generazione predizioni in HCB
            try:
                suggestions = fill_mask(
                    text=truncated_x,
                    n_chars=case.gap_length,
                    model=model,
                    tokenizer=tokenizer,
                    K=10,
                    beam_size=10,
                    method="modified_best_to_worst",
                    return_raw=True, 
                    normalize_probs=True
                )
            except Exception as e:
                print(f"Errore su {case.file_id}: {e}")
                continue

            if not suggestions:
                continue

            # Per Top-K serve il formato raw id [prob, ID_1, ID_2]
            y_tokens_ids = tokenizer(case.y, add_special_tokens=False)["input_ids"]
            
            beam_preds = []
            for s in suggestions:
                # s è (token_ids, prob_normalizzata) perche' normalize_probs=True
                beam_preds.append([s[1]] + s[0])
                
            predictions_ids.append(beam_preds)
            true_ids_batch.append(y_tokens_ids)
            
            # Per il BERTScore usiamo solo il testo ricostruito del suggerimento più probabile (Top-1)
            # s[0] alla posizione 0 è la raw list del candidate top
            top_prediction_str = tokenizer.decode(suggestions[0][0], skip_special_tokens=True).replace(" ", "")
            
            preds_text.append(top_prediction_str)
            refs_text.append(case.y)

        if len(predictions_ids) == 0:
            print("Nessuna predizione utile per questa window.")
            continue
            
        # Calcolo TOP-K
        top_k_metrics = evaluate_topK(
            predictions_hcb_format=predictions_ids,
            true_ids=true_ids_batch,
            tokenizer=tokenizer
        )
        print(f"Top-K Metrics: {top_k_metrics}")
        
        # Calcolo BERTScore
        b_score_metrics = {}
        if scorer is not None and len(preds_text) > 0:
            P, R, F1 = scorer.score(preds_text, refs_text)
            b_score_metrics = {
                "precision": P.mean().item(),
                "recall": R.mean().item(),
                "f1": F1.mean().item()
            }
            print(f"BERTScore Metrics: {b_score_metrics}")

        # Salva a dizionario principale
        ablation_results[w] = {
            "topK": top_k_metrics,
            "bertscore": b_score_metrics
        }
    
    # Salva JSON con i risultati
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=4)
        
    print(f"\nAblation completata e salvata su '{out_path}'.")

if __name__ == "__main__":
    main()
