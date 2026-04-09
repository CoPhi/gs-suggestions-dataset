import os
import math
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Import custom suggestions metric wrappers
from models.bert.dataset.dev_set import build_dev_set
from models.bert.inference.suggestions import generate_hcb_suggestions
from models.bert.evaluation.topk import evaluate_topK, evaluate_bertscore_custom

def main():
    # 1. Caricamento del dataset e del modello
    print("Caricamento dataset e modello...")
    
    # model_name = "cabrooks/LOGION-50k_wordpiece"
    model_name = "bowphs/PhilBerta"  # Oppure "bowphs/GreBerta" o il modello finetunato
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.model_max_length = 512
    
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    
    print(f"Modello {model_name} caricato. Parametri: {model.num_parameters() / 1_000_000:.2f} M")

    # 2. Estrazione e Costruzione DevCase PURI
    # A differenza del dummy test usato dal Trainer di HuggingFace, qui usiamo i gap REALI.
    raw_dataset = load_dataset("CNR-ILC/gs-maat-corpus")
    
    # Se il test split non esiste separatamente ma è una lista, proviamo a unire e filtrare.
    # Spesso "test" è una split
    if "test" in raw_dataset:
        herc_abs = raw_dataset["test"].to_list()
    elif "train" in raw_dataset:
        # Fallback o test custom prelevando uno split dal train
        split = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
        herc_abs = split["test"].to_list()
    else:
        herc_abs = []
        for split_key in raw_dataset:
            herc_abs.extend(raw_dataset[split_key].to_list())

    print(f"Estrapolazione DevCase per l'evaluation HCB...")
    # Assumiamo di testare su max gap length 10 per limiti cognitivi rapidi
    test_cases = build_dev_set(herc_abs, normalize=False, min_gap_length=1, max_gap_length=10)
    print(f"Totale DevCase trovati da riempire: {len(test_cases)}")

    if len(test_cases) == 0:
        print("Attenzione: Non sono stati trovati supplementi nel testo di test.")
        return

    # 3. Processamento beam-search HCB e metriche
    predictions_ids = []
    true_ids_batch = []
    
    # Per il calcolo custom del BERTscore, prepariamo gli arrays
    masked_inputs = []
    masked_positions = []

    print("Inizio iterazione HCB (Left to Right / Best to Worst)...")
    for idx, case in enumerate(test_cases):
        if idx % 10 == 0 and idx > 0:
            print(f"  processati {idx}/{len(test_cases)} casi...")
            
        try:
            # Uso method = "modified_best_to_worst" o "modified_left_to_right"
            suggestions = generate_hcb_suggestions(
                case=case,
                tokenizer=tokenizer,
                model=model,
                num_suggestions=10,
                mask_token=tokenizer.mask_token or "[MASK]",
                method="modified_best_to_worst",
                device=device
            )
            
            # Formato output: lista di K tuple (prob, token_ids, text_suggestion)
            # Raccogliamo la true label idenificata con il tokenizzatore
            # Nota: le etichette per BERT sono tipicamente subwords senza spazi
            y_tokens_ids = tokenizer(case.y, add_special_tokens=False)["input_ids"]
            
            # Conserviamo i risultati necessari per score_batch
            # [ [prob, id1, id2...], [prob, ...] ] per le varie generazioni del beam
            beam_preds = []
            for s in suggestions:
                beam_preds.append([s[0]] + s[1]) # logprob followed by sequence of ids
                
            predictions_ids.append(beam_preds)
            true_ids_batch.append(y_tokens_ids)
            
            # Se volessimo implementare evaluate_bertscore_custom, 
            # dovremmo tenere conto di seq intera e positions,
            # ma siccome `case.x` è già risolto all'interno di generate_hcb_suggestions
            # per non sporcare la memoria per ora si omette (ma si piò espandere se richiesto)
            
        except Exception as e:
            # Per evitare freeze se un case si rompe o logica fuori scala
            print(f"Errore al caso {idx} con lacuna {case.x}: {e}")

    # 4. Calcolo metriche
    if len(predictions_ids) > 0:
        print("Calcolo metriche TopK...")
        top_k_metrics = evaluate_topK(
            predictions_hcb_format=predictions_ids,
            true_ids=true_ids_batch,
            tokenizer=tokenizer
        )

        results = {
            "model_name": model_name,
            "metrics_top_k": top_k_metrics,
            "evaluated_cases": len(predictions_ids)
        }

        print("\n--- RISULTATI HCB ---")
        for k, v in top_k_metrics.items():
            print(f"{k.upper()}: {v:.2f}%")

        # Salva in json
        out_filename = f"eval_results_HCB_{model_name.replace('/', '_')}.json"
        with open(out_filename, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Salvati risultati in {out_filename}")
    else:
        print("Nessuna predizione valida per il calcolo delle metriche.")

if __name__ == "__main__":
    main()
