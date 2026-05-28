import sys
import os
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from models.bert.inference.predict import fill_mask, get_contextual_embeddings
from models.bert.evaluation.plot import plot_umap_suggestions_cluster
from models.bert.dataset.dev_set import DevCase

def test_umap_cohesion_comparison():
    # 1. Configurazione dei modelli da confrontare
    # Usiamo il modello base (Pre-FT) e il modello fine-tuned (Post-FT)
    pre_ft_checkpoint = "bowphs/GreBerta"
    post_ft_checkpoint = "CNR-ILC/gs-GreBerta"
    
    print(f"Modello Pre-FT:  {pre_ft_checkpoint}")
    print(f"Modello Post-FT: {post_ft_checkpoint}\n")
    
    # 2. Definiamo un caso di test campionato casualmente da Hugging Face
    import random
    from datasets import load_dataset
    from models.bert.dataset import EVAL_CHECKPOINT
    
    print(f"Caricamento del dataset di valutazione '{EVAL_CHECKPOINT}'...")
    eval_dataset = load_dataset(EVAL_CHECKPOINT)
    
    # Filtriamo i casi con gap_length appropriati (1-6) per il test set
    # Selezioniamo solo i casi in cui la lacuna rappresenta una parola INTERA (delimitata da spazi o punteggiatura)
    # Questo evita il problema della fusione dei subword token (es. διδαcκα[.]ί̣αϲ) che causerebbe embedding azzerati.
    import re
    word_gap_pattern = re.compile(r"(?:^|\s)\[\.+\](?:$|\s|[.,;:·\s])")
    
    test_rows = eval_dataset["test"].to_list()
    valid_cases = [
        DevCase(
            x=row["x"],
            y=row["y"],
            gap_length=row["gap_length"],
            corpus_id=row["corpus_id"],
            file_id=row["file_id"],
        )
        for row in test_rows
        if 1 <= row["gap_length"] <= 6 and word_gap_pattern.search(row["x"])
    ]
    
    if not valid_cases:
        raise ValueError("Nessun caso valido trovato nel test set con lacuna a parola intera e 1 <= gap_length <= 6")
        
    case = random.choice(valid_cases)
    
    gold_text = " ".join(case.y) if isinstance(case.y, list) else case.y
    
    # Estraiamo una porzione ridotta del contesto per la visualizzazione pulita
    pattern = r"\[\.+\]"
    match = re.search(pattern, case.x)
        
    if match:
        start_idx = max(0, match.start() - 100)
        end_idx = min(len(case.x), match.end() + 100)
        snippet = "..." + case.x[start_idx:end_idx].strip() + "..."
    else:
        snippet = case.x[:200] + "..."
        
    print(f"Caso selezionato (File ID: {case.file_id}, Corpus: {case.corpus_id}):")
    print(f"Contesto (snippet): {snippet}")
    print(f"Gold Label attesa: '{gold_text}' (lunghezza lacuna stimata: {case.gap_length} caratteri)\n")
    
    results = {}
    
    for label, checkpoint in [("Pre-FT", pre_ft_checkpoint), ("Post-FT", post_ft_checkpoint)]:
        print(f"--- Elaborazione modello {label} ({checkpoint}) ---")
        
        # Carichiamo modello e tokenizer
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForMaskedLM.from_pretrained(checkpoint)
        
        # Generiamo i top-K candidati per la lacuna
        K = 20
        suggestions = fill_mask(
            text=case.x,
            checkpoint=post_ft_checkpoint,  # Usiamo la configurazione Post-FT anche per il Pre-FT per uniformità di preprocessing
            model=model,
            tokenizer=tokenizer,
            n_chars=case.gap_length,
            K=K,
        )
        
        candidate_labels = [sug[0] for sug in suggestions]
        print(f"Candidati generati da {label}: {candidate_labels}")
        
        if not candidate_labels:
            print(f"Nessun suggerimento generato per {label}. Salto.\n")
            continue
            
        # Estraiamo gli embedding contestuali dei candidati
        candidate_embeddings = get_contextual_embeddings(
            text_with_gap=case.x,
            candidates=candidate_labels,
            model=model,
            tokenizer=tokenizer,
        )
        
        # Estraiamo l'embedding contestuale della gold label
        gold_embeddings_list = get_contextual_embeddings(
            text_with_gap=case.x,
            candidates=[gold_text],
            model=model,
            tokenizer=tokenizer,
        )
        
        if not gold_embeddings_list:
            print("Impossibile estrarre l'embedding della gold label.")
            continue
        gold_embedding = gold_embeddings_list[0]
        
        # Generiamo il grafico UMAP e otteniamo la coesione del cluster
        output_path = f"models/bert/evaluation/umap_cluster_{label.lower()}.png"
        
        try:
            cohesion = plot_umap_suggestions_cluster(
                candidate_embeddings=candidate_embeddings,
                gold_embedding=gold_embedding,
                candidate_labels=candidate_labels,
                gold_label=gold_text,
                output_path=output_path
            )
            print(f"Grafico UMAP salvato in: {output_path}")
            print(f"Coesione (similarità coseno media intra-cluster): {cohesion:.4f}")
        except Exception as e:
            print(f"Errore durante il plotting UMAP/calcolo coesione: {e}")
            cohesion = None
            
        # Calcoliamo anche la Plausibilità quantitativa
        # Misuriamo la similarità coseno media dei candidati rispetto alla Gold Label
        if candidate_embeddings:
            pooled_cands = torch.stack([
                torch.mean(emb, dim=0) if emb.dim() != 1 else emb 
                for emb in candidate_embeddings
            ])
            pooled_gold = gold_embedding.mean(dim=0).unsqueeze(0) if gold_embedding.dim() != 1 else gold_embedding.unsqueeze(0)
            
            similarities = F.cosine_similarity(pooled_cands, pooled_gold, dim=1)
            mean_gold_sim = similarities.mean().item()
            max_gold_sim = similarities.max().item()
            print(f"Plausibilità (sim. coseno media candidati vs Gold): {mean_gold_sim:.4f}")
            print(f"Plausibilità (sim. coseno max candidato vs Gold): {max_gold_sim:.4f}")
        else:
            mean_gold_sim, max_gold_sim = 0.0, 0.0
            
        results[label] = {
            "cohesion": cohesion,
            "mean_gold_similarity": mean_gold_sim,
            "max_gold_similarity": max_gold_sim,
            "suggestions": candidate_labels
        }
        print()
        
    # 3. Analisi comparativa finale
    print("=== ANALISI COMPARATIVA FINALE ===")
    if "Pre-FT" in results and "Post-FT" in results:
        pre = results["Pre-FT"]
        post = results["Post-FT"]
        
        print(f"Coesione Intra-Cluster: Pre-FT = {pre['cohesion']:.4f} | Post-FT = {post['cohesion']:.4f}")
        if pre["cohesion"] is not None and post["cohesion"] is not None:
            diff_cohesion = post["cohesion"] - pre["cohesion"]
            cohesion_status = "AUMENTATA (Migliore)" if diff_cohesion > 0 else "DIMINUITA"
            print(f"  -> La coesione è {cohesion_status} di {abs(diff_cohesion):.4f}")
            
        print(f"Plausibilità Media vs Gold: Pre-FT = {pre['mean_gold_similarity']:.4f} | Post-FT = {post['mean_gold_similarity']:.4f}")
        diff_gold = post["mean_gold_similarity"] - pre["mean_gold_similarity"]
        plausibility_status = "AUMENTATA (Migliore)" if diff_gold > 0 else "DIMINUITA"
        print(f"  -> La plausibilità media è {plausibility_status} di {abs(diff_gold):.4f}")
        
        print("\nInterpretazione delle Metriche:")
        print("1. Coesione più alta post-FT indica che il modello fine-tuned restringe lo spazio delle ipotesi")
        print("   generando suggerimenti più vicini semanticamente (più coerenti nel contesto).")
        print("2. Plausibilità media più alta post-FT indica che i suggerimenti generati dal modello fine-tuned")
        print("   sono semanticamente e contestualmente più affini alla parola reale attesa (Gold Label).")
        print("3. UMAP Plotting: Visivamente, nel grafico post-FT, la stella rossa (Gold) dovrebbe trovarsi")
        print("   più integrata all'interno del cluster blu (Suggerimenti) rispetto al grafico pre-FT.")
    else:
        print("Impossibile effettuare il confronto (dati mancanti per una delle due fasi).")

if __name__ == "__main__":
    test_umap_cohesion_comparison()
