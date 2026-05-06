
from models.bert.evaluation.metrics import evaluate_topK_text
from models.bert.dataset.dev_set import DevCase

def verify_pipeline():
    # 1. Definiamo un caso di test realistico (con sigma finale nella gold)
    # y è una lista di token (come prodotta da build_dev_cases)
    case = DevCase(
        x="ο αγαθος [....]", 
        y=["αγαθος"], # Gold con sigma finale 'ς'
        gap_length=6,
        corpus_id="test",
        file_id="test"
    )
    
    # 2. Simuliamo i suggerimenti restituiti da fill_mask (ora foldati)
    # Ogni suggerimento è (text, score)
    # Qui simuliamo la predizione che prima falliva: 'αγαθοσ' (sigma mediano)
    predictions_text = [
        [("αγαθοσ", 0.9), ("βελτιστοσ", 0.05), ("κακοσ", 0.01)]
    ]
    
    # 3. La lista di gold labels (batch format)
    gold_labels = [case.y]
    
    print("--- Verifica Pipeline Metriche ---")
    print(f"Gold originale (case.y): {case.y}")
    print(f"Suggerimento top-1 simulato (fill_mask foldato): {predictions_text[0][0][0]}")
    
    # 4. Eseguiamo la valutazione
    metrics = evaluate_topK_text(predictions_text, gold_labels)
    
    print(f"Metriche risultanti: {metrics}")
    
    if metrics['top1'] == 100.0:
        print("\nSUCCESS: Il match avviene correttamente nonostante la differenza tra ς e σ!")
    else:
        print("\nFAILURE: Il match non è avvenuto.")

if __name__ == "__main__":
    verify_pipeline()
