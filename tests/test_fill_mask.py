import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from models.bert.inference.predict import fill_mask

def test_inference_pipeline():
    print("Caricamento del modello (bowphs/GreBerta)...")
    model_name = "bowphs/GreBerta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    # Esempio di una frase greca in cui mancano 5 caratteri
    # Potrebbe essere per esempio "[.....]"
    test_text = "Ἀριστοτέλης ἐστὶν ὁ [.....]ξαν ἀληθῆ φαίνε"
    n_chars_lacuna = 5
    
    print(f"\nTesto di Input: {test_text}")
    print(f"Grandezza suggerimento (n_chars): {n_chars_lacuna}")
    print("\nGenerazione HCB in corso... (potrebbe richiedere qualche istante)\n")
    
    suggestions = fill_mask(
        text=test_text,
        n_chars=n_chars_lacuna,
        model=model,
        tokenizer=tokenizer,
        K=5,                # Restituisce la TOP 5
        beam_size=10,       # Ampiezza del check 
        method="modified_best_to_worst",
        case_folding=False
    )
    
    print("--- TOP 5 SUGGERIMENTI ---")
    for i, (text, score) in enumerate(suggestions, 1):
        print(f"{i}. Text: '{text}' | Score: {score:.4f}")

if __name__ == "__main__":
    test_inference_pipeline()
