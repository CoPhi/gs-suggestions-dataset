import asyncio
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from models.bert.inference.predict import fill_mask
from backend.core.preprocess import normalize_greek, strip_diacritics
from models.bert.finetuning import get_model_config

async def debug_model(checkpoint, text):
    print(f"\n--- Testing Model: {checkpoint} ---")
    try:
        model = AutoModelForMaskedLM.from_pretrained(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        config = get_model_config(checkpoint)
        
        # Preprocessing similar to service
        processed_text = normalize_greek(text, case_folding=False)
        case_folding = config.get("case_folding")
        if case_folding == "upper":
            processed_text = processed_text.upper()
        elif case_folding == "lower":
            processed_text = processed_text.lower()
            
        print(f"Processed Text: {processed_text}")
        
        suggestions = fill_mask(
            text=processed_text,
            model=model,
            tokenizer=tokenizer,
            K=5,
            normalize_probs=True
        )
        
        for cand, score in suggestions:
            print(f"  - {cand} ({score:.4f})")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    text = "α̣λλα μην εν τωι κατ̣[.]ς̣κευαζειν"
    models = ["CNR-ILC/gs-GreBerta", "CNR-ILC/gs-aristoBERTo", "CNR-ILC/gs-Logion"]
    for m in models:
        await debug_model(m, text)

if __name__ == "__main__":
    asyncio.run(main())
