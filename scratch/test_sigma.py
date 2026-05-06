
import unicodedata
from backend.core.preprocess import normalize_greek

def test_sigma_issue():
    gold = "αγαθος λεγει" 
    pred_upper = "ΑΓΑΘΟΣΛΕΓΕΙ" # concatenated as in fill_mask
    
    # Simula metrics.py gold_norm
    gold_norm = normalize_greek(
        text=gold,
        case_folding="lower",
        strip_diacritics_flag=True,
    ).replace(" ", "").strip()
    
    # Simula metrics.py sugg_norm
    sugg_norm = normalize_greek(
        text=pred_upper,
        case_folding="lower",
        strip_diacritics_flag=True,
    ).replace(" ", "").strip()
    
    print(f"Gold: {gold} -> {gold_norm}")
    print(f"Pred: {pred_upper} -> {sugg_norm}")
    print(f"Match: {gold_norm == sugg_norm}")
    
    print(f"Gold codes: {[hex(ord(c)) for c in gold_norm]}")
    print(f"Pred codes: {[hex(ord(c)) for c in sugg_norm]}")

if __name__ == "__main__":
    test_sigma_issue()
