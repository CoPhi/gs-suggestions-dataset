
from backend.core.preprocess import normalize_greek

def test_sigma_casefold():
    gold = "αγαθος λεγει" 
    pred_upper = "ΑΓΑΘΟΣΛΕΓΕΙ" 
    
    # Gold norm con casefold
    gold_norm_fold = normalize_greek(gold, case_folding="none").casefold().replace(" ", "")
    
    # Pred norm con casefold
    pred_norm_fold = normalize_greek(pred_upper, case_folding="none").casefold().replace(" ", "")
    
    print(f"Gold Fold: {gold_norm_fold}")
    print(f"Pred Fold: {pred_norm_fold}")
    print(f"Match: {gold_norm_fold == pred_norm_fold}")

if __name__ == "__main__":
    test_sigma_casefold()
