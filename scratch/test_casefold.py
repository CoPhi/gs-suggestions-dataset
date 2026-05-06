
from backend.core.preprocess import normalize_greek

def test_casefold():
    text1 = "ΑΓΑΘΟΣ"
    text2 = "αγαθος"
    
    print(f"Original 1: {text1}")
    print(f"Original 2: {text2}")
    
    low1 = normalize_greek(text1, case_folding="lower")
    low2 = normalize_greek(text2, case_folding="lower")
    
    print(f"Lower 1: {low1} ({[hex(ord(c)) for c in low1]})")
    print(f"Lower 2: {low2} ({[hex(ord(c)) for c in low2]})")
    print(f"Match Lower: {low1 == low2}")
    
    fold1 = text1.casefold()
    fold2 = text2.casefold()
    
    print(f"Fold 1: {fold1} ({[hex(ord(c)) for c in fold1]})")
    print(f"Fold 2: {fold2} ({[hex(ord(c)) for c in fold2]})")
    print(f"Match Fold: {fold1 == fold2}")

if __name__ == "__main__":
    test_casefold()
