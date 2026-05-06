import sys
import os
sys.path.append(os.getcwd())
import numpy as np
from models.bert.evaluation.metrics import evaluate_bertscore_topk_text

predictions_text = [
    [("κακό", 0.9), ("καλή", 0.1)], # Sample 1: Match at rank 1 (index 1)
    [("νύχτα", 0.8), ("απόγευμα", 0.2), ("μέρα", 0.1)] # Sample 2: Match at rank 2 (index 2)
]
gold_labels = ["καλή", "μέρα"]

print("Testing evaluate_bertscore_topk_text (Complex Case)...")
metrics = evaluate_bertscore_topk_text(predictions_text, gold_labels, k_values=[1, 3])

for k, v in sorted(metrics.items()):
    print(f"{k}: {v:.2f}")

# BS@1 should be low (no perfect matches at rank 0)
# BS@3 should be high (perfect matches at ranks 1 and 2)
assert metrics["bertscore_f1_top3"] > metrics["bertscore_f1_top1"]
print("Success!")
