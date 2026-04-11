from collections import defaultdict

import torch
import nltk
import numpy as np

from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    RobertaForMaskedLM,
    RobertaTokenizer,
    DistilBertTokenizer,
    DistilBertForMaskedLM,
)

from packages.hcb_infilling.hcb_infilling.decode import (
   decode_modified_BestToWorst_vectorized,
   decode_modified_LeftToRight_vectorized,
   decode_standard_BestToWorst_vectorized,
   decode_standard_LeftToRight_vectorized
)
from packages.hcb_infilling.hcb_infilling.metrics import (
   compute_bertscore,
   score_batch,
   update_metrics,
)
from packages.hcb_infilling.hcb_infilling.utils import(
  mask_tokens_batch,
  sep_sents,
)

nltk.download('brown')
print("Downloaded corpus.")

from nltk.corpus import brown

model_name = 'bert-base-uncased'
method = 'topk'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#tokenizer = BertTokenizer.from_pretrained(model_name)
#model = BertForMaskedLM.from_pretrained(model_name).to(device)
if "roberta" in model_name:
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name).to(device)
elif "distilbert" in model_name:
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForMaskedLM.from_pretrained(model_name).to(device)
else:
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name).to(device)

np.random.seed(42)

beam_size = 5
total_examples = len(brown.words())  # Use entire corpus.
context_length = 512
report_period = 10
num_masks = 2
num_examples = 16
num_experiments = 32

# Construct data

data = sep_sents(' '.join(brown.words()), 0, tokenizer, max_toks = context_length)
print("Separated data.")

data = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
print("Tokenized data.")


num_total = 0
num_modified_LeftToRight_correct = np.zeros(beam_size) # Track top-k accuracy
num_modified_BestToWorst_correct = np.zeros(beam_size) # Track top-k accuracy
num_standard_LeftToRight_correct = np.zeros(beam_size) # Track top-k accuracy
num_standard_BestToWorst_correct = np.zeros(beam_size) # Track top-k accuracy
bertscore_metrics = defaultdict(dict)

rng = np.random.default_rng(seed=42)

for batch_num in range(num_experiments):
  if batch_num % report_period == 0:
    print("Starting batch", batch_num+1)
    print()
  example_nums = rng.choice(np.arange(len(data.input_ids)), size=num_examples, replace=False)

  input_ids = data.input_ids[example_nums]
  attention_mask = data.attention_mask[example_nums].to(device)
  total = input_ids.shape[1]

  if batch_num % report_period == 0:
    print("Example text:")
    print(' '.join(tokenizer.convert_ids_to_tokens(input_ids[0][:10])))

  mask_start_ind = rng.choice(np.arange(1, total-num_masks))
  if batch_num % report_period == 0:
    print(f"Index: {mask_start_ind} / {total-num_masks}")
  masked_positions = list(range(mask_start_ind,mask_start_ind+num_masks))
  masked_inputs_batch, true_ids_batch = mask_tokens_batch(input_ids, masked_positions, tokenizer.mask_token_id, tokenizer.pad_token_id)

  suggestions_batch = decode_modified_LeftToRight_vectorized(model, masked_inputs_batch, attention_mask, beam_size, tokenizer.mask_token_id)
  count_batch, num_correct_batch = score_batch(suggestions_batch, true_ids_batch, tokenizer, method=method)
  num_modified_LeftToRight_correct += num_correct_batch
  update_metrics(
    bertscore_metrics,
    "modified_left_to_right",
    compute_bertscore(
      suggestions_batch,
      true_ids_batch,
      masked_inputs_batch,
      masked_positions,
      tokenizer,
    )
  )

  suggestions_batch = decode_modified_BestToWorst_vectorized(model, masked_inputs_batch, attention_mask, beam_size, tokenizer.mask_token_id)
  count_batch, num_correct_batch = score_batch(suggestions_batch, true_ids_batch, tokenizer, method=method)
  num_modified_BestToWorst_correct += num_correct_batch
  update_metrics(
    bertscore_metrics,
    "modified_best_to_worst",
    compute_bertscore(
      suggestions_batch,
      true_ids_batch,
      masked_inputs_batch,
      masked_positions,
      tokenizer,
    )
  )

  suggestions_batch = decode_standard_LeftToRight_vectorized(model=model, input_ids=masked_inputs_batch, attention_mask= attention_mask, beam_size=beam_size, mask_id=tokenizer.mask_token_id)
  count_batch, num_correct_batch = score_batch(suggestions_batch, true_ids_batch, tokenizer, method=method)
  num_standard_LeftToRight_correct += num_correct_batch
  update_metrics(
    bertscore_metrics,
    "standard_left_to_right",
    compute_bertscore(
      suggestions_batch,
      true_ids_batch,
      masked_inputs_batch,
      masked_positions,
      tokenizer,
    )
  )

  suggestions_batch = decode_standard_BestToWorst_vectorized(model=model, input_ids=masked_inputs_batch, attention_mask= attention_mask, beam_size=beam_size, mask_id=tokenizer.mask_token_id)
  count_batch, num_correct_batch = score_batch(suggestions_batch, true_ids_batch, tokenizer, method=method)
  num_standard_BestToWorst_correct += num_correct_batch
  update_metrics(
    bertscore_metrics,
    "standard_best_to_worst",
    compute_bertscore(
      suggestions_batch,
      true_ids_batch,
      masked_inputs_batch,
      masked_positions,
      tokenizer,
    )
  )

  # Update total count only once:
  num_total += count_batch

  if batch_num % report_period == 0:
    print()
    print(f"Modified Left-to-Right Correct: {num_modified_LeftToRight_correct}/{num_total}")
    print(f"Modified Best-to-Worst Correct: {num_modified_BestToWorst_correct}/{num_total}")
    print(f"Standard Left-To-Right Correct: {num_standard_LeftToRight_correct}/{num_total}")
    print(f"Standard Best-to-Worst Correct: {num_standard_BestToWorst_correct}/{num_total}")
    print()

    # Printing top-k accuracies:
    for k in range(beam_size):
      print(f"Modified Left-to-Right Top-{k+1}: {sum(num_modified_LeftToRight_correct[:k+1])/num_total:.2%}")
      print(f"Modified Best-to-Worst Top-{k+1}: {sum(num_modified_BestToWorst_correct[:k+1])/num_total:.2%}")
      print(f"Standard Left-To-Right Top-{k+1}: {sum(num_standard_LeftToRight_correct[:k+1])/num_total:.2%}")
      print(f"Standard Best-to-Worst Top-{k+1}: {sum(num_standard_BestToWorst_correct[:k+1])/num_total:.2%}")
      print()
    print(f"Total: {num_total}")

for exp_name, results in bertscore_metrics.items():
  for metric, value in bertscore_metrics[exp_name].items():
      bertscore_metrics[exp_name][metric] = value / num_experiments
print(bertscore_metrics)
# -


