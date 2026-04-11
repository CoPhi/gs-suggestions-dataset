import torch
import nltk

import numpy as np

from bert_score import BERTScorer


def getBLEU(reference, hypothesis):
  return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights=(1,0,0,0))

def score_batch(suggestions_batch, true_ids_batch, tokenizer, method='topk'):
  count = 0 # Track total number of non-padding ids
  num_correct = np.zeros(len(suggestions_batch[0])) # Track top-k accuracy for each k < batchsize
  for suggestions, true_ids in zip(suggestions_batch, true_ids_batch):
    if tokenizer.pad_token_id in true_ids: continue # This is just an instance of padding
    else:
      sorted_suggestions = [s[1:] for s in sorted(suggestions, key=lambda x: x[0], reverse=True)]
      if method=='topk':
        if true_ids in sorted_suggestions:
          num_correct[sorted_suggestions.index(true_ids)] += 1
      elif method=='BLEU':
        num_correct[0] += getBLEU(true_ids, sorted_suggestions[0])
      count += 1

  return count, num_correct

def reconstruct_masked_sentence(masked_ids, replacements, tokenizer):
  replacement_tensor = replacements.flatten()
  reconstructed_tensor = masked_ids.clone()
  indices = torch.nonzero(masked_ids == tokenizer.mask_token_id)
  for i, index in enumerate(indices):
    reconstructed_tensor[tuple(index.tolist())] = replacement_tensor[i]
  return reconstructed_tensor

default_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
def compute_bertscore(
    suggestions,
    true_ids_batch,
    masked_inputs_batch,
    masked_positions,
    tokenizer,
    scorer=default_scorer,
):
    # Currently just looking at top prediction (i.e. top-k = 1)
    preds_for_decode_top_only = [suggestions[i][0][1:]  for i in range(len(suggestions))]
    full_preds = masked_inputs_batch.clone()
    full_preds[:, masked_positions] = torch.tensor(preds_for_decode_top_only)
    preds_text = tokenizer.batch_decode(full_preds, skip_special_tokens=True)

    full_true = masked_inputs_batch.clone()
    full_true[:, masked_positions] = torch.tensor(true_ids_batch)
    true_text = tokenizer.batch_decode(full_true, skip_special_tokens=True)

    # Scorer returns a tuple of 1-D tensors giving precision, recall, and F1
    scores = scorer.score(preds_text, true_text)
    return {
        "precision": scores[0].mean().item(),
        "recall": scores[1].mean().item(),
        "f1": scores[2].mean().item(),
    }

def update_metrics(metrics, key, got):
    if key not in metrics:
        metrics[key] = got
    else:
        for (k, v) in got.items():
            metrics[key][k] += v
