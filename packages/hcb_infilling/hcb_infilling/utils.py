import re

import torch

import numpy as np

def get_masked_positions(inputs, mask_id):
  masked_positions = [(inputs[i] == mask_id).nonzero().squeeze() for i in range(len(inputs))]
  num_masked_per_input = [len(x) for x in masked_positions]
  # Currently require that all inputs have the same number of masked positions.
  # Could probably relax this in the future if need be.
  assert len(np.unique(num_masked_per_input)) == 1

  return torch.cat(masked_positions, dim=-1).reshape(len(inputs), -1)

def get_best_masked_positions(log_probs, remaining_masked_positions):
  remaining_masked_positions = remaining_masked_positions.to(log_probs.device)
  # First take max over vocab dimension, giving us top probs per position
  # in each input.
  max_per_pos, _ = log_probs.max(dim=-1)
  # Subset to just the remaining masked positions and get the position
  # with the highest prob option.
  max_pos_idx = torch.gather(max_per_pos, 1, remaining_masked_positions).argmax(dim=-1)
  # Map back to what the actual masked position is.
  best_mask_positions = torch.gather(remaining_masked_positions, 1, max_pos_idx.unsqueeze(1)).squeeze().detach().cpu()

  # Remove the selected masked positions from the tensor of remaining masked
  # positions.
  to_remove = torch.ones_like(remaining_masked_positions).scatter_(1, max_pos_idx.unsqueeze(1), 0)
  remaining_masked_positions = (remaining_masked_positions
                                [to_remove.bool()]
                                .reshape(remaining_masked_positions.shape[0], -1)
                                .detach()
                                .cpu())

  return best_mask_positions, remaining_masked_positions


def mask_tokens_batch(input_ids, indices, mask_token_id, pad_token_id = 0):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    true_ids = input_ids[:,indices]

    num_pads_masked = (true_ids == pad_token_id).sum().item()
    if num_pads_masked != 0:
      print(f"WARN: {num_pads_masked} / {true_ids.numel()} masked positions are pad tokens")

    masked_input_ids = input_ids.clone()
    masked_input_ids[:, indices] = mask_token_id
    return masked_input_ids, true_ids.tolist()

def display_suggestions_batch(input_ids, suggestions, tokenizer):
    input_ids_copy = torch.clone(input_ids)
    for i in range(input_ids.shape[0]):
      mask_token_ids = (input_ids_copy[i] == tokenizer.mask_token_id).nonzero().squeeze().tolist()
      for suggestion in suggestions[i]:
          for token_idx, token_id in enumerate(suggestion[1:]):
                input_ids_copy[i, mask_token_ids[token_idx]] = token_id
          print(f"{suggestion[0]:.2%} - {tokenizer.decode(input_ids_copy[i, 1:-1])}")

def sent_too_long(sent, max_toks, tokenizer):
  # split up a list until no more than 500 tokens anywhere
  words = sent.split(' ')
  # tokenize each word, concatenate until too many tokens!
  final_ls = []
  cur_str = ''
  cur_toks = 0
  for word in words:
    toks = tokenizer(word)['input_ids']
    if (len(toks) + cur_toks) <= max_toks:
      cur_str += (' ' + word)
      cur_toks += len(toks)
    else:
      final_ls.append(cur_str)
      cur_str = word
      cur_toks = len(toks)
  if cur_toks > 0:
    final_ls.append(cur_str)
    return final_ls

# returns a list of sentences which we can write to a file, separated by newlines
def sep_sents(text, ind, tokenizer, max_toks=500):

  # add special separator after end punctuation
  text = re.sub(r'\·', '· [SEP] ', text)
  text = re.sub(r'\.', '. [SEP] ', text)
  text = re.sub(r'\;', '; [SEP] ', text)
  text = re.sub(r'\!', '! [SEP] ', text)
  text = re.sub(r'\?', '? [SEP] ', text)

  sents = text.split('[SEP]')
  # tokenize each sentence, concatenate until too many tokens!
  final_ls = []
  cur_str = ''
  cur_toks = 0
  for sent in sents:
    toks = tokenizer(sent)['input_ids']
    if (len(toks) + cur_toks) <= max_toks:
      cur_str += (' ' + sent)
      cur_toks += len(toks)
    elif cur_toks == 0:
      final_ls += sent_too_long(sent, max_toks)
      cur_toks = 0
      cur_str = ''
    else:
      final_ls.append(cur_str)
      cur_str = sent
      cur_toks = len(toks)
  if cur_toks > 0:
    final_ls.append(cur_str)
  if ' ' not in cur_str:
    print('HERE: ' + str(ind))
  ret_ls = []
  for el in final_ls:
    if len(tokenizer(el)['input_ids']) > max_toks:
      ret_ls += sep_sents(el, -1 * ind)
    else:
      ret_ls.append(el)
  return ret_ls