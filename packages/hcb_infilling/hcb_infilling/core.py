import torch

from torch import nn

from .utils import get_best_masked_positions, get_masked_positions


@torch.no_grad()
def decode_base(
    input_ids,
    attention_mask,
    beam_size,
    probs_update_fxn,
    probs_to_token_fxn,
    model,
    mask_id,
    pivots = None,
    best_to_worst = False,
):

  device = next(model.parameters()).device
  masked_positions = get_masked_positions(input_ids, mask_id=mask_id)
  remaining_masked_positions = masked_positions.clone()
  num_masked_positions = masked_positions.shape[1]
  num_inputs = len(input_ids)

  log_softmax = nn.LogSoftmax(dim=-1)

  # Get initial pool of candidates by considering first masked position separately.
  initial_logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask).logits
  log_probs = log_softmax(initial_logits)
  if best_to_worst:
    mask_ids, remaining_masked_positions = get_best_masked_positions(
      log_probs,
      remaining_masked_positions,
    )
  else:
    mask_ids = masked_positions[:, 0]
    remaining_masked_positions = remaining_masked_positions[:, 1:]

  log_probs = log_probs[torch.arange(len(log_probs)), mask_ids, :].detach().cpu()
  candidates, candidate_log_likelihoods, remaining_masked_positions = probs_to_token_fxn(
    num_inputs,
    input_ids,
    torch.zeros(len(input_ids)),
    remaining_masked_positions,
    log_probs,
    mask_ids,
    probs_update_fxn,
    beam_size,
    pivots,
    initial=True,
  )

  # Create repeated versions of the various tensors that have num_rows =  num_inputs * beam_size,
  # instead of just num_rows = num_inputs, where each input is repeated beam_size times.
  # Rows (i, i+beam_size) are the current candidates for the i'th input
  if pivots is not None:
    pivots_repeated = torch.repeat_interleave(pivots, repeats=beam_size, dim=0)
  else:
    pivots_repeated = None
  attn_mask_beam_search = torch.repeat_interleave(attention_mask, repeats=beam_size, dim=0)
  masked_positions_repeated = torch.repeat_interleave(masked_positions, repeats=beam_size, dim=0)
  remaining_mask_ids_repeated = torch.repeat_interleave(remaining_masked_positions, repeats=beam_size, dim=0)

  # Do the rest of the beam search given the initial candidates.
  for _  in range(num_masked_positions - 1):
    # Token probabilities for all candidates for all inputs.
    logits = model(input_ids=candidates.to(device), attention_mask=attn_mask_beam_search).logits
    log_probs = log_softmax(logits)
    if best_to_worst:
      mask_ids, remaining_mask_ids_repeated = get_best_masked_positions(
        log_probs,
        remaining_mask_ids_repeated,
      )
    else:
      mask_ids = remaining_mask_ids_repeated[:, 0]
      remaining_mask_ids_repeated = remaining_mask_ids_repeated[:, 1:]

    log_probs = log_probs[torch.arange(len(log_probs)), mask_ids, :].detach().cpu()

    candidates, candidate_log_likelihoods, remaining_mask_ids_repeated = probs_to_token_fxn(
      num_inputs,
      candidates,
      candidate_log_likelihoods,
      remaining_mask_ids_repeated,
      log_probs,
      mask_ids,
      probs_update_fxn,
      beam_size,
      pivots_repeated,
      initial=False,
    )

  probs_by_input = candidate_log_likelihoods.reshape(num_inputs, beam_size)
  completions_by_input = (torch.gather(candidates, 1, masked_positions_repeated)
                          .reshape(num_inputs, beam_size, num_masked_positions))

  # Repackage into same output format as previous functions.
  output = []
  for input_idx in range(num_inputs):
    probs = probs_by_input[input_idx].tolist()
    completions = completions_by_input[input_idx].tolist()
    output.append([[probs[i]] + completions[i] for i in range(beam_size)])
  return output
