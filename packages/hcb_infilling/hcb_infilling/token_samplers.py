"""Contains functions for determining the next set of tokens to use for candidate in-fillings."""
import torch

def example_sampler(
    num_inputs,
    candidates,
    candidate_log_likelihoods,
    remaining_masked_positions,
    token_log_probs,
    mask_ids,
    probs_update_fxn,
    beam_size,
    pivots = None,
    initial = False,
):
  """Example function describing common arguments for token samplers.

  num_inputs - Number of inputs being infilled, i.e. batch size
  candidates - The current set of candidate infillings for all inputs,
               stored as a tensor of size N x L were N is the `num_inputs`
               times `beam_size` and L is the padded width of the batch.
               May be `num_inputs` x L if `initial` is True.
  candidate_log_likelihoods - The current log-likelihoods of each candidate
                              as a 1D tensor of size N.
  remaining_masked_positions - The indices of masked positions that still
                               need to be filled in each candidate. Tensor
                               of size N x X, where X is the remaining number
                               of rounds of infilling.
  token_log_probs - The log-probability of each token for a given candidate,
                    tensor of size N x V, where V is the vocab size.
  mask_ids - The indices of the masked positions currently being filled in
             each candidate (all the same in the left-to-right case, possibly
             different in best-to-worst). 1D tensor of size N.
  probs_update_fxn - Function for updating token log-probs given the candidate
                     log-likelihoods. See `probs_update.py` for options.
  beam_size - Number of infilling candidates maintained per input.
  pivots - Tensor of size N x L specifying the pivot to use for each masked
           position. Only used of `probs_update_fxn` is `hcb_update`.
  initial - Bool indicating if this is the first iteration of infilling. This
            is a bit of a workaround to handle the initial fanning out of
            a tensor with `num_inputs` rows to a tensor with `num_inputs * beam_size`
            rows.
  """
  raise("Unimplemented")

def beam_search(
    num_inputs,
    candidates,
    candidate_log_likelihoods,
    remaining_masked_positions,
    token_log_probs,
    mask_ids,
    probs_update_fxn,
    beam_size,
    pivots = None,
    initial = False,
):
  """Candidate extensions via beam search.

  See `example_sampler` above for argument descriptions.
  """
  if initial:
    top_log_probs, top_indices = token_log_probs.topk(beam_size, dim=-1)

    mask_ids_repeated = torch.repeat_interleave(mask_ids, repeats=beam_size, dim=0)
    candidates = torch.repeat_interleave(candidates, repeats=beam_size, dim=0)
    candidates[torch.arange(len(candidates)), mask_ids_repeated] = top_indices.ravel()

    candidate_log_likelihoods = probs_update_fxn(
          top_log_probs,
          candidate_log_likelihoods,
          token_log_probs,
          pivots,
          mask_ids,
    )
    candidate_log_likelihoods = candidate_log_likelihoods.ravel()
    return candidates, candidate_log_likelihoods, remaining_masked_positions
  else:
    top_log_probs, top_indices = token_log_probs.topk(beam_size, dim=-1)
    top_log_probs = probs_update_fxn(
        top_log_probs,
        candidate_log_likelihoods,
        token_log_probs,
        pivots,
        mask_ids,
    )
    # Update each input's candidates in place.
    for input_idx in range(num_inputs):
      # Indices of this input's candidates in the overall candidate tensor.
      start = input_idx * beam_size
      end = start + beam_size

      # Collect token probabilities and ids across all candidates for this input.
      values_for_input = top_log_probs[start:end]
      tokens_for_input = top_indices[start:end]

      # Get top-k across all candidates.
      top_candidate_likelihoods, top_candidate_idxs = values_for_input.ravel().topk(beam_size)

      # Map back to which original candidate for this input we're extending.
      orig_candidate_idxs = (top_candidate_idxs // beam_size) + start

      # Update in place.
      candidates[start:end] = candidates[orig_candidate_idxs]
      remaining_masked_positions[start:end] = remaining_masked_positions[orig_candidate_idxs]
      mask_token_ids = mask_ids[orig_candidate_idxs]

      candidates[torch.arange(start, end), mask_token_ids] = tokens_for_input.ravel()[top_candidate_idxs]
      candidate_log_likelihoods[start:end] = top_candidate_likelihoods

    return candidates, candidate_log_likelihoods, remaining_masked_positions

def token_sampling(
    num_inputs,
    candidates,
    candidate_log_likelihoods,
    remaining_masked_positions,
    token_log_probs,
    mask_ids,
    probs_update_fxn,
    beam_size,
    pivots = None,
    initial = False,
    nucleus_prob = None,
    rng = None,
):
  """Candidate extensions via sampling methods.

  Most args are described above in `example_sampler`, except:
  nucleus_prob - If not None, perform nucleus sampling with this value as the
                 total proability mass of the nucleus.
  rng - A torch.Generator object for controlling random number generation, for
        reproducibility.
  """
  token_probs = token_log_probs.exp()
  if nucleus_prob is not None:
    sorted_probs, sorted_indices = token_probs.sort(dim=1, descending=True)
    mask = sorted_probs.cumsum(dim=1) > nucleus_prob

    flattened_idxs = (sorted_indices + (torch.arange(token_probs.shape[0]) * token_probs.shape[1])[:, None])
    zero_indices = flattened_idxs[mask.bool()]

    sample_probs = token_probs.clone()
    sample_probs.ravel()[zero_indices] = 0
    # Hacky way to work around the case where one element has prob greater than nucleus_prob
    sample_probs.ravel()[flattened_idxs[:, 0].flatten()] = token_probs.ravel()[flattened_idxs[:, 0].flatten()]
  else:
    sample_probs = token_probs

  if initial:
    sampled_tokens = torch.multinomial(sample_probs, beam_size, generator=rng)
    flattened_idxs = (sampled_tokens + (torch.arange(token_probs.shape[0]) * token_probs.shape[1])[:, None]).ravel()
    sampled_probs = token_probs.ravel()[flattened_idxs].reshape(num_inputs, beam_size)

    mask_ids_repeated = torch.repeat_interleave(mask_ids, repeats=beam_size, dim=0)
    candidates = torch.repeat_interleave(candidates, repeats=beam_size, dim=0)
    candidates[torch.arange(len(candidates)), mask_ids_repeated] = sampled_tokens.ravel()

    candidate_log_likelihoods = probs_update_fxn(
          sampled_probs.log(),
          candidate_log_likelihoods,
          token_probs,
          pivots,
          mask_ids,
    )
    candidate_log_likelihoods = candidate_log_likelihoods.ravel()

    return candidates, candidate_log_likelihoods, remaining_masked_positions
  else:
    sampled_tokens = torch.multinomial(sample_probs, 1, generator=rng)
    flattened_idxs = (sampled_tokens + (torch.arange(token_probs.shape[0]) * token_probs.shape[1])[:, None]).ravel()
    sampled_probs = token_probs.ravel()[flattened_idxs].unsqueeze(1)

    candidates[torch.arange(len(candidates)), mask_ids] = sampled_tokens.ravel()

    candidate_log_likelihoods = probs_update_fxn(
        sampled_probs.log(),
        candidate_log_likelihoods,
        token_probs,
        pivots,
        mask_ids,
    ).squeeze()

    return candidates, candidate_log_likelihoods, remaining_masked_positions