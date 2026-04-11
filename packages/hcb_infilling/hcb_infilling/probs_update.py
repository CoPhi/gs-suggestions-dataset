"""Contains function for updating probabilities of candidate in-filling tokens."""
import torch

def pseudo_ll_update(
    topk_log_probs,
    curr_log_likelihoods,
    log_probs,
    pivots,
    mask_idxs,
):
  """Pseudo log-likelihood update where candidate prob is just p(candidate) * p(token)

  topk_log_probs - The log-probability of the top-k tokens for each candidate,
                   tensor of size N x k, where k is the beam size.
  curr_log_likelihoods - The current log-likelihoods of each candidate
                         as a 1D tensor of size N.
  """
  return topk_log_probs + curr_log_likelihoods[:, None]

def hcb_update(
    topk_log_probs,
    curr_log_likelihoods,
    log_probs,
    pivots,
    mask_idxs,
):
  """HCB log-likelihood update.

  topk_log_probs - The log-probability of the top-k tokens for each candidate,
                   tensor of size N x k, where k is the beam size.
  curr_log_likelihoods - The current log-likelihoods of each candidate
                         as a 1D tensor of size N.
  log_probs - The log-probability of each token for a given candidate,
              tensor of size N x V, where V is the vocab size.
  pivots - Tensor of size N x L specifying the pivot to use for each masked
           position. Only used of `probs_update_fxn` is `hcb_update`.
  mask_idxs - The indices of the masked positions currently being filled in
             each candidate (all the same in the left-to-right case, possibly
             different in best-to-worst). 1D tensor of size N.
  """
  N = pivots.shape[0]
  pivot_idxs = pivots[torch.arange(N), mask_idxs]
  pivot_vals = log_probs[torch.arange(N), pivot_idxs]
  topk_log_probs -= pivot_vals[:, None]
  return topk_log_probs + curr_log_likelihoods[:, None]
