import torch

UNK_IDX = 0
PAD_IDX = 1
BOS_IDX = 3
EOS_IDX = 4

class MalformedDictionaryError(Exception):
	"""Exception raised when a dictionary has invalid entries."""
	def __init__(self, message: str="Invalid dictionary data.") -> None:
		super().__init__(message)

def make_len_mask(input: torch.Tensor) -> torch.Tensor:
	"""Creates boolean tensor where true indicates padded positions.

	Args:
		input (torch.Tensor): Input tensor.

	Returns:
		torch.Tensor: Output boolean tensor.
	"""
	return (input == PAD_IDX).transpose(0, 1)

# https://github.com/spring-media/DeepPhonemizer/blob/5dce7e27556aef4426f5623baf6351d266a30a73/dp/model/utils.py#L38
def get_dedup_tokens(logits_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
	"""Converts a batch of logits into the batch most probable tokens and their probabilities.

    Args:
      logits_batch (torch.Tensor): Batch of logits (N x T x V).

    Returns:
      Tuple: Deduplicated tokens. The first element is a tensor (token indices) and the second element
      is a tensor (token probabilities)
    """

	logits_batch = logits_batch.softmax(-1)
	out_tokens, out_probs = [], []

	for i in range(logits_batch.size(0)):
		logits = logits_batch[i]

		max_logits, max_indices = torch.max(logits, dim=-1)
		max_logits = max_logits[max_indices != UNK_IDX]
		max_indices = max_indices[max_indices != UNK_IDX]

		cons_tokens, counts = torch.unique_consecutive(max_indices, return_counts=True)
		out_probs_i = torch.zeros(len(counts), device=logits.device)

		ind = 0
		for i, c in enumerate(counts):
			max_logit = max_logits[ind:ind + c].max()
			out_probs_i[i] = max_logit
			ind = ind + c

		out_tokens.append(cons_tokens)
		out_probs.append(out_probs_i)

	out_tokens = torch.nn.utils.rnn.pad_sequence(out_tokens, batch_first=True, padding_value=PAD_IDX).long()
	out_probs = torch.nn.utils.rnn.pad_sequence(out_probs, batch_first=True, padding_value=PAD_IDX)

	return out_tokens, out_probs