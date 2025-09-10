from g2p.util import MalformedDictionaryError, UNK_IDX
import torch

class TSVDataset(torch.utils.data.Dataset):
	def __init__(self, dict_path: str, graphemes: list[str], phonemes: list[str]) -> None:
		"""Creates dataset from TSV dictionary file. Dictionary should have two entries.

		Args:
			dict_path (str): Path to dictionary file.
			graphemes (list[str]): List of all valid graphemes.
			phonemes (list[str]): List of all valid phonemes.
		"""
		self.graphemes = graphemes
		self.phonemes = phonemes

		self.grapheme_indices = {token: idx for idx, token in enumerate(self.graphemes)}
		self.phoneme_indices = {token: idx for idx, token in enumerate(self.phonemes)}

		try:
			self.entries = self.load_dict(dict_path)
		except Exception as e:
			print(f"Error loading dictionary: {e}")

	def __len__(self) -> int:
		"""Returns length of dataset.

		Returns:
			int: Length of dataset.
		"""
		return len(self.entries)
	
	def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
		"""Returns tensor pair at index.

		Args:
			idx (int): Index to query.

		Returns:
			tuple[Tensor, Tensor]: Tensor pair of graphemes and phonemes
		"""
		word, phonemes = self.entries[idx]

		grapheme_tensor = torch.tensor([self.grapheme_indices.get(grapheme, UNK_IDX) for grapheme in word], dtype=torch.long)
		phoneme_tensor = torch.tensor([self.phoneme_indices.get(phoneme, UNK_IDX) for phoneme in phonemes], dtype=torch.long)

		return grapheme_tensor, phoneme_tensor

	def load_dict(self, dict_path: str) -> list[tuple[str, list[str]]]:
		"""Load dictionary entries from TSV path.

		Args:
			dict_path (str): Path to dictionary file.

		Raises:
			MalformedDictionaryError: Raised if there is an error reading file or entries.

		Returns:
			list[tuple[str, list[str]]]: List of word and phoneme pairs.
		"""
		entries = []

		with open(dict_path, 'r', encoding='utf-8') as dictionary:
			idx = 1
			for line in dictionary:
				line = line.strip()
				entry = line.split('\t')

				if len(entry) != 2:
					raise MalformedDictionaryError(f"Got {len(entry)} values at line {idx}, expected 2.")
				
				word = entry[0]
				phonemes = entry[1].split(' ')

				entries.append((word, phonemes))

				idx += 1

		return entries
				
				