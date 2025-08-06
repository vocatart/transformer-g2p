UNK_IDX = 0

class MalformedDictionaryError(Exception):
	"""Exception raised when a dictionary has invalid entries."""
	def __init__(self, message: str="Invalid dictionary data.") -> None:
		super().__init__(message)
		