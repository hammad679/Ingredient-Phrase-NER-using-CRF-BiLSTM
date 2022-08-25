import pickle
import pathlib
import numpy as np

class Utils:
	def __init__(self, data_path):
		data_path = pathlib.Path(data_path)
		if not data_path.exists():
			raise FileNotFoundError()

		self.PAD_OP = "<PAD>"
		self.UNK_OP = "<UNK>"

		data_files = [file for file in data_path.iterdir() if file.is_file()]
		self.loaded_data = {}

		for file in data_files:
			with open(file, "rb") as f:
				self.loaded_data[file.stem] = pickle.load(f)

	def pad_phrase(self, phrase, max_len, by_index=True):
		pad_op_to_use = self.loaded_data["token2idx"][self.PAD_OP] if by_index else self.PAD_OP
		return phrase + [pad_op_to_use] * (max_len - len(phrase))

	def pad_phrase_labels(self, phrase_labels, max_len, by_index=True):
		pad_op_to_use = self.loaded_data["label2idx"][self.PAD_OP] if by_index else self.PAD_OP
		return phrase_labels + [pad_op_to_use] * (max_len - len(phrase_labels))

	def get_token_idx(self, token):
		return self.loaded_data["token2idx"][token] if token in self.loaded_data["token2idx"] else self.loaded_data["token2idx"][self.UNK_OP]

	def get_label_idx(self, label):
		return self.loaded_data["label2idx"][label]

	def get_processed_inputs(self, phrases):
		inputs = [[self.get_token_idx(token)
				   for token in phrase] for phrase in phrases]
		inputs = [self.pad_phrase(phrase, self.max_len) for phrase in inputs]
		return np.array(inputs)

	def convert_tokens2idx(self, phrase, max_len=None):
		idx_phrase = []
		for token in phrase.split():
			idx_phrase.append(self.get_token_idx(token))

		if max_len is None:
			return idx_phrase

		return self.pad_phrase(idx_phrase, max_len=max_len)
	
	def convert_idx2labels(self, phrase_labels):
		return [self.loaded_data["idx2label"][label] for label in phrase_labels]

	def get_labels(self):
		return [self.PAD_OP] + list(self.loaded_data["label2idx"].keys())


if __name__ == "__main__":
	u = Utils("./data")
	print(u.loaded_data.keys())
