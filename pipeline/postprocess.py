class PostProcessor:
	def __init__(self, utils):
		self.utils = utils

	def remove_padding(self, tokenized_phrase, prediction):
		return prediction[:len(tokenized_phrase)]

	def get_prediction_labels(self, prediction):
		return self.utils.convert_idx2labels(prediction)

	def map_tokens2predictions(self, tokenized_phrase, prediction):
		return list(zip(tokenized_phrase, prediction))

	def run(self, tokenized_phrases, predictions):
		mapped_tokens2labels = []
		for idx, prediction in enumerate(predictions):
			unpadded_predictions = self.remove_padding(tokenized_phrases[idx], prediction)
			labelled_predictions = self.get_prediction_labels(unpadded_predictions)
			print(tokenized_phrases[idx], labelled_predictions)
			mapped_tokens2labels.append(self.map_tokens2predictions(tokenized_phrases[idx], labelled_predictions))
		return mapped_tokens2labels
