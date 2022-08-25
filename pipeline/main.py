from .preprocess import PreProcessor
from .predict import Predictor
from .postprocess import PostProcessor
from settings import Utils

class Pipeline:
	def __init__(self, data_path, model_path):
		self.utils = Utils(data_path)
		self.preprocessor = PreProcessor(utils=self.utils)
		self.predictor = Predictor(model_path)
		self.postprocessor = PostProcessor(utils=self.utils)
	
	def run(self, inputs):
		outputs = self.preprocessor.run(inputs)
		outputs = self.predictor.run(outputs)
		outputs = self.postprocessor.run(self.preprocessor.tokenized_phrases, outputs)
		self.preprocessor.tokenized_phrases = []
		return outputs
