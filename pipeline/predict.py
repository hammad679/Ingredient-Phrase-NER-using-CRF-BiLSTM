import pathlib
import tensorflow as tf

class Predictor():
	def __init__(self, model_path):
		model_path = pathlib.Path(model_path)
		if not model_path.exists() or not model_path.is_dir():
			raise ValueError()

		self.model = tf.keras.models.load_model(model_path)
	
	def run(self, inputs):
		return self.model.predict(inputs)
