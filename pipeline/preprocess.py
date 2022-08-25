import re

import spacy
from spacy.tokenizer import Tokenizer

import tensorflow as tf
import numpy as np

class PreProcessor():
	def __init__(self, utils, spacy_opts={ "spacy_lang_pack": "en_core_web_sm" }, max_len=50):
		self.nlp = spacy.load("en_core_web_sm")
		self.max_len = max_len

		self.utils = utils

		self.tokenized_phrases = []

		if "tokenizer" not in spacy_opts:
			# only split on whitespace
			self.nlp.tokenizer = Tokenizer(self.nlp.vocab, token_match=re.compile(r"\S+").match)
		elif spacy_opts["tokenizer"] != "default" and type(spacy_opts["tokenizer"]) is str:
			self.nlp.tokenizer = spacy_opts["tokenizer"]
	
	def truncate(self, tokenized_input):
		if len(tokenized_input) <= self.max_len:
			return tokenized_input
		return tokenized_input[:self.max_len]

	def tokenize_phrase(self, phrase):
		return self.truncate([token.lemma_ for token in self.nlp(phrase)])

	def normalize_phrase(self, phrase):
		tokens = self.tokenize_phrase(phrase)
		self.tokenized_phrases.append(tokens)
		return " ".join(tokens)
	
	def run(self, phrases):
		inputs = [self.utils.convert_tokens2idx(self.normalize_phrase(phrase), max_len=self.max_len) for phrase in phrases]
		return tf.convert_to_tensor(np.array(inputs), dtype="int32", name="inputs")
