# coding:utf-8

# @Created : Macielyoung
# @Time : 2019/7/26
# @Function : Train a fasttext language model

from gensim.models import FastText

root = "/Users/maciel/Documents/Competition/InformationExtraction/"

class LanguageModel():
	def __init__(self, read_file, save_file):
		self.read_file = read_file
		self.save_file = save_file
		self.sentences = []

	def get_sentences(self):
		with open(root+self.read_file, 'r', encoding='utf-8') as f:
			for line in f.readlines():
				sentence = line.strip("\n").split("_")
				self.sentences.append(sentence)

	def train_and_save(self):
		model = FastText(self.sentences, size=256, window=5, min_count=1, iter=100)
		mode.save(self.save_file)

if __name__ == "__main__":
	read_file = "data/datagrand/corpus.txt"
	save_file = "data/datagrand/model"
	lm = LanguageModel(read_file, save_file)
	lm.get_sentences()
	lm.train_and_save()

