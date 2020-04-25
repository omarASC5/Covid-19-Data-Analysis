import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Class to hold article data
class Article(object):
	def __init__(self, title, abstract, link):
		self.title = title
		self.abstract = abstract
		self.link = link
		self.words = list()

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
    # Function for filtering the non covid related words
	def filter_words(self, mode="most", which="abstract"):

		# Initialize Model
		vec = CountVectorizer()
		stop_words = set(stopwords.words('english'))
		if (which == "abstract"):
			tokenized_abstract = word_tokenize(str(self.abstract))
			tokenized_abstract = [word for word in tokenized_abstract if word not in stop_words]
			vec.fit(tokenized_abstract)
			bag_of_words = vec.fit_transform(tokenized_abstract)
			sum_words = bag_of_words.sum(axis=0)
			words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
			if (mode == "most"):
				words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
			elif (mode == "least"):
				words_freq = sorted(words_freq, key = lambda x: x[1])

			return words_freq

		elif (which == "title"):
			tokenized_title = word_tokenize(str(self.title))
			tokenized_title = [word for word in tokenized_title if word not in stop_words]
			vec.fit(tokenized_title)
			bag_of_words = vec.fit_transform(tokenized_title)
			sum_words = bag_of_words.sum(axis=0)
			words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
			if (mode == "most"):
				words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
			elif (mode == "least"):
				words_freq = sorted(words_freq, key = lambda x: x[1])

			return words_freq

	def filter_related_words(self, mode="most", which="abstract", limit=5):
		return self.filter_words(mode, which)[:limit]
        
# Function for parsing data
def parse_data(filename):
	with open(filename, 'r') as f:
		i = 0;
		for line in f:
			#print(line)
			if i == 5:
				break
			i += 1

# Gather data
parse_data('metadata.csv')
dataframe = pd.read_csv('metadata.csv', encoding='utf-8')

# Grab useful columns
titles = dataframe['title'].tolist()
abstracts = dataframe['abstract'].tolist()
links = dataframe['url'].tolist()

# Make Article objects
def make_article_objects(titles, abstracts, links):
	article_objects = [None] * len(titles)

	for i in range(len(titles)):
		article_objects[i] = Article(titles[i], abstracts[i], links[i])

	return article_objects

article_objects = make_article_objects(titles, abstracts, links)

i = 0
for article_object in article_objects:
	if (i == 100) : break
	# print(article_object.filter_words(mode="least", which="title"))
	print(article_object.filter_related_words(mode="most", which="abstract"), end='\n\n')
	i += 1