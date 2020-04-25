import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
import csv

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
		vec = CountVectorizer(min_df=1)
		stop_words = set(stopwords.words('english'))
		if (which == "abstract"):
			tokenized_abstract = word_tokenize(str(self.abstract))
			tokenized_abstract = [word for word in tokenized_abstract if word not in stop_words]
			try:
				vec.fit(tokenized_abstract)
				bag_of_words = vec.fit_transform(tokenized_abstract)
				sum_words = bag_of_words.sum(axis=0)
				words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
				if (mode == "most"):
					words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
				elif (mode == "least"):
					words_freq = sorted(words_freq, key = lambda x: x[1])

				return words_freq

			except ValueError:
				return list()


		elif (which == "title"):
			tokenized_title = word_tokenize(str(self.title))
			tokenized_title = [word for word in tokenized_title if word not in stop_words]
			try:
				vec.fit(tokenized_title)
				bag_of_words = vec.fit_transform(tokenized_title)
				sum_words = bag_of_words.sum(axis=0)
				words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
				if (mode == "most"):
					words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
				elif (mode == "least"):
					words_freq = sorted(words_freq, key = lambda x: x[1])

				return words_freq

			except ValueError:
				return list()

	def filter_related_words(self, mode="most", which="abstract", limit=5):
		return self.filter_words(mode, which)[:limit]
        
# Function for parsing data
def parse_data(filename):
	with open(filename, 'r') as f:
		i = 0;
		for line in f:
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

def find_trends(article_objects, mode="most", which="abstract", limit=5):
	words = []

	step = 0
	for article_object in article_objects:
		# Get the most or least common words that relate to and describe COVID-19
		# From either the title or abstract
		if (step % 500 == 0) : print('Done', step)
		related_words = article_object.filter_related_words(mode=mode, which=which, limit=limit)
		for related_word in related_words:
			words.append(related_word)
		
		step += 1

	return words

def write_findings_to_files(words_file, numbers_file, mode, which):
	trends = find_trends(article_objects, mode=mode, which=which)
	with open(words_file, 'w+') as f:
		writer = csv.writer(f)
		trends_words = []

		for frequency in trends:
			trends_words.append([frequency[0]])

		writer.writerows(trends_words)

	with open(numbers_file, 'w+') as f:
		writer = csv.writer(f)
		trends_numbers = []

		for frequency in trends:
			trends_numbers.append([str(frequency[1])])
			
		writer.writerows(trends_numbers)

def read_findings_from_files(words_file, numbers_file):
	trends = []

	with open(words_file, 'r') as f1, open(numbers_file, 'r') as f2:
		for word, number in zip(f1, f2):
			word = word.rstrip()
			number = int(number.rstrip())
			trends.append((word, number))

	return trends

# These write_findings_to_files() only have to be run ONCE -> to populate files starting all corresponding data. Process is LONG!
# write_findings_to_files('most_common_abstract_words.csv', 'most_common_abstract_numbers.csv', 'most', 'abstract')
# write_findings_to_files('most_common_title_words.csv', 'most_common_title_numbers.csv', 'most', 'title')
# write_findings_to_files('least_common_abstract_words.csv', 'least_common_abstract_numbers.csv', 'least', 'abstract')
# write_findings_to_files('least_common_title_words.csv', 'least_common_title_numbers.csv', 'least', 'title')

# Read preprocessed calls from files for speedy operations
most_common_abstract = read_findings_from_files('most_common_abstract_words.csv', 'most_common_abstract_numbers.csv')
least_common_abstract = read_findings_from_files('least_common_abstract_words.csv', 'least_common_abstract_numbers.csv')
most_common_title = read_findings_from_files('most_common_title_words.csv', 'most_common_title_numbers.csv')
least_common_title = read_findings_from_files('least_common_title_words.csv', 'least_common_title_numbers.csv')

print("Most common words from abstract:")	
print(len(most_common_abstract), most_common_abstract[:10])

print("Most common words from title:")
print(len(most_common_title), most_common_title[:10])

print("Least common words from abstract:")
print(len(least_common_abstract), least_common_abstract[:10])

print("Least common words from title:")
print(len(least_common_title), least_common_title[:10])