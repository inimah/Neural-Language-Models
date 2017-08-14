# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "01.08.2017"
#__email__ = "i.nimah@tue.nl"


##################################################################################
# This tokenizer removes stopwords and uses stemmer
# as compared to tokenizer in text_preprocessing.py that preserves original texts 
##################################################################################

import os
import nltk
import string
from porter_stemmer import PorterStemmer

cwd = os.getcwd()


class Tokenizer:
	STOP_WORDS_FILE = cwd + '/stopwords/english' 

	stemmer = None
	stopwords = []

	def __init__(self, stopwords_io_stream = None):
		self.stemmer = PorterStemmer()
		
		if(not stopwords_io_stream):
		  stopwords_io_stream = open(Tokenizer.STOP_WORDS_FILE, 'r')

		self.stopwords = stopwords_io_stream.read().split()

	def tokenise_and_remove_stop_words(self, document):
		#if not document_list:
		#  return []
		  
		#vocabulary_string = " ".join(document_list)
		# return tokens of word	(in array list format)		
		tokenised_vocabulary_list = self._tokenise(document)
		clean_word_list = self._remove_stop_words(tokenised_vocabulary_list)

		

		return clean_word_list

	# here tokenized_documents is in 2D array list format (merging all documents into 1 list)
	def _vocab(self, tokenized_documents):

		# frequency occurence of terms
		tf = nltk.FreqDist(tokenized_documents)
		# terms / unique words
		terms = tf.keys()
		
		vocab = dict([(i,terms[i]) for i in range(len(terms))])

		return vocab

	def _remove_stop_words(self, list):
		""" Remove common words which have no search value """
		return [word for word in list if word not in self.stopwords ]


	def _tokenise(self, string):
		""" break string up into tokens and stem words """
		string = self._clean(string)
		words = string.split(" ")
		
		return [self.stemmer.stem(word, 0, len(word)-1) for word in words]

	def _clean(self, text):
		""" remove any nasty grammar tokens from string """
		text = text.translate(None,string.punctuation)
		text = text.replace(".","")
		text = text.replace("\s+"," ")
		text = text.lower()
		return text