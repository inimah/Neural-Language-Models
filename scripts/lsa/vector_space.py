# -*- coding: utf-8 -*-
#__author__ = "@tita"
# extended + simplified version of Joseph Wilk's github code
#__update__ = "01.08.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from lsa import LSA 
from tfidf import TFIDF 
import sys
import numpy as np
import math


try:
	from numpy import dot
	from numpy.linalg import norm
except:
	print "Error: Requires numpy from http://www.scipy.org/. Have you installed scipy?"
	sys.exit() 

class VectorSpace:
	""" A algebraic model for representing text documents as vectors of identifiers. 
	A document is represented as a vector. Each dimension of the vector corresponds to a 
	separate term. If a term occurs in the document, then the value in the vector is non-zero.
	"""

	# matrix for term documents
	td_bow = []
	td_bow_sublin = []
	td_tfidf = []
	# vocab (in dictionary format)
	word_index = {}


	def __init__(self, documents = [], vocab ={}, transforms = [TFIDF, LSA]):
		self.td_bow = []
		self.td_bow_sublin = []
		self.td_tfidf = []
		self.word_index = {}
		if len(documents) > 0:
			self.word_index = vocab
			self.td_bow, self.td_bow_sublin, self.td_tfidf = self.buildTermDocMatrix(documents, vocab, transforms)
			

	# document similarity
	def cosineSimilarity(self, document_id, matrix):
		""" find documents that are related to the document indexed by passed Id within the document Vectors"""
		doc_rank = [self._cosine(matrix[document_id], document_vector) for document_vector in matrix]
		doc_rank.sort(reverse = True)
		return doc_rank


	# term frequency
	def countFrequency(self, term, tokenized_document):
		return tokenized_document.count(term)

	# sublinear tf scaling --> log tf
	def sublinearTf(self, term, tokenized_document):
		count = tokenized_document.count(term)
		if count == 0:
			return 0
		return 1 + math.log(count)


	def buildTermDocMatrix(self, tokenized_documents, vocab, transforms):
		
		

		bow_matrix = self.createBOW(tokenized_documents) 
		#bow_matrix = reduce(lambda matrix,transform: transform(bow_matrix).transform(), transforms, bow_matrix)
		self.td_bow = bow_matrix

		sublin_bow_matrix = self.createSublinearBOW(tokenized_documents) 
		#sublin_bow_matrix = reduce(lambda matrix,transform: transform(sublin_bow_matrix).transform(), transforms, sublin_bow_matrix)
		self.td_bow_sublin = sublin_bow_matrix

		tfidf_doc = self.createTfIdf(tokenized_documents)
		#tfidf_doc = reduce(lambda matrix,transform: transform(tfidf_doc).transform(), transforms, tfidf_doc)
		self.td_tfidf = tfidf_doc

		return self.td_bow, self.td_bow_sublin, self.td_tfidf

  
	# create BOW dictionary list (term frequency)
	# rows : by documents
	# cols : by words
	def createBOW(self, tokenized_documents):

		bow_matrix = []

		print('creating Bag-of-words matrix...')
		
		for doc in tokenized_documents:
			row_vec = []
			for term in self.word_index.values():
				tf = self.countFrequency(term, doc)
				row_vec.append(tf)
			bow_matrix.append(row_vec)

		return bow_matrix

	def createSublinearBOW(self, tokenized_documents):

		bow_matrix = []

		print('creating Bag-of-words matrix with sublinear smoothing...')
		
		for doc in tokenized_documents:
			row_vec = []
			for term in self.word_index.values():
				tf = self.sublinearTf(term, doc)
				row_vec.append(tf)
			bow_matrix.append(row_vec)

		return bow_matrix

	def docFrequency(self, term, tokenized_documents):
		count = 0
		for doc in tokenized_documents:
			if self.countFrequency(term, doc) > 0:
				count += 1
		return 1 + count



	def computeIDF(self, term, tokenized_documents):

		# idf(term) = ( log ((1 + nd)/(1 + df(doc,term))) ) 
		# where nd : number of document in corpus; 
		# df : doc frequency (number of documents containing term)

		idf = math.log( (1 + len(tokenized_documents)) / float(self.docFrequency(term, tokenized_documents)))

		return idf

	
   
	def createTfIdf(self, tokenized_documents):

		print('creating TFIDF matrix...')


		tfidfDoc = []
		for doc in tokenized_documents:
			tfidf = []
			for term in self.word_index.values():
				#tf = self.sublinearTf(term, doc)
				tf = self.countFrequency(term, doc)

				# idf(term) = ( log ((1 + nd)/(1 + df(doc,term))) ) 
				# where nd : number of document in corpus; 
				# df : doc frequency (number of documents containing term)

				idf = math.log( (1 + len(tokenized_documents)) / float(self.docFrequency(term, tokenized_documents)))

				tfidf.append(tf * idf)
			tfidfDoc.append(tfidf)

		return tfidfDoc

	
	
		
	def _cosine(self, vector1, vector2):
		""" related documents j and q are in the concept space by comparing the vectors :
			cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
		return float(dot(vector1,vector2) / (norm(vector1) * norm(vector2)))