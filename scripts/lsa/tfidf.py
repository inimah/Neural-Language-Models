# -*- coding: utf-8 -*-
# Copyright (c) 2008-2013 Joseph Wilk
#__modification__ = "@tita"
#__update__ = "01.08.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from math import *
from transform import Transform

class TFIDF(Transform):

	transformed_matrix = []

	def __init__(self, matrix):
		Transform.__init__(self, matrix)
		self.document_total = len(self.matrix)
		self.transformed_matrix = []


	def transform(self):
		
		""" Apply TermFrequency (tf) * inverseDocumentFrequency(idf) for each matrix element.
		This evaluates how important a word is to a document in a corpus

		With a document-term matrix: matrix[x][y]
		tf[x][y] = frequency of term y in document x 
		idf[x][y] = log( abs(1 + total number of documents in corpus) / abs( 1 + number of documents with term y)  )
		or in math expression
		idf(term) = ( log ((1 + nd)/(1 + df(doc,term))) ) 
		where nd : number of document in corpus; 
		df : doc frequency (number of documents containing term)
		Note: This is not the only way to calculate tf*idf
		"""

		rows,cols = self.matrix.shape
		self.transformed_matrix = self.matrix.copy()

		for row in xrange(0, rows): #For each document

			# frequency of all terms in document x
			word_total = reduce(lambda x, y: x+y, self.matrix[row] )
			word_total = float(word_total)

			for col in xrange(0, cols): #For each term
				self.transformed_matrix[row,col] = float(self.transformed_matrix[row,col])

				if self.transformed_matrix[row][col] != 0:
					self.transformed_matrix[row,col] = self.computeTfIdf(row, col)

		return self.transformed_matrix

	
	# augmented smoothing version of doc frequency : 1 + df
	def docFrequency(self, col):
		count = 0
		rows,cols = self.matrix.shape
		for n in range(0,rows):
			if self.matrix[n][col] > 0:
				count += 1
		return 1 + count



	def computeTfIdf(self, row, col):

		tf = self.matrix[row][col] 

		# idf(term) = ( log ((1 + nd)/(1 + df(doc,term))) ) 
		idf = log(((1 + self.document_total) / float(self.docFrequency(col))))

		return tf * idf

	

	