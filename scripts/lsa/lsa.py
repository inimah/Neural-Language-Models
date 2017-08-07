# -*- coding: utf-8 -*-
#__author__ = "@tita"
# extended + simplified version of Joseph Wilk's github code
#__update__ = "01.08.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from scipy import linalg,dot
import numpy as np
from transform import Transform

class LSA(Transform):
	""" Latent Semantic Analysis(LSA).
		Apply transform to a document-term matrix to bring out latent relationships.
		These are found by analysing relationships between the documents and the terms they 
		contain.
	"""
	u = []
	sigma = []
	vt = []
	v = []
	diag_sigma = []

	transformed_matrix = []
	word_vector = []
	doc_vector = []



	def __init__(self, matrix):

		Transform.__init__(self, matrix)

		self.u = []
		self.sigma = []
		self.vt = []
		self.v = []

		self.diag_sigma = []

		self.transformed_matrix = []
		self.word_vector = []
		self.doc_vector = []


	def transform(self, dimensions=0):
		""" Calculate SVD of objects matrix: U . SIGMA . VT = MATRIX 
			Reduce the dimension of sigma by specified factor producing sigma'. 
			Then dot product the matrices:  U . SIGMA' . VT = MATRIX'
		"""
		rows,cols = self.matrix.shape

		if dimensions <= rows: #Its a valid reduction

			#Sigma comes out as a list rather than a matrix
			# if set as full_matrices=False, then the dimension of U (word vector) is not (n_words x n_words) but (n_words x n_docs)
			self.u,self.sigma,self.vt = linalg.svd(self.matrix,full_matrices=True)

			#Dimension reduction, build SIGMA'
			self.diag_sigma = linalg.diagsvd(self.sigma, len(self.matrix), len(self.vt))
			#for index in xrange(rows - dimensions, rows):
			#	self.diag_sigma[index] = 0



			#Reconstruct MATRIX'
			
			self.transformed_matrix = dot(dot(self.u, self.diag_sigma) ,self.vt)
			# word vector = u . s
			# this represents word similarity
			#self.word_vector = dot(self.u, linalg.diagsvd(self.sigma, self.sigma.shape[0], self.sigma.shape[1]))

			# doc vector = v . s
			# this represents document similarity 
			#self.v = np.transpose(self.vt)
			#self.doc_vector =  dot(linalg.diagsvd(self.sigma, self.sigma.shape[0], self.sigma.shape[1]),self.v)

			#return self.u, self.sigma, self.v, self.transformed_matrix, self.word_vector, self.doc_vector
			return self.u, self.sigma, self.vt, self.transformed_matrix
			

		else:
			print "dimension reduction cannot be greater than %s" % rows

