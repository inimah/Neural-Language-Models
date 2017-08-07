# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "01.08.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function

import sys
import numpy as np
from text_preprocessing import *
from lsa.vector_space import VectorSpace
from lsa.tfidf import TFIDF
from lsa.lsa import LSA


if __name__ == '__main__':

	#examples_1 = ["The cat in the hat disabled", "A cat is a fine pet ponies.", "Dogs and cats make good pets.","I haven't got a hat."]
	#tokenized_docs, vocab = generateSentVocab(examples_1)

	examples_2 = ["Machine learning is super fun", "Python is super, super cool", "Statistics is cool, too", "Data science is fun", "Python is great for machine learning", "I like football", "Football is great to watch"]
	tokenized_docs, vocab = generateSentVocab(examples_2)

	# save data
	savePickle(tokenized_docs,'tokenized_docs')
	savePickle(vocab,'vocab')

	vs = VectorSpace(tokenized_docs,vocab)

	
	# save matrices: BOW, sub-linear BOW, 
	savePickle(vs.td_bow,'td_bow')
	savePickle(vs.td_bow_sublin,'td_bow_sublin')


	tfidf = TFIDF(vs.td_bow)
	td_tfidf = tfidf.transform()

	# save tf-idf matrix transformed from BOW 
	savePickle(td_tfidf,'td_tfidf')


	# transpose input matrix (originally doc x term dimension -> to term x doc dimension)
	bow_t = np.transpose(vs.td_bow)
	# matrix decomposition of BOW
	lsa_bow = LSA(bow_t)
	lsa_bow.transform()
	# save matrices and objects
	savePickle(lsa_bow.u,'lsa_bow_u')
	savePickle(lsa_bow.sigma,'lsa_bow_sigma')
	savePickle(lsa_bow.diag_sigma,'lsa_bow_diag_sigma')
	savePickle(lsa_bow.vt,'lsa_bow_vt')
	savePickle(lsa_bow.transformed_matrix,'svd_bow')

	# matrix decomposition of BOW-sublinear
	bow_sublin_t = np.transpose(vs.td_bow_sublin)
	# matrix decomposition of BOW
	lsa_bow_sublin = LSA(bow_sublin_t)
	lsa_bow_sublin.transform()
	# save matrices and objects
	savePickle(lsa_bow_sublin.u,'lsa_bow_sublin_u')
	savePickle(lsa_bow_sublin.sigma,'lsa_bow_sublin_sigma')
	savePickle(lsa_bow_sublin.diag_sigma,'lsa_bow_sublin_diag_sigma')
	savePickle(lsa_bow_sublin.vt,'lsa_bow_sublin_vt')
	savePickle(lsa_bow_sublin.transformed_matrix,'svd_bow_sublin')
	
	# matrix decomposition of TF-IDF
	td_tfidf_t = np.transpose(td_tfidf)
	lsa_tfidf = LSA(td_tfidf_t)
	lsa_tfidf.transform()
	# save matrices and objects
	savePickle(lsa_tfidf.u,'lsa_tfidf_u')
	savePickle(lsa_tfidf.sigma,'lsa_tfidf_sigma')
	savePickle(lsa_tfidf.diag_sigma,'lsa_tfidf_diag_sigma')
	savePickle(lsa_tfidf.vt,'lsa_tfidf_vt')
	savePickle(lsa_tfidf.transformed_matrix,'svd_tfidf')


	
