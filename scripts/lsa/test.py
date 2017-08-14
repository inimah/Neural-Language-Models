# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "01.08.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function

import sys
import numpy as np
from text_preprocessing import *
from vector_space import VectorSpace
from tfidf import TFIDF
from lsa import LSA
from tokenizer import Tokenizer


if __name__ == '__main__':

	#examples_1 = ["The cat in the hat disabled", "A cat is a fine pet ponies.", "Dogs and cats make good pets.","I haven't got a hat."]
	#tokenized_docs, vocab = generateSentVocab(examples_1)

	examples_2 = ["Machine learning is super fun", "Python is super, super cool", "Statistics is cool, too", "Data science is fun", "Python is great for machine learning", "I like football", "Football is great to watch"]

	# preserves stopwords & without stemming
	# tokenized_docs, vocab = generateSentVocab(examples_2)
	## tokenizer with eliminating stopwords and stemming
	


	#examples_3 = ["Dogs eat the same things that cats eat", "No dog is a mouse", "Mice eat little things", "Cats often play with rats and mice", "Cats often play, but not with other cats"]
	#tokenized_docs, vocab = generateSentVocab(examples_3)

	tokenized_docs = []
	docwords = []

	tokenizer = Tokenizer()


	for text in examples_2:
		
		tmp = tokenizer.tokenise_and_remove_stop_words(text)
		tokenized_docs.append(tmp)

	docwords = sum(tokenized_docs,[])

	vocab = tokenizer._vocab(docwords)

	
	

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
	# eliminate non-occur words in document corpus 
	bow_null_ind = np.where(~bow_t.any(axis=1))[0]
	savePickle(bow_null_ind,'bow_null_ind')
	td_bow_t = []
	for i in range(len(bow_t)):
		if i not in bow_null_ind:
			td_bow_t.append(bow_t[i])

	# matrix decomposition of BOW
	lsa_bow = LSA(td_bow_t)
	lsa_bow.transform()
	# save matrices and objects
	savePickle(lsa_bow.u,'lsa_bow_u')
	savePickle(lsa_bow.sigma,'lsa_bow_sigma')
	savePickle(lsa_bow.diag_sigma,'lsa_bow_diag_sigma')
	savePickle(lsa_bow.vt,'lsa_bow_vt')
	savePickle(lsa_bow.transformed_matrix,'svd_bow')

	# matrix decomposition of BOW-sublinear
	bow_sublin_t = np.transpose(vs.td_bow_sublin)
	# eliminate non-occur words in document corpus 
	bow_sublin_null_ind = np.where(~bow_sublin_t.any(axis=1))[0]
	savePickle(bow_sublin_null_ind,'bow_sublin_null_ind')
	td_bow_sublin_t = []
	for i in range(len(bow_sublin_t)):
		if i not in bow_sublin_null_ind:
			td_bow_sublin_t.append(bow_sublin_t[i])

	# matrix decomposition of BOW
	lsa_bow_sublin = LSA(td_bow_sublin_t)
	lsa_bow_sublin.transform()
	# save matrices and objects
	savePickle(lsa_bow_sublin.u,'lsa_bow_sublin_u')
	savePickle(lsa_bow_sublin.sigma,'lsa_bow_sublin_sigma')
	savePickle(lsa_bow_sublin.diag_sigma,'lsa_bow_sublin_diag_sigma')
	savePickle(lsa_bow_sublin.vt,'lsa_bow_sublin_vt')
	savePickle(lsa_bow_sublin.transformed_matrix,'svd_bow_sublin')
	
	# matrix decomposition of TF-IDF
	tfidf_t = np.transpose(td_tfidf)
	# eliminate non-occur words in document corpus 
	tfidf_null_ind = np.where(~tfidf_t.any(axis=1))[0]
	savePickle(tfidf_null_ind,'tfidf_null_ind')
	td_tfidf_t = []
	for i in range(len(tfidf_t)):
		if i not in tfidf_null_ind:
			td_tfidf_t.append(tfidf_t[i])

	lsa_tfidf = LSA(td_tfidf_t)
	lsa_tfidf.transform()
	# save matrices and objects
	savePickle(lsa_tfidf.u,'lsa_tfidf_u')
	savePickle(lsa_tfidf.sigma,'lsa_tfidf_sigma')
	savePickle(lsa_tfidf.diag_sigma,'lsa_tfidf_diag_sigma')
	savePickle(lsa_tfidf.vt,'lsa_tfidf_vt')
	savePickle(lsa_tfidf.transformed_matrix,'svd_tfidf')



