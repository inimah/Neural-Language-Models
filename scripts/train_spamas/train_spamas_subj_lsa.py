# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "15.09.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
sys.path.insert(0,'..')
import time
import numpy as np
from text_preprocessing import *
from language_models import *
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences 
from sklearn.preprocessing import LabelEncoder
# the following modules are in lsa folder
from lsa.vector_space import VectorSpace
from lsa.tfidf import TFIDF
from lsa.lsa import LSA
from lsa.tokenizer import Tokenizer


# Update the code and re-run due to the final-reduced version of preprocessed data


PATH = '/home/inimah/git/Neural-Language-Models/scripts/prepdata/spamassasin'



if __name__ == '__main__':



	# reading stored pre-processed (in pickle format)

	'''

	

	############################
	# Matrix decomposition
	# without stopwords elimination and stemming


	vs = VectorSpace(tokenized_docs,vocab)

	# save matrices: BOW, sub-linear BOW, 
	savePickle(vs.td_bow,'spamas_lsa_td_bow')
	savePickle(vs.td_bow_sublin,'spamas_lsa_td_bow_sublin')
	savePickle(vs.td_bow_info,'spamas_lsa_td_bow_info')
	savePickle(vs.td_bow_sublin_info,'spamas_lsa_td_bow_sublin_info')


	tfidf = TFIDF(vs.td_bow)
	td_tfidf = tfidf.transform()

	# save tf-idf matrix transformed from BOW 
	savePickle(td_tfidf,'spamas_lsa_td_tfidf')
	savePickle(vs.td_tfidf_info,'spamas_lsa_td_tfidf_info')

	############################
	# matrix decomposition of BOW

	# transpose input matrix (originally doc x term dimension -> to term x doc dimension)
	bow_t = np.transpose(vs.td_bow)

	# eliminate non-occur words in document corpus (due to sampling)
	bow_null_ind = np.where(~bow_t.any(axis=1))[0]
	savePickle(bow_null_ind,'spamas_lsa_bow_null_ind')

	td_bow_t = []
	for i in range(len(bow_t)):
		if i not in bow_null_ind:
			td_bow_t.append(bow_t[i])

	
	lsa_bow = LSA(td_bow_t)
	lsa_bow.transform()
	# save matrices and objects
	savePickle(lsa_bow.u,'spamas_lsa_bow_u')
	savePickle(lsa_bow.sigma,'spamas_lsa_bow_sigma')
	savePickle(lsa_bow.diag_sigma,'spamas_lsa_bow_diag_sigma')
	savePickle(lsa_bow.vt,'spamas_lsa_bow_vt')
	savePickle(lsa_bow.transformed_matrix,'spamas_svd_bow')

	############################
	# matrix decomposition of BOW-sublinear
	bow_sublin_t = np.transpose(vs.td_bow_sublin)
	# eliminate non-occur words in document corpus 
	bow_sublin_null_ind = np.where(~bow_sublin_t.any(axis=1))[0]
	savePickle(bow_sublin_null_ind,'spamas_lsa_bow_sublin_null_ind')
	td_bow_sublin_t = []
	for i in range(len(bow_sublin_t)):
		if i not in bow_sublin_null_ind:
			td_bow_sublin_t.append(bow_sublin_t[i])

	lsa_bow_sublin = LSA(td_bow_sublin_t)
	lsa_bow_sublin.transform()
	# save matrices and objects
	savePickle(lsa_bow_sublin.u,'spamas_lsa_bow_sublin_u')
	savePickle(lsa_bow_sublin.sigma,'spamas_lsa_bow_sublin_sigma')
	savePickle(lsa_bow_sublin.diag_sigma,'spamas_lsa_bow_sublin_diag_sigma')
	savePickle(lsa_bow_sublin.vt,'spamas_lsa_bow_sublin_vt')
	savePickle(lsa_bow_sublin.transformed_matrix,'spamas_svd_bow_sublin')

	############################
	# matrix decomposition of TF-IDF
	tfidf_t = np.transpose(td_tfidf)
	# eliminate non-occur words in document corpus 
	tfidf_null_ind = np.where(~tfidf_t.any(axis=1))[0]
	savePickle(tfidf_null_ind,'spamas_lsa_tfidf_null_ind')
	td_tfidf_t = []
	for i in range(len(tfidf_t)):
		if i not in tfidf_null_ind:
			td_tfidf_t.append(tfidf_t[i])

	lsa_tfidf = LSA(td_tfidf_t)
	lsa_tfidf.transform()
	# save matrices and objects
	savePickle(lsa_tfidf.u,'spamas_lsa_tfidf_u')
	savePickle(lsa_tfidf.sigma,'spamas_lsa_tfidf_sigma')
	savePickle(lsa_tfidf.diag_sigma,'spamas_lsa_tfidf_diag_sigma')
	savePickle(lsa_tfidf.vt,'spamas_lsa_tfidf_vt')
	savePickle(lsa_tfidf.transformed_matrix,'spamas_svd_tfidf')


	

	############################
	# Matrix decomposition
	# with stopwords elimination and stemming

	
	vs_SW = VectorSpace(tokenized_docsSW,vocabSW)

	# save matrices: BOW, sub-linear BOW, 
	savePickle(vs_SW.td_bow,'spamas_lsa_td_bowSW')
	savePickle(vs_SW.td_bow_sublin,'spamas_lsa_td_bow_sublinSW')
	savePickle(vs_SW.td_bow_info,'spamas_lsa_td_bowSW_info')
	savePickle(vs_SW.td_bow_sublin_info,'spamas_lsa_td_bow_sublinSW_info')


	tfidfSW = TFIDF(vs_SW.td_bow)
	td_tfidfSW = tfidfSW.transform()

	# save tf-idf matrix transformed from BOW 
	savePickle(td_tfidfSW,'spamas_lsa_td_tfidfSW')
	savePickle(vs_SW.td_tfidf_info,'spamas_lsa_td_tfidfSW_info')

	############################
	# matrix decomposition of BOW

	# transpose input matrix (originally doc x term dimension -> to term x doc dimension)
	bow_tSW = np.transpose(vs_SW.td_bow)

	# eliminate non-occur words in document corpus (due to sampling)
	bow_null_indSW = np.where(~bow_tSW.any(axis=1))[0]
	savePickle(bow_null_indSW,'spamas_lsa_bow_null_indSW')

	td_bow_tSW = []
	for i in range(len(bow_tSW)):
		if i not in bow_null_indSW:
			td_bow_tSW.append(bow_tSW[i])

	
	lsa_bowSW = LSA(td_bow_tSW)
	lsa_bowSW.transform()
	# save matrices and objects
	savePickle(lsa_bowSW.u,'spamas_lsa_bow_uSW')
	savePickle(lsa_bowSW.sigma,'spamas_lsa_bow_sigmaSW')
	savePickle(lsa_bowSW.diag_sigma,'spamas_lsa_bow_diag_sigmaSW')
	savePickle(lsa_bowSW.vt,'spamas_lsa_bow_vtSW')
	savePickle(lsa_bowSW.transformed_matrix,'spamas_svd_bowSW')

	############################
	# matrix decomposition of BOW-sublinear
	bow_sublin_tSW = np.transpose(vs_SW.td_bow_sublin)
	# eliminate non-occur words in document corpus 
	bow_sublin_null_indSW = np.where(~bow_sublin_tSW.any(axis=1))[0]
	savePickle(bow_sublin_null_indSW,'spamas_lsa_bow_sublin_null_indSW')
	td_bow_sublin_tSW = []
	for i in range(len(bow_sublin_tSW)):
		if i not in bow_sublin_null_indSW:
			td_bow_sublin_tSW.append(bow_sublin_tSW[i])

	lsa_bow_sublinSW = LSA(td_bow_sublin_tSW)
	lsa_bow_sublinSW.transform()
	# save matrices and objects
	savePickle(lsa_bow_sublinSW.u,'spamas_lsa_bow_sublin_uSW')
	savePickle(lsa_bow_sublinSW.sigma,'spamas_lsa_bow_sublin_sigmaSW')
	savePickle(lsa_bow_sublinSW.diag_sigma,'spamas_lsa_bow_sublin_diag_sigmaSW')
	savePickle(lsa_bow_sublinSW.vt,'spamas_lsa_bow_sublin_vtSW')
	savePickle(lsa_bow_sublinSW.transformed_matrix,'spamas_svd_bow_sublinSW')

	############################
	# matrix decomposition of TF-IDF
	tfidf_tSW = np.transpose(td_tfidfSW)
	# eliminate non-occur words in document corpus 
	tfidf_null_indSW = np.where(~tfidf_tSW.any(axis=1))[0]
	savePickle(tfidf_null_indSW,'spamas_lsa_tfidf_null_indSW')
	td_tfidf_tSW = []
	for i in range(len(tfidf_tSW)):
		if i not in tfidf_null_indSW:
			td_tfidf_tSW.append(tfidf_tSW[i])

	lsa_tfidfSW = LSA(td_tfidf_tSW)
	lsa_tfidfSW.transform()
	# save matrices and objects
	savePickle(lsa_tfidfSW.u,'spamas_lsa_tfidf_uSW')
	savePickle(lsa_tfidfSW.sigma,'spamas_lsa_tfidf_sigmaSW')
	savePickle(lsa_tfidfSW.diag_sigma,'spamas_lsa_tfidf_diag_sigmaSW')
	savePickle(lsa_tfidfSW.vt,'spamas_lsa_tfidf_vtSW')
	savePickle(lsa_tfidfSW.transformed_matrix,'spamas_svd_tfidfSW')

	'''

	# Final vocabulary list after being reduced from less frequent words (links, noises)
	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))
	subjectSW_vocab = readPickle(os.path.join(PATH,'spamas_reducedvocabSW'))

	# Final tokenized documents with maximum word length = 25 words
	# with labels 
	subject = readPickle(os.path.join(PATH,'spamas_fin_labelled_subj'))
	
	# for tokenized subject title with stopword removing and stemming, have not been preprocessed yet
	# as such it is run here
	pre_subjectSW = readPickle(os.path.join(PATH,'spamas_reducedTokenSubjSW'))

	labelled_subjSW = []
	for i in pre_subjectSW:
		for j, tokens in enumerate(pre_subjectSW[i]):
			labelled_subjSW.append((i,tokens))

	# discard subjects with number of words > 25 (as being seen in statistics of subject title)
	subjectSW = []
	for i, data in enumerate(labelled_subjSW):
		if len(data[1]) <= 25:
			subjectSW.append((data[0],data[1]))

	# save reduced versioned of labelled tokenized documents
	savePickle(subjectSW,'spamas_fin_labelled_subjSW')

	# Encode text into numerical tokenized format
	encoded_docsSW = _encodeLabelledText(subjectSW,subjectSW_vocab)
	savePickle(encoded_docsSW,'spamas_fin_encoded_subjSW')

	# check statistic of each class (maximum - average number of words per class)
	count_wordsSW = _countWord(encoded_docsSW)
	savePickle(count_wordsSW,'spamas_count_wordsSW')




