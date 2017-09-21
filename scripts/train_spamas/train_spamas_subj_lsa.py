# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "15.09.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
sys.path.insert(0,'../..')
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
np.random.seed([3,1415])


# Update the code and re-run due to the final-reduced version of preprocessed data


PATH = '/home/inimah/git/Neural-Language-Models/scripts/prepdata/spamassasin'



if __name__ == '__main__':



	# reading stored pre-processed (in pickle format)

	# Final vocabulary list after being reduced from less frequent words (links, noises)
	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))
	subjectSW_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocabSW'))

	# Final tokenized documents with labels 
	subject = readPickle(os.path.join(PATH,'spamas_fin_labelled_subj'))
	subjectSW = readPickle(os.path.join(PATH,'spamas_fin_labelled_subjSW'))

	# sampling 1000 documents for each class
	spam_docs = []
	easy_ham_docs = []
	hard_ham_docs = [] 
	for i, data in enumerate(subject):
		if data[0] == 'spam':
			spam_docs.append(data)
		elif data[0] == 'hard_ham':
			hard_ham_docs.append(data)
		elif data[0] == 'easy_ham':
			easy_ham_docs.append(data)

	spam = np.array(spam_docs,dtype=object)
	easy_ham = np.array(easy_ham_docs,dtype=object)
	hard_ham = np.array(hard_ham_docs,dtype=object)

	# shuffling data for each class
	ind_rand1 = np.arange(len(spam))
	np.random.shuffle(ind_rand1)
	spam_dat = list(spam[ind_rand1])

	ind_rand2 = np.arange(len(easy_ham))
	np.random.shuffle(ind_rand2)
	easy_ham_dat = list(easy_ham[ind_rand2])

	ind_rand3 = np.arange(len(hard_ham))
	np.random.shuffle(ind_rand3)
	hard_ham_dat = list(hard_ham[ind_rand3])

	# and sampling 1000 instances / documents

	subject1000 = []
	nSpam = len(spam_dat)
	nEasyHam = len(easy_ham_dat)
	nHardHam = len(hard_ham_dat)

	if nSpam > 1000:
		spam_sampling = list(spam_dat[:1001])
	else:
		spam_sampling = list(spam_dat)

	if nEasyHam > 1000:
		easy_ham_sampling = list(easy_ham_dat[:1001])
	else:
		easy_ham_sampling = list(easy_ham_dat)

	if nHardHam > 1000:
		hard_ham_sampling = list(hard_ham_dat[:1001])
	else:
		hard_ham_sampling = list(hard_ham_dat)

	subject1000.extend(spam_sampling)
	subject1000.extend(easy_ham_sampling)
	subject1000.extend(hard_ham_sampling)


	tokenized_docs = []
	class_labels = []
	for i, data in enumerate(subject1000):
		class_labels.append(data[0])
		tokenized_docs.append(data[1])

	savePickle(subject1000,'subject1000')

	##################################
	# for stopword removing and stemming

	spam_docsSW = []
	easy_ham_docsSW = []
	hard_ham_docsSW = [] 
	for i, data in enumerate(subjectSW):
		if data[0] == 'spam':
			spam_docsSW.append(data)
		elif data[0] == 'hard_ham':
			hard_ham_docsSW.append(data)
		elif data[0] == 'easy_ham':
			easy_ham_docsSW.append(data)

	spamSW = np.array(spam_docsSW,dtype=object)
	easy_hamSW = np.array(easy_ham_docsSW,dtype=object)
	hard_hamSW = np.array(hard_ham_docsSW,dtype=object)

	# shuffling data for each class
	ind_rand1SW = np.arange(len(spamSW))
	np.random.shuffle(ind_rand1SW)
	spam_datSW = list(spamSW[ind_rand1SW])

	ind_rand2SW = np.arange(len(easy_hamSW))
	np.random.shuffle(ind_rand2SW)
	easy_ham_datSW = list(easy_hamSW[ind_rand2SW])

	ind_rand3SW = np.arange(len(hard_hamSW))
	np.random.shuffle(ind_rand3SW)
	hard_ham_datSW = list(hard_hamSW[ind_rand3SW])

	# and sampling 1000 instances / documents

	subject1000SW = []
	nSpamSW = len(spam_datSW)
	nEasyHamSW = len(easy_ham_datSW)
	nHardHamSW = len(hard_ham_datSW)

	if nSpamSW > 1000:
		spam_samplingSW = list(spam_datSW[:1001])
	else:
		spam_samplingSW = list(spam_datSW)

	if nEasyHamSW > 1000:
		easy_ham_samplingSW = list(easy_ham_datSW[:1001])
	else:
		easy_ham_samplingSW = list(easy_ham_datSW)

	if nHardHamSW > 1000:
		hard_ham_samplingSW = list(hard_ham_datSW[:1001])
	else:
		hard_ham_samplingSW = list(hard_ham_datSW)

	subject1000SW.extend(spam_samplingSW)
	subject1000SW.extend(easy_ham_samplingSW)
	subject1000SW.extend(hard_ham_samplingSW)

	tokenized_docsSW = []
	class_labelsSW = []
	for i, data in enumerate(subject1000SW):
		class_labelsSW.append(data[0])
		tokenized_docsSW.append(data[1])

	savePickle(subject1000SW,'subject1000SW')


	############################
	# Matrix decomposition
	# without stopwords elimination and stemming


	vs = VectorSpace(tokenized_docs,subject_vocab)

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

	# eliminate non-occur words in document corpus (due to sampling if anys)
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

	
	vs_SW = VectorSpace(tokenized_docsSW,subjectSW_vocab)

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

	