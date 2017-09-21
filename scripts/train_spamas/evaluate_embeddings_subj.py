# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "01.07.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
sys.path.insert(0,'..')
import itertools
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import sys
from scipy import linalg,dot
from text_preprocessing import *
from gensim.models import Word2Vec, Doc2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

np.random.seed([3,1415])


PATH = '/home/inimah/git/Neural-Language-Models/scripts/prepdata/spamassasin'
VECTOR_PATH = '/home/inimah/git/Neural-Language-Models/scripts/train_spamas/'

if __name__ == '__main__':

	# vocabulary
	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))
	subjectSW_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocabSW'))
	# subjects
	subject1000 = readPickle(os.path.join(VECTOR_PATH,'sampling/subject1000'))
	subject1000SW = readPickle(os.path.join(VECTOR_PATH,'sampling/subject1000SW'))

	numClasses = 3
	
	# the following labels are still in nominal form ('spam', 'easy_ham', 'hard_ham')
	subject1000_arr = np.array(subject1000,dtype=object)
	yLabels = list(subject1000_arr[:,0])
	

	# also need to know what words being discarded through the reconstruction of vectors as opposed to original vocab
	# due to sampling (in LSA, only maximum 1000 documents per class ie being used to train model)
	bow_discarded_words = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_null_ind'))
	bow_discarded_wordsSW = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_null_indSW'))
	sublinbow_discarded_words = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_sublin_null_ind'))
	sublinbow_discarded_wordsSW = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_sublin_null_indSW'))
	tfidf_discarded_words = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_tfidf_null_ind'))
	tfidf_discarded_wordsSW = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_tfidf_null_indSW'))


	# create new reduced vocab after being discarded by LSA training process 
	bow_words = []
	for i, v in subject_vocab.iteritems():
		if i not in bow_discarded_words:
			bow_words.append(v)

	vocab_bow=dict([(i,bow_words[i]) for i in range(len(bow_words))])

	sublinbow_words = []
	for i, v in subject_vocab.iteritems():
		if i not in sublinbow_discarded_words:
			sublinbow_words.append(v)

	vocab_sublinbow=dict([(i,sublinbow_words[i]) for i in range(len(sublinbow_words))])

	tfidf_words = []
	for i, v in subject_vocab.iteritems():
		if i not in tfidf_discarded_words:
			tfidf_words.append(v)

	vocab_tfidf=dict([(i,tfidf_words[i]) for i in range(len(tfidf_words))])

	bow_wordsSW = []
	for i, v in subjectSW_vocab.iteritems():
		if i not in bow_discarded_wordsSW:
			bow_wordsSW.append(v)

	vocab_bowSW=dict([(i,bow_wordsSW[i]) for i in range(len(bow_wordsSW))])

	sublinbow_wordsSW = []
	for i, v in subjectSW_vocab.iteritems():
		if i not in sublinbow_discarded_wordsSW:
			sublinbow_wordsSW.append(v)

	vocab_sublinbowSW=dict([(i,sublinbow_wordsSW[i]) for i in range(len(sublinbow_wordsSW))])

	tfidf_wordsSW = []
	for i, v in subjectSW_vocab.iteritems():
		if i not in tfidf_discarded_wordsSW:
			tfidf_wordsSW.append(v)

	vocab_tfidfSW=dict([(i,tfidf_wordsSW[i]) for i in range(len(tfidf_wordsSW))])

	numClasses = 3
	

	##########################################
	# LSA embeddings

	# BOW 
	lsa_bow_u = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_u'))
	lsa_bow_sigma = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_sigma'))
	lsa_bow_vt = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_vt'))
	lsa_bow_diag_sigma = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_diag_sigma'))
	lsa_bow_v = np.transpose(lsa_bow_vt)
	lsa_bow_sigma_vt = linalg.diagsvd(lsa_bow_sigma, len(lsa_bow_sigma), len(lsa_bow_vt))

	'''

	lsa_bow_u.shape
	(2154, 2154)


	lsa_bow_diag_sigma.shape
	(2154, 2502)

	lsa_bow_sigma.shape
	(2154,)


	lsa_bow_vt.shape
	(2502, 2502)

	# variation in index <50
	lsa_bow_sigma[:5]
	array([ 44.69389812,  26.80601562,  25.78234304,  23.13539066,  18.15379787])


	# variation in index > 50
	lsa_bow_sigma[50:60]
	array([ 6.55075234,  6.51878863,  6.44386869,  6.39970322,  6.34844877,
        6.31909528,  6.3161271 ,  6.25930152,  6.2256125 ,  6.19416863])


	'''
	wv_bow = lsa_bow_u[:,:50]
	dv_bow = lsa_bow_v[:,:50]


	lsa_bow_uSW = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_uSW'))
	lsa_bow_sigmaSW = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_sigmaSW'))
	lsa_bow_vtSW = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_vtSW'))
	lsa_bow_diag_sigmaSW = readPickle(os.path.join(VECTOR_PATH,'sampling/spamas_lsa_bow_diag_sigmaSW'))
	lsa_bow_vSW = np.transpose(lsa_bow_vtSW)
	lsa_bow_sigma_vtSW = linalg.diagsvd(lsa_bow_sigmaSW, len(lsa_bow_sigmaSW), len(lsa_bow_vtSW))

	wv_bowSW = lsa_bow_uSW[:,:50]
	dv_bowSW = lsa_bow_vSW[:,:50]

	# split into training and testing set

	xy = list(zip(yLabels,dv_bow))
	xy_vec = np.array(xy,dtype=object)
	# shuffling data
	ind_rand = np.arange(len(xy_vec))
	np.random.shuffle(ind_rand)
	vec_data = xy_vec[ind_rand]

	nTrain = int(.8 * len(vec_data))

	trainDat = vec_data[:nTrain]
	testDat = vec_data[nTrain:]

	x_train = list(trainDat[:,1])
	y_train = list(trainDat[:,0])

	x_test = list(testDat[:,1])
	y_test = list(testDat[:,0])

	# the following sklearn module will tranform nominal to numerical (0,1,2)
	numEncoder = LabelEncoder()
	numEncoder.fit(y_train)
	y_train_num = numEncoder.transform(y_train)
	# because our output is multiclass classification problems, 
	# we need to transform the class label into categorical encoding ([1,0,0],[0,1,0],[0,0,1])
	y_train_cat = to_categorical(y_train_num, num_classes=numClasses)

	savePickle(trainDat,'lsa_trainDat')
	savePickle(testDat,'lsa_testDat')

	# evaluate on MLP classifier
	mlp_BOW = mlpClassifier(dv_bow, NUM_CLASSES)
	mlp_BOW.fit(x_train, y_train_cat, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[history])
	score = mlp_BOW.evaluate(x_test, y_test, verbose=0)

	print('Test loss:', score[0])
	print('Test accuracy:', score[1])


	

	# SUBLINEAR BOW

	# TFIDF

	##########################################
	# LDA embeddings

	# BOW

	# SUBLINEAR BOW

	# TFIDF



	##########################################
	# WORD2VEC embeddings

	# SKIPGRAM

	# CBOW


	##########################################
	# DOC2VEC embeddings

	# DBOW

	# DM


	##########################################
	# LANGUAGE MODEL embeddings

	# language Model with LSTM 

	# language Model with Bidirectional LSTM 

	# language Model with GRU 

	# language Model with Bidirectional GRU 


	##########################################
	# CLASSIFIER MODEL embeddings

	# Classifier Model with LSTM 

	# Classifier Model with Bidirectional LSTM 

	# Classifier Model with GRU 

	# Classifier Model with Bidirectional GRU 

	
