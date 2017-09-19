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
from sklearn.preprocessing import LabelEncoder
# the following modules are in lsa folder
from lsa.vector_space import VectorSpace
from lsa.tfidf import TFIDF
from lsa.tokenizer import Tokenizer
import lda




#PATH = '/home/inimah/git/Neural-Language-Models/scripts/prepdata/spamassasin'
PATH = '/home/inimah/git/Neural-Language-Models/scripts/train_spamas/subj/lda/'



if __name__ == '__main__':

	

	# reading stored pre-processed (in pickle format)

	# Final vocabulary list after being reduced from less frequent words (links, noises)
	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))
	subjectSW_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocabSW'))

	# Final tokenized documents with labels 
	subject = readPickle(os.path.join(PATH,'spamas_fin_labelled_subj'))
	subjectSW = readPickle(os.path.join(PATH,'spamas_fin_labelled_subjSW'))

	

	
	tokenized_docs = []
	class_labels = []
	for i, data in enumerate(subject):
		class_labels.append(data[0])
		tokenized_docs.append(data[1])

	tokenized_docsSW = []
	class_labelsSW = []
	for i, data in enumerate(subjectSW):
		class_labelsSW.append(data[0])
		tokenized_docsSW.append(data[1])
	


	########################################################
	# For text without stopword eliminating and stemming
	########################################################

	
	vs = VectorSpace(tokenized_docs,subject_vocab)

	td_bow = np.array(vs.td_bow, dtype='int64')
	td_sublinbow = np.array(vs.td_bow_sublin, dtype='int64')


	# save matrices: BOW, sub-linear BOW, 
	savePickle(td_bow,'spamas_lda_td_bow')
	savePickle(td_sublinbow,'spamas_lda_td_bow_sublin')
	savePickle(vs.td_bow_info,'spamas_lda_td_bow_info')
	savePickle(vs.td_bow_sublin_info,'spamas_lda_td_bow_sublin_info')


	tfidf = TFIDF(vs.td_bow)
	td_tfidf = tfidf.transform()
	td_tfidf_arr = np.array(td_tfidf, dtype='int64')

	# save tf-idf matrix transformed from BOW 
	savePickle(td_tfidf_arr,'spamas_lda_td_tfidf')
	savePickle(vs.td_tfidf_info,'spamas_lda_td_tfidf_info')

	############################
	# LDA on BOW matrix

	# default iteration = 1000
	model_bow = lda.LDA(n_topics=5, n_iter=1000, random_state=1)
	model_bow.fit(td_bow)
	topic_word_bow = model_bow.topic_word_
	topic_doc_bow = model_bow.doc_topic_



	savePickle(model_bow,'spamas_lda_bow')
	savePickle(topic_word_bow,'spamas_topic_word_bow')
	savePickle(topic_doc_bow,'spamas_topic_doc_bow')



	############################
	# LDA on Sublinear-BOW matrix

	# default iteration = 1000
	model_sublinbow = lda.LDA(n_topics=5, n_iter=1000, random_state=1)
	model_sublinbow.fit(td_sublinbow)
	topic_word_sublinbow = model_sublinbow.topic_word_
	topic_doc_sublinbow = model_sublinbow.doc_topic_

	

	savePickle(model_sublinbow,'spamas_lda_sublinbow')
	savePickle(topic_word_sublinbow,'spamas_topic_word_sublinbow')
	savePickle(topic_doc_sublinbow,'spamas_topic_doc_sublinbow')

	############################
	# LDA on TFIDF matrix

	# default iteration = 1000
	model_tfidf = lda.LDA(n_topics=5, n_iter=1000, random_state=1)
	model_tfidf.fit(td_tfidf_arr)
	topic_word_tfidf = model_tfidf.topic_word_
	topic_doc_tfidf = model_tfidf.doc_topic_

	savePickle(model_tfidf,'spamas_lda_tfidf')
	savePickle(topic_word_tfidf,'spamas_topic_word_tfidf')
	savePickle(topic_doc_tfidf,'spamas_topic_doc_tfidf')


	########################################################
	# For text with stopword eliminating and stemming
	########################################################


	vsSW = VectorSpace(tokenized_docsSW,subjectSW_vocab)

	td_bowSW = np.array(vsSW.td_bow, dtype='int64')
	td_sublinbowSW = np.array(vsSW.td_bow_sublin, dtype='int64')

	# save matrices: BOW, sub-linear BOW, 
	savePickle(td_bowSW,'spamas_lda_td_bowSW')
	savePickle(td_sublinbowSW,'spamas_lda_td_bow_sublinSW')
	savePickle(vsSW.td_bow_info,'spamas_lda_td_bowSW_info')
	savePickle(vsSW.td_bow_sublin_info,'spamas_lda_td_bow_sublinSW_info')


	tfidfSW = TFIDF(vsSW.td_bow)
	td_tfidfSW = tfidfSW.transform()
	td_tfidfSW_arr = np.array(td_tfidfSW, dtype='int64')

	# save tf-idf matrix transformed from BOW 
	savePickle(td_tfidfSW_arr,'spamas_lda_td_tfidfSW')
	savePickle(vsSW.td_tfidf_info,'spamas_lda_td_tfidfSW_info')

	############################
	# LDA on BOW matrix

	# default iteration = 1000
	model_bowSW = lda.LDA(n_topics=5, n_iter=1000, random_state=1)
	model_bowSW.fit(td_bowSW)
	topic_word_bowSW = model_bowSW.topic_word_
	topic_doc_bowSW = model_bowSW.doc_topic_



	savePickle(model_bowSW,'spamas_lda_bowSW')
	savePickle(topic_word_bowSW,'spamas_topic_word_bowSW')
	savePickle(topic_doc_bowSW,'spamas_topic_doc_bowSW')



	############################
	# LDA on Sublinear-BOW matrix

	# default iteration = 1000
	model_sublinbowSW = lda.LDA(n_topics=5, n_iter=1000, random_state=1)
	model_sublinbowSW.fit(td_sublinbowSW)
	topic_word_sublinbowSW = model_sublinbowSW.topic_word_
	topic_doc_sublinbowSW = model_sublinbowSW.doc_topic_

	

	savePickle(model_sublinbowSW,'spamas_lda_sublinbowSW')
	savePickle(topic_word_sublinbowSW,'spamas_topic_word_sublinbowSW')
	savePickle(topic_doc_sublinbowSW,'spamas_topic_doc_sublinbowSW')

	############################
	# LDA on TFIDF matrix

	# default iteration = 1000
	model_tfidfSW = lda.LDA(n_topics=5, n_iter=1000, random_state=1)
	model_tfidfSW.fit(td_tfidfSW_arr)
	topic_word_tfidfSW = model_tfidfSW.topic_word_
	topic_doc_tfidfSW = model_tfidfSW.doc_topic_

	savePickle(model_tfidfSW,'spamas_lda_tfidfSW')
	savePickle(topic_word_tfidfSW,'spamas_topic_word_tfidfSW')
	savePickle(topic_doc_tfidfSW,'spamas_topic_doc_tfidfSW')

	
