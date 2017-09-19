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

	'''

	# reading stored pre-processed (in pickle format)

	# Vocabulary
	subject_vocab = readPickle(os.path.join(PATH,'spamas_subjVocab'))
	subject_vocabTF = readPickle(os.path.join(PATH,'spamas_subjVocabTF'))
	subjectSW_vocab = readPickle(os.path.join(PATH,'spamas_subjSWVocab'))
	subjectSW_vocabTF = readPickle(os.path.join(PATH,'spamas_subjSWVocabTF'))

	# Tokenized documents
	subject = readPickle(os.path.join(PATH,'spamas_tokenSubj'))
	subjectSW = readPickle(os.path.join(PATH,'spamas_tokenSubjSW'))

	# discard words with frequency less than 5
	tmp1=[]  
	wordIndex = []
	for k,v in subject_vocabTF.iteritems():
		if v<3:
			tmp1.append((k,v))
		else:
			wordIndex.append(k)
	wordIndex.insert(0,'SOF')
	wordIndex.append('EOF')
	wordIndex.append('UNK')
	vocab=dict([(i,wordIndex[i]) for i in range(len(wordIndex))])

	# length of vocab after being discarded
	
	#In [5]: len(vocab)
	#Out[5]: 2814
	

	tmp2=[]  
	wordIndexSW = []
	for k,v in subjectSW_vocabTF.iteritems():
		if v<3:
			tmp2.append((k,v))
		else:
			wordIndexSW.append(k)
	wordIndexSW.insert(0,'SOF')
	wordIndexSW.append('EOF')
	wordIndexSW.append('UNK')
	vocabSW=dict([(i,wordIndexSW[i]) for i in range(len(wordIndexSW))])

	
	#In [6]: len(vocabSW)
	#Out[6]: 2467


	savePickle(vocab,'spamas_reducedVocab')
	savePickle(vocabSW,'spamas_reducedvocabSW')


	# also discard less frequent words in tokenized documents based on new vocabulary list
	new_subject = dict()
	for i in subject:
		tmp_docs = []
		for j, tokens in enumerate(subject[i]):
			tmp_tokens = []
			for k, word in enumerate(tokens):
				if word in vocab.values():
					tmp_tokens.append(word)
			tmp_docs.append(tmp_tokens)
		new_subject[i] = tmp_docs

	new_subjectSW = dict()
	for i in subject:
		tmp_docs = []
		for j, tokens in enumerate(subject[i]):
			tmp_tokens = []
			for k, word in enumerate(tokens):
				if word in vocab.values():
					tmp_tokens.append(word)
			tmp_docs.append(tmp_tokens)
		new_subjectSW[i] = tmp_docs


	savePickle(new_subject,'spamas_reducedTokenSubj')
	savePickle(new_subjectSW,'spamas_reducedTokenSubjSW')

	
	#In [10]: len(new_subject['hard_ham'])
	#Out[10]: 500
	#In [11]: len(new_subject['easy_ham'])
	#Out[11]: 6449
	#In [12]: len(new_subject['spam'])
	#Out[12]: 2379

	#Total docs = 9328



	# sampling 500 documents from each class
	sampling_subj = dict()
	for i in new_subject:
		tmp = []
		for j, tokens in enumerate(new_subject[i]):
			if j<500:
				tmp.append(tokens)
		sampling_subj[i] = tmp

	sampling_subjSW = dict()
	for i in new_subjectSW:
		tmp = []
		for j, tokens in enumerate(new_subjectSW[i]):
			if j<500:
				tmp.append(tokens)
		sampling_subjSW[i] = tmp

	savePickle(sampling_subj,'spamas_sampling_subj')
	savePickle(sampling_subjSW,'spamas_sampling_subjSW')

	
	# For tokenized document without Stopword elimination and stemming
	# merge all array and class labels
	labelled_subj = []
	for i in sampling_subj:
		for j, tokens in enumerate(sampling_subj[i]):
			labelled_subj.append((i,tokens))

	
	#In [17]: len(labelled_subj)
	#Out[17]: 9328

	'''

	labelled_subj = readPickle(os.path.join(PATH,'spamas_labelled_subj'))
	
	tokenized_docs = []
	class_labels = []
	for i, data in enumerate(labelled_subj):
		class_labels.append(data[0])
		tokenized_docs.append(data[1])

	#savePickle(labelled_subj,'spamas_labelled_subj')
	savePickle(tokenized_docs,'spamas_lda_tokenized_docs')
	savePickle(class_labels,'spamas_lda_class_labels')

	'''

	# For tokenized document with Stopword elimination and stemming
	labelled_subjSW = []
	for i in sampling_subjSW:
		for j, tokens in enumerate(sampling_subjSW[i]):
			labelled_subjSW.append((i,tokens))

	'''

	labelled_subjSW = readPickle(os.path.join(PATH,'spamas_labelled_subjSW'))

	tokenized_docsSW = []
	class_labelsSW = []
	for i, data in enumerate(labelled_subjSW):
		class_labelsSW.append(data[0])
		tokenized_docsSW.append(data[1])

	savePickle(tokenized_docs,'spamas_lda_tokenized_docsSW')
	savePickle(class_labels,'spamas_lda_class_labelsSW')

	vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))
	vocabSW = readPickle(os.path.join(PATH,'spamas_reducedvocabSW'))

	#savePickle(labelled_subjSW,'spamas_labelled_subjSW')


	########################################################
	# For text without stopword eliminating and stemming
	########################################################

	
	vs = VectorSpace(tokenized_docs,vocab)

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
	model_tfidf = lda.LDA(n_topics=100, n_iter=1000, random_state=1)
	model_tfidf.fit(td_tfidf_arr)
	topic_word_tfidf = model_tfidf.topic_word_
	topic_doc_tfidf = model_tfidf.doc_topic_

	savePickle(model_tfidf,'spamas_lda_tfidf')
	savePickle(topic_word_tfidf,'spamas_topic_word_tfidf')
	savePickle(topic_doc_tfidf,'spamas_topic_doc_tfidf')


	########################################################
	# For text with stopword eliminating and stemming
	########################################################


	vsSW = VectorSpace(tokenized_docsSW,vocabSW)

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

	
