# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "31.07.2017"
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




PATH = '/home/inimah/git/Neural-Language-Models/scripts/prepdata/spamassasin'

if __name__ == '__main__':


	

	# reading stored pre-processed (in pickle format)

	# vocabulary
	cont_vocab = readPickle(os.path.join(PATH,'spamas_contVocab'))
	cont_vocabTF = readPickle(os.path.join(PATH,'spamas_contVocabTF'))
	contSW_vocab = readPickle(os.path.join(PATH,'spamas_contSWVocab'))
	contSW_vocabTF = readPickle(os.path.join(PATH,'spamas_contSWVocabTF'))

	# mail content
	#content = readPickle(os.path.join(PATH, 'spamas_mergeTokenCont'))
	#contentSW = readPickle(os.path.join(PATH,'spamas_mergeTokenContSW'))
	content_sent = readPickle(os.path.join(PATH, 'spamas_tokenCont'))
	contentSW_sent = readPickle(os.path.join(PATH,'spamas_tokenContSW'))

	
	# discard words with frequency less than 5
	tmp1=[]  
	wordIndex = []
	for k,v in cont_vocabTF.iteritems():
		if v<3:
			tmp1.append((k,v))
		else:
			wordIndex.append(k)
	wordIndex.insert(0,'SOF')
	wordIndex.append('EOF')
	wordIndex.append('UNK')
	vocab=dict([(i,wordIndex[i]) for i in range(len(wordIndex))])

	# length of vocab after being discarded
	'''
	In [5]: len(vocab)
	Out[5]: 2814
	'''

	tmp2=[]  
	wordIndexSW = []
	for k,v in contSW_vocabTF.iteritems():
		if v<3:
			tmp2.append((k,v))
		else:
			wordIndexSW.append(k)
	wordIndexSW.insert(0,'SOF')
	wordIndexSW.append('EOF')
	wordIndexSW.append('UNK')
	vocabSW=dict([(i,wordIndexSW[i]) for i in range(len(wordIndexSW))])

	'''
	In [6]: len(vocabSW)
	Out[6]: 2467

	'''

	#savePickle(vocab,'spamas_reducedVocab')
	#savePickle(vocabSW,'spamas_reducedvocabSW')


	# discard less frequent words in tokenized documents based on new vocabulary list
	# and ignore/discard documents with length of sentences > 100 sentences
	# and discard array list of empty sentence and < 3 word tokens 
	doc_index = []
	sent_index = []
	new_contSent = dict()
	for i in content_sent:
		tmp_docs = []
		for j, arrSentences in enumerate(content_sent[i]):
			# discard documents with > 100 sentences
			if len(arrSentences) <= 100:
				tmp_sent = []
				for k, tokens in enumerate(arrSentences):
					if len(tokens) > 3:
						tmp_tokens = []
						for l, word in enumerate(tokens):
							if word in vocab.values():
								tmp_tokens.append(word)
						tmp_sent.append(tmp_tokens)
					else:
						sent_index.append((i,j,k))
				tmp_docs.append(tmp_sent)
			else:
				doc_index.append((i,j))

		new_contSent[i] = tmp_docs

	'''
In [4]: len(new_contSent['spam'])
Out[4]: 1975

In [5]: len(new_contSent['easy_ham'])
Out[5]: 6123
In [6]: len(new_contSent['hard_ham'])
Out[6]: 158

In [7]: 1975+6123+158
Out[7]: 8256

In [8]: len(doc_index)
Out[8]: 1072

In [9]: 8256+1072
Out[9]: 9328

In [10]: len(content_sent)
Out[10]: 3

In [11]: len(content_sent['spam'])
Out[11]: 2379

In [12]: len(content_sent['easy_ham'])
Out[12]: 6449

In [13]: len(content_sent['hard_ham'])
Out[13]: 500

In [14]: 2379+6449+500
Out[14]: 9328


	'''

	# Likewise for documents with stopword elimination and stemming
	doc_indexSW = []
	sent_indexSW = []
	new_contSentSW = dict()
	for i in contentSW_sent:
		tmp_docs = []
		for j, arrSentences in enumerate(contentSW_sent[i]):
			# discard documents with > 100 sentences
			if len(arrSentences) <= 100:
				tmp_sent = []
				for k, tokens in enumerate(arrSentences):
					if len(tokens) > 3:
						tmp_tokens = []
						for l, word in enumerate(tokens):
							if word in vocab.values():
								tmp_tokens.append(word)
						tmp_sent.append(tmp_tokens)
					else:
						sent_indexSW.append((i,j,k))
				tmp_docs.append(tmp_sent)
			else:
				doc_indexSW.append((i,j))

		new_contSentSW[i] = tmp_docs
	

	# provided fixed balanced training sets (with sampling)
	# sampling 500 documents from each class
	# except for class "hard-ham" since the number of documents is less than 500 (take all samples)

	sampling500_cont = dict()
	for i in new_contSent:
		tmp = []
		for j, sents in enumerate(new_contSent[i]):
			if (i != 'hard_ham' and j<500):
				tmp.append(sents)
			elif (i == 'hard_ham'):
				tmp.append(sents)
		sampling500_cont[i] = tmp

	sampling500_contSW = dict()
	for i in new_contSentSW:
		tmp = []
		for j, sents in enumerate(new_contSentSW[i]):
			if (i != 'hard_ham' and j<500):
				tmp.append(sents)
			elif (i == 'hard_ham'):
				tmp.append(sents)
		sampling500_contSW[i] = tmp

	savePickle(sampling500_cont,'spamas_sampling500_cont')
	savePickle(sampling500_contSW,'spamas_sampling500_contSW')

	# provided fixed balanced training sets (with sampling)
	# sampling 1000 documents from each class
	# except for class "hard-ham" since the number of documents is less than 1000 (take all samples)

	sampling1000_cont = dict()
	for i in new_contSent:
		tmp = []
		for j, sents in enumerate(new_contSent[i]):
			if (i != 'hard_ham' and j<1000):
				tmp.append(sents)
			elif (i == 'hard_ham'):
				tmp.append(sents)
		sampling1000_cont[i] = tmp

	sampling1000_contSW = dict()
	for i in new_contSentSW:
		tmp = []
		for j, sents in enumerate(new_contSentSW[i]):
			if (i != 'hard_ham' and j<1000):
				tmp.append(sents)
			elif (i == 'hard_ham'):
				tmp.append(sents)
		sampling1000_contSW[i] = tmp

	savePickle(sampling1000_cont,'spamas_sampling1000_cont')
	savePickle(sampling1000_contSW,'spamas_sampling1000_contSW')

	# provided fixed training sets but according to the proportion for each class --> resulting unbalanced class for total sets
	# to reduce computation load
	# sampling 50% total number documents from each class
	# except for class "hard-ham" (take all samples)

	sampling_unbalanced_cont = dict()
	for i in new_contSent:
		tmp = []
		num_docs = .5 * (len(new_contSent[i]))
		for j, tokens in enumerate(new_contSent[i]):
			if (i != 'hard_ham' and j<=num_docs):
				tmp.append(tokens)
			elif (i == 'hard_ham'):
				tmp.append(tokens)
		sampling_unbalanced_cont[i] = tmp

	sampling_unbalanced_contSW = dict()
	for i in new_contSentSW:
		tmp = []
		num_docs = .5 * (len(new_contSentSW[i]))
		for j, tokens in enumerate(new_contSentSW[i]):
			if (i != 'hard_ham' and j<=num_docs):
				tmp.append(tokens)
			elif (i == 'hard_ham'):
				tmp.append(tokens)
		sampling_unbalanced_contSW[i] = tmp

	savePickle(sampling_unbalanced_cont,'spamas_sampling_unbalanced_cont')
	savePickle(sampling_unbalanced_contSW,'spamas_sampling_unbalanced_contSW')

	'''


	# check statistics of sentences (min, average, max number of sentences per document)
	count_sent = []
	for i in sampling_cont:
		for j,arrSentences in enumerate(sampling_cont[i]):
			count_words = 0
			avg_words = 0
			num_sent = len(arrSentences)
			for k, words in enumerate(arrSentences):
				count_words += len(words)
			avg_words = count_words/num_sent
			count_sent.append((i,j,num_sent,avg_words))

	count_sentSW = []
	for i in sampling_contSW:
		for j,arrSentences in enumerate(sampling_contSW[i]):
			count_words = 0
			avg_words = 0
			num_sent = len(arrSentences)
			for k, words in enumerate(arrSentences):
				count_words += len(words)
			avg_words = count_words/num_sent
			count_sentSW.append((i,j,num_sent,avg_words))

	savePickle(count_sent,'spamas_count_sent')
	savePickle(count_sentSW,'spamas_count_sentSW')

	

	x1 = []
	w1 = []
    x2 = []
    w2 = []
    x3 = []
    w3 = []
    for i,data in enumerate(count_sent):
    	if data[0] == 'spam':
    		x1.append(data[2])
    		w1.append(data[3])
        elif data[0] == 'easy_ham':
        	x2.append(data[2])
            w2.append(data[3])
        elif data[0] == 'hard_ham':
        	x3.append(data[2])
            w3.append(data[3])

   
   	ax1 = sns.distplot(x1)
	fig_x1 = ax1.get_figure()
	fig_x1.savefig('spam_sent_num.png')
	fig_x1.clf()
	ax2 = sns.distplot(x2)
	fig_x2 = ax2.get_figure()
	fig_x2.savefig('easyham_sent_num.png')
	fig_x2.clf()
	ax3 = sns.distplot(x3)
	fig_x3 = ax3.get_figure()
	fig_x3.savefig('hardham_sent_num.png')
	fig_x3.clf()
	aw1 = sns.distplot(w1)
	fig_w1 = aw1.get_figure()
	fig_w1.savefig('spam_avg_words_per_sent.png')
	fig_w1.clf()
	aw2 = sns.distplot(w2)
	fig_w2 = aw2.get_figure()
	fig_w2.savefig('easyham_avg_words_per_sent.png')
	fig_w2.clf()
	aw3 = sns.distplot(w3)
	fig_w3 = aw3.get_figure()
	fig_w3.savefig('hardham_avg_words_per_sent.png')
	fig_w3.clf()


    '''

    ############################
    # For 500 sampling data
    ############################

    # For tokenized document without Stopword elimination and stemming
	# merge all array and class labels
	labelled_cont500 = []
	for i in sampling500_cont:
		for j, tokens in enumerate(sampling500_cont[i]):
			labelled_cont500.append((i,tokens))

	
	tokenized_docs500 = []
	class_labels500 = []
	for i, data in enumerate(labelled_cont500):
		class_labels500.append(data[0])
		tokenized_docs500.append(data[1])

	savePickle(labelled_cont500,'spamas_labelled_cont500')
	savePickle(class_labels500,'spamas_class_labels500')
	savePickle(tokenized_docs500,'spamas_tokenized_docs500')



	# For tokenized document with Stopword elimination and stemming
	labelled_cont500SW = []
	for i in sampling500_contSW:
		for j, tokens in enumerate(sampling500_contSW[i]):
			labelled_cont500SW.append((i,tokens))

	tokenized_docs500SW = []
	class_labels500SW = []
	for i, data in enumerate(labelled_cont500SW):
		class_labels500SW.append(data[0])
		tokenized_docs500SW.append(data[1])

	savePickle(labelled_cont500SW,'spamas_labelled_cont500SW')
	savePickle(class_labels500SW,'spamas_class_labels500SW')
	savePickle(tokenized_docs500SW,'spamas_tokenized_docs500SW')

	############################
    # For 1000 sampling data
    ############################


    # For tokenized document without Stopword elimination and stemming
	# merge all array and class labels
	labelled_cont1000 = []
	for i in sampling1000_cont:
		for j, tokens in enumerate(sampling1000_cont[i]):
			labelled_cont1000.append((i,tokens))

	
	tokenized_docs1000 = []
	class_labels1000 = []
	for i, data in enumerate(labelled_cont1000):
		class_labels1000.append(data[0])
		tokenized_docs1000.append(data[1])

	savePickle(labelled_cont1000,'spamas_labelled_cont1000')
	savePickle(class_labels1000,'spamas_class_labels1000')
	savePickle(tokenized_docs1000,'spamas_tokenized_docs1000')



	# For tokenized document with Stopword elimination and stemming
	labelled_cont1000SW = []
	for i in sampling1000_contSW:
		for j, tokens in enumerate(sampling1000_contSW[i]):
			labelled_cont1000SW.append((i,tokens))

	tokenized_docs1000SW = []
	class_labels1000SW = []
	for i, data in enumerate(labelled_cont1000SW):
		class_labels1000SW.append(data[0])
		tokenized_docs1000SW.append(data[1])

	savePickle(labelled_cont1000SW,'spamas_labelled_cont1000SW')
	savePickle(class_labels1000SW,'spamas_class_labels1000SW')
	savePickle(tokenized_docs1000SW,'spamas_tokenized_docs1000SW')



    ########################################################
    # For unbalanced sampling data (50% class member)
    ########################################################


    # For tokenized document without Stopword elimination and stemming
	# merge all array and class labels
	labelled_unbalanced_cont = []
	for i in sampling_unbalanced_cont:
		for j, tokens in enumerate(sampling_unbalanced_cont[i]):
			labelled_unbalanced_cont.append((i,tokens))

	
	tokenized_unbalanced_docs = []
	class_unbalanced_labels = []
	for i, data in enumerate(labelled_unbalanced_cont):
		class_unbalanced_labels.append(data[0])
		tokenized_unbalanced_docs.append(data[1])

	savePickle(labelled_unbalanced_cont,'spamas_labelled_unbalanced_cont')
	savePickle(class_unbalanced_labels,'spamas_class_unbalanced_labels')
	savePickle(tokenized_unbalanced_docs,'spamas_tokenized_unbalanced_docs')



	# For tokenized document with Stopword elimination and stemming
	labelled_unbalanced_contSW = []
	for i in sampling_unbalanced_contSW:
		for j, tokens in enumerate(sampling_unbalanced_contSW[i]):
			labelled_unbalanced_contSW.append((i,tokens))

	tokenized_unbalanced_docsSW = []
	class_unbalanced_labelsSW = []
	for i, data in enumerate(labelled_unbalanced_contSW):
		class_unbalanced_labelsSW.append(data[0])
		tokenized_unbalanced_docsSW.append(data[1])

	savePickle(labelled_unbalanced_contSW,'spamas_labelled_unbalanced_contSW')
	savePickle(class_unbalanced_labelsSW,'spamas_class_unbalanced_labelsSW')
	savePickle(tokenized_unbalanced_docsSW,'spamas_tokenized_unbalanced_docsSW')


	########################################################
	# Matrix decomposition
	# and generating term-document matrix
	########################################################


	############################
    # For 500 sampling data
    ############################


	vs500 = VectorSpace(tokenized_docs500,vocab)

	# save matrices: BOW, sub-linear BOW, 
	savePickle(vs500.td_bow,'spamas_td_bow500')
	savePickle(vs500.td_bow_sublin,'spamas_td_bow_sublin500')


	tfidf500 = TFIDF(vs500.td_bow)
	td_tfidf500 = tfidf500.transform()

	# save tf-idf matrix transformed from BOW 
	savePickle(td_tfidf500,'spamas_td_tfidf500')

	############################
	# matrix decomposition of BOW

	# transpose input matrix (originally doc x term dimension -> to term x doc dimension)
	bow_t = np.transpose(vs.td_bow)

	# eliminate non-occur words in document corpus (due to sampling)
	bow_null_ind = np.where(~bow_t.any(axis=1))[0]
	savePickle(bow_null_ind,'spamas_bow_null_ind')

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
	savePickle(bow_sublin_null_ind,'spamas_bow_sublin_null_ind')
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




	