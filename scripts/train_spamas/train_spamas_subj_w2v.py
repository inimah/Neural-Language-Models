# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "15.09.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
sys.path.insert(0,'..')
import numpy as np
from text_preprocessing import *
from language_models import *
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


# Update the code and re-run due to the final-reduced version of preprocessed data


PATH = '/home/inimah/git/Neural-Language-Models/scripts/prepdata/spamassasin'

def _encodeLabelledText(tokenized_docs, vocab):

	# encode tokenized of words in document into its index/numerical value in vocabulary list
	# the input is in array list tokenized documents

	encoded_docs = []

	for i, data in enumerate(tokenized_docs):
		encoded_docs.append((data[0],wordsToIndex(vocab,data[1])))

	return encoded_docs


if __name__ == '__main__':



	# reading stored pre-processed (in pickle format)

	# Vocabulary

	'''
	subjectSW_vocab = readPickle(os.path.join(PATH,'spamas_reducedvocabSW'))

	# Tokenized documents
	subject = readPickle(os.path.join(PATH,'spamas_reducedTokenSubj'))
	subjectSW = readPickle(os.path.join(PATH,'spamas_reducedTokenSubjSW'))

	# Tokenized documents from sampling (500 docs per class)
	subject500 = readPickle(os.path.join(PATH,'spamas_sampling_subj'))
	subjectSW500 = readPickle(os.path.join(PATH,'spamas_sampling_subjSW'))


	########################################################
	# Without sampling

	labelled_subj = []
	for i in subject:
		for j, tokens in enumerate(subject[i]):
			labelled_subj.append((i,tokens))

	
	tokenized_docs = []
	class_labels = []
	for i, data in enumerate(labelled_subj):
		class_labels.append(data[0])
		tokenized_docs.append(data[1])

	savePickle(labelled_subj,'spamas_w2v_labelled_subj')
	savePickle(tokenized_docs,'spamas_w2v_tokenized_docs')
	savePickle(class_labels,'spamas_w2v_class_labels')



	# For tokenized document with Stopword elimination and stemming
	labelled_subjSW = []
	for i in subjectSW:
		for j, tokens in enumerate(subjectSW[i]):
			labelled_subjSW.append((i,tokens))

	tokenized_docsSW = []
	class_labelsSW = []
	for i, data in enumerate(labelled_subjSW):
		class_labelsSW.append(data[0])
		tokenized_docsSW.append(data[1])

	savePickle(labelled_subjSW,'spamas_w2v_labelled_subjSW')
	savePickle(tokenized_docsSW,'spamas_w2v_tokenized_docsSW')
	savePickle(class_labelsSW,'spamas_w2v_class_labelsSW')

	########################################################
	# With sampling

	labelled_subj500 = []
	for i in subject500:
		for j, tokens in enumerate(subject500[i]):
			labelled_subj500.append((i,tokens))

	
	tokenized_docs500 = []
	class_labels500 = []
	for i, data in enumerate(labelled_subj500):
		class_labels500.append(data[0])
		tokenized_docs500.append(data[1])

	savePickle(labelled_subj500,'spamas_w2v_labelled_subj500')
	savePickle(tokenized_docs500,'spamas_w2v_tokenized_docs500')
	savePickle(class_labels500,'spamas_w2v_class_labels500')



	# For tokenized document with Stopword elimination and stemming
	labelled_subjSW500 = []
	for i in subjectSW500:
		for j, tokens in enumerate(subjectSW500[i]):
			labelled_subjSW500.append((i,tokens))

	tokenized_docsSW500 = []
	class_labelsSW500 = []
	for i, data in enumerate(labelled_subjSW500):
		class_labelsSW500.append(data[0])
		tokenized_docsSW500.append(data[1])

	savePickle(labelled_subjSW500,'spamas_w2v_labelled_subjSW500')
	savePickle(tokenized_docsSW500,'spamas_w2v_tokenized_docsSW500')
	savePickle(class_labelsSW500,'spamas_w2v_class_labelsSW500')


	########################################################
	# Without sampling


	# without stopword removing & stemming

	# word2vec model of mail subjects
	# word dimension = 50
	w2v_subj_1, w2v_subj_2, w2v_subj_embed1, w2v_subj_embed2 = wordEmbedding(tokenized_docs, subject_vocab, 50, 50)
	# Skipgram
	w2v_subj_1.save('w2v_subj_1')
	# CBOW
	w2v_subj_2.save('w2v_subj_2')

	savePickle(w2v_subj_embed1,'w2v_subj_embed1')
	savePickle(w2v_subj_embed2,'w2v_subj_embed2')


	# create document representation of word vectors

	# By averaging word vectors
	avg_subj_embed1 = averageWE(w2v_subj_embed1, subject_vocab, tokenized_docs)
	avg_subj_embed2 = averageWE(w2v_subj_embed2, subject_vocab, tokenized_docs)

	savePickle(avg_subj_embed1,'avg_subj_embed1')
	savePickle(avg_subj_embed2,'avg_subj_embed2')


	# By averaging and idf weights of word vectors
	avgIDF_subj_embed1 = averageIdfWE(w2v_subj_embed1, subject_vocab, tokenized_docs)
	avgIDF_subj_embed2 = averageIdfWE(w2v_subj_embed2, subject_vocab, tokenized_docs)

	savePickle(avgIDF_subj_embed1,'avgIDF_subj_embed1')
	savePickle(avgIDF_subj_embed2,'avgIDF_subj_embed2')


	# with stopword removing & stemming

	# word2vec model of mail subjects
	# word dimension = 50
	w2v_subj_1SW, w2v_subj_2SW, w2v_subj_embed1SW, w2v_subj_embed2SW = wordEmbedding(tokenized_docsSW, subjectSW_vocab, 50, 50)
	# Skipgram
	w2v_subj_1SW.save('w2v_subj_1SW')
	# CBOW
	w2v_subj_2SW.save('w2v_subj_2SW')

	savePickle(w2v_subj_embed1SW,'w2v_subj_embed1SW')
	savePickle(w2v_subj_embed2SW,'w2v_subj_embed2SW')


	# create document representation of word vectors

	# By averaging word vectors
	avg_subj_embed1SW = averageWE(w2v_subj_embed1SW, subjectSW_vocab, tokenized_docsSW)
	avg_subj_embed2SW = averageWE(w2v_subj_embed2SW, subjectSW_vocab, tokenized_docsSW)

	savePickle(avg_subj_embed1SW,'avg_subj_embed1SW')
	savePickle(avg_subj_embed2SW,'avg_subj_embed2SW')


	# By averaging and idf weights of word vectors
	avgIDF_subj_embed1SW = averageIdfWE(w2v_subj_embed1SW, subjectSW_vocab, tokenized_docsSW)
	avgIDF_subj_embed2SW = averageIdfWE(w2v_subj_embed2SW, subjectSW_vocab, tokenized_docsSW)

	savePickle(avgIDF_subj_embed1SW,'avgIDF_subj_embed1SW')
	savePickle(avgIDF_subj_embed2SW,'avgIDF_subj_embed2SW')


	########################################################
	# With sampling


	# without stopword removing & stemming

	# word2vec model of mail subjects
	# word dimension = 50
	w2v_subj_1_500, w2v_subj_2_500, w2v_subj_embed1_500, w2v_subj_embed2_500 = wordEmbedding(tokenized_docs500, subject_vocab, 50, 50)
	# Skipgram
	w2v_subj_1_500.save('w2v_subj_1_500')
	# CBOW
	w2v_subj_2_500.save('w2v_subj_2_500')

	savePickle(w2v_subj_embed1_500,'w2v_subj_embed1_500')
	savePickle(w2v_subj_embed2_500,'w2v_subj_embed2_500')


	# create document representation of word vectors

	# By averaging word vectors
	avg_subj_embed1_500 = averageWE(w2v_subj_embed1_500, subject_vocab, tokenized_docs500)
	avg_subj_embed2_500 = averageWE(w2v_subj_embed2_500, subject_vocab, tokenized_docs500)

	savePickle(avg_subj_embed1_500,'avg_subj_embed1_500')
	savePickle(avg_subj_embed2_500,'avg_subj_embed2_500')


	# By averaging and idf weights of word vectors
	avgIDF_subj_embed1_500 = averageIdfWE(w2v_subj_embed1_500, subject_vocab, tokenized_docs500)
	avgIDF_subj_embed2_500 = averageIdfWE(w2v_subj_embed2_500, subject_vocab, tokenized_docs500)

	savePickle(avgIDF_subj_embed1_500,'avgIDF_subj_embed1_500')
	savePickle(avgIDF_subj_embed2_500,'avgIDF_subj_embed2_500')


	# with stopword removing & stemming

	# word2vec model of mail subjects
	# word dimension = 50
	w2v_subj_1SW_500, w2v_subj_2SW_500, w2v_subj_embed1SW_500, w2v_subj_embed2SW_500 = wordEmbedding(tokenized_docsSW500, subjectSW_vocab, 50, 50)
	# Skipgram
	w2v_subj_1SW_500.save('w2v_subj_1SW_500')
	# CBOW
	w2v_subj_2SW_500.save('w2v_subj_2SW_500')

	savePickle(w2v_subj_embed1SW_500,'w2v_subj_embed1SW_500')
	savePickle(w2v_subj_embed2SW_500,'w2v_subj_embed2SW_500')


	# create document representation of word vectors

	# By averaging word vectors
	avg_subj_embed1SW_500 = averageWE(w2v_subj_embed1SW_500, subjectSW_vocab, tokenized_docsSW500)
	avg_subj_embed2SW_500 = averageWE(w2v_subj_embed2SW_500, subjectSW_vocab, tokenized_docsSW500)

	savePickle(avg_subj_embed1SW_500,'avg_subj_embed1SW_500')
	savePickle(avg_subj_embed2SW_500,'avg_subj_embed2SW_500')


	# By averaging and idf weights of word vectors
	avgIDF_subj_embed1SW_500 = averageIdfWE(w2v_subj_embed1SW_500, subjectSW_vocab, tokenized_docsSW500)
	avgIDF_subj_embed2SW_500 = averageIdfWE(w2v_subj_embed2SW_500, subjectSW_vocab, tokenized_docsSW500)

	savePickle(avgIDF_subj_embed1SW_500,'avgIDF_subj_embed1SW_500')
	savePickle(avgIDF_subj_embed2SW_500,'avgIDF_subj_embed2SW_500')

	'''

	# Final vocabulary list after being reduced from less frequent words (links, noises)
	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))
	subjectSW_vocab = readPickle(os.path.join(PATH,'spamas_reducedvocabSW'))

	# Final tokenized documents with maximum word length = 25 words
	# with labels 
	subject = readPickle(os.path.join(PATH,'spamas_fin_labelled_subj'))
	tokenized_docs = []
	class_labels = []
	for i, data in enumerate(subject):
		class_labels.append(data[0])
		tokenized_docs.append(data[1])
	
	# for tokenized subject title with stopword removing and stemming, have not been preprocessed yet
	# as such it is run here
	# this is not being used for word2vecmodel
	# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

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

	# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx



	# word2vec model of mail subjects
	# word dimension = 50
	w2v_subj_1, w2v_subj_2, w2v_subj_embed1, w2v_subj_embed2 = wordEmbedding(tokenized_docs, subject_vocab, 50, 50)
	# Skipgram
	w2v_subj_1.save('w2v_subj_1')
	# CBOW
	w2v_subj_2.save('w2v_subj_2')

	savePickle(w2v_subj_embed1,'w2v_subj_embed1')
	savePickle(w2v_subj_embed2,'w2v_subj_embed2')


	# create document representation of word vectors

	# By averaging word vectors
	avg_subj_embed1 = averageWE(w2v_subj_embed1, subject_vocab, tokenized_docs)
	avg_subj_embed2 = averageWE(w2v_subj_embed2, subject_vocab, tokenized_docs)

	savePickle(avg_subj_embed1,'avg_subj_embed1')
	savePickle(avg_subj_embed2,'avg_subj_embed2')


	# By averaging and idf weights of word vectors
	avgIDF_subj_embed1 = averageIdfWE(w2v_subj_embed1, subject_vocab, tokenized_docs)
	avgIDF_subj_embed2 = averageIdfWE(w2v_subj_embed2, subject_vocab, tokenized_docs)

	savePickle(avgIDF_subj_embed1,'avgIDF_subj_embed1')
	savePickle(avgIDF_subj_embed2,'avgIDF_subj_embed2')




