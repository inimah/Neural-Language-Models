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



if __name__ == '__main__':



	# reading stored pre-processed (in pickle format)

	# Final vocabulary list after being reduced from less frequent words (links, noises)
	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))

	# Final tokenized documents with labels 
	subject = readPickle(os.path.join(PATH,'spamas_fin_labelled_subj'))

	tokenized_docs = []
	class_labels = []
	for i, data in enumerate(subject):
		class_labels.append(data[0])
		tokenized_docs.append(data[1])


	# word2vec model of mail subjects
	# word dimension = 50
	w2v_subj_1, w2v_subj_2, w2v_subj_embed1, w2v_subj_embed2, w2v_vocab1, w2v_vocab2, w2v_weights1, w2v_weights2 = wordEmbedding(tokenized_docs, subject_vocab, 50, 50)
	# Skipgram
	w2v_subj_1.save('w2v_subj_1')
	# CBOW
	w2v_subj_2.save('w2v_subj_2')

	savePickle(w2v_subj_embed1,'w2v_subj_embed1')
	savePickle(w2v_subj_embed2,'w2v_subj_embed2')

	savePickle(w2v_vocab1,'w2v_vocab1')
	savePickle(w2v_vocab2,'w2v_vocab2')

	savePickle(w2v_weights1,'w2v_weights1')
	savePickle(w2v_weights2,'w2v_weights2')


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




