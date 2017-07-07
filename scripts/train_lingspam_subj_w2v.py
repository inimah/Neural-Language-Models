# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "14.06.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import numpy as np
from text_preprocessing import *
from language_models import *
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-batch_size', type=int, default=100)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
args = vars(ap.parse_args())


BATCH_SIZE = args['batch_size']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']


PATH = 'prepdata/lingspam'
EMBED_PATH = 'embedding/lingspam'


if __name__ == '__main__':


	class TrainingHistory(Callback):
		
		def on_train_begin(self, logs={}):
			self.losses = []
			self.acc = []
			self.i = 0
			self.save_every = 50
		def on_batch_end(self, batch, logs={}):
			self.losses.append(logs.get('loss'))
			self.acc.append(logs.get('acc'))
			self.i += 1
		
	history = TrainingHistory()

	# reading stored pre-processed (in pickle format)

	subject_vocab = readPickle(os.path.join(PATH,'lingspam_subjVocab'))
	mail_vocab = readPickle(os.path.join(PATH,'lingspam_contVocab'))
	allSubjects = readPickle(os.path.join(PATH, 'allSubjects'))
	allMails = readPickle(os.path.join(PATH,'allMails'))
	allNumSubjects = readPickle(os.path.join(PATH, 'allNumSubjects'))
	allNumMails = readPickle(os.path.join(PATH,'allNumMails'))

	ls_classLabel = readPickle(os.path.join(PATH,'ls_classLabel'))
	binEncoder = LabelEncoder()
	binEncoder.fit(ls_classLabel)
	yEncoded = binEncoder.transform(ls_classLabel)

	
	subjSentences = readPickle(os.path.join(PATH,'ls_subjSentences'))
	numSentences = readPickle(os.path.join(PATH,'ls_subjNumSentences'))

	

	# word2vec model of mail subjects
	w2v_subj_ls1, w2v_subj_ls2, w2v_subjls_embed1, w2v_subjls_embed2 = wordEmbedding(subjSentences, subject_vocab, 200, 50)

	w2v_subj_ls1.save('w2v_subj_ls1')
	w2v_subj_ls2.save('w2v_subj_ls2')
	savePickle(w2v_subjls_embed1,'w2v_subjls_embed1')
	savePickle(w2v_subjls_embed2,'w2v_subjls_embed2')


	# create document representation of word vectors

	# By averaging word vectors
	avg_subjls_embed1 = averageWE(w2v_subj_ls1, subjSentences)
	avg_subjls_embed2 = averageWE(w2v_subj_ls2, subjSentences)

	savePickle(avg_subjls_embed1,'avg_subjls_embed1')
	savePickle(avg_subjls_embed2,'avg_subjls_embed2')


	# By averaging and idf weights of word vectors
	avgIDF_subjls_embed1 = averageIdfWE(w2v_subj_ls1, subject_vocab, subjSentences)
	avgIDF_subjls_embed2 = averageIdfWE(w2v_subj_ls2, subject_vocab, subjSentences)

	savePickle(avgIDF_subjls_embed1,'avgIDF_subjls_embed1')
	savePickle(avgIDF_subjls_embed2,'avgIDF_subjls_embed2')

