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
	label = ls_classLabel[:200]
	binEncoder = LabelEncoder()
	binEncoder.fit(label)
	yEncoded = binEncoder.transform(label)

	
	## For mail contents
	#######################################################
	

	mailSentences = readPickle(os.path.join(PATH,'ls_mailSentences'))
	#numSentences = readPickle(os.path.join(PATH,'ls_mailNumSentences'))
	

	# cleaning from punctuations - including consecutive duplicates of punctuation

	newSent=[]
	for sent in sentences:
		tmp = []
		for word in sent:
			if word not in string.punctuation:
				tmp.append(word)
		newSent.append(tmp)

	# transform into numerical sequence values
	numSentences = []
	for sent in newSent:
		numSentences.append(wordsToIndex(mail_vocab,sent))

	x_train = np.array(numSentences[:200])

	n_samples = len(x_train)

	#load pretrained embedding weight
	w2v_contls_embed1 = readPickle(os.path.join(EMBED_PATH,'w2v_contls_embed1'))
	VOCAB_LENGTH = len(mail_vocab)
	TIME_STEPS = 200
	
	EMBEDDING_DIM = w2v_contls_embed1.shape[1]

	minlength, maxlength, avglength = calcSentencesStats(x_train)

	if maxlength > TIME_STEPS:
		MAX_SEQUENCE_LENGTH = maxlength
	else:
		MAX_SEQUENCE_LENGTH = TIME_STEPS


	print('[INFO] Zero padding...')
	X = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32')

	

	# create model
	model = classificationModel(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, w2v_contls_embed1)

	model.fit(X, yEncoded, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[history])

	model.save('ls_cont4_CM.h5')
	model.save_weights('ls_cont4_weights_CM.hdf5')
	savePickle(history.losses,'ls_cont4_CM_history.losses')
	savePickle(history.acc,'ls_cont4_CM_history.acc')

	encoderSubj = Model(inputs=model.input, outputs=model.get_layer('lstm_enc').output)
	encoded_subj = encoderSubj.predict(X)
	savePickle(encoded_subj,'ls_cont4_encoded_CM')



