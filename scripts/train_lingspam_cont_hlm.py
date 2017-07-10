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

	
	## For mail contents
	#######################################################
	

	mailSentences = readPickle(os.path.join(PATH,'ls_mailSentences'))
	numSentences = readPickle(os.path.join(PATH,'ls_mailNumSentences'))

	#load pretrained embedding weight
	w2v_contls_embed1 = readPickle(os.path.join(EMBED_PATH,'w2v_contls_embed1'))
	VOCAB_LENGTH = len(mail_vocab)
	
	EMBEDDING_DIM = w2v_contls_embed1.shape[1]

	minlength, maxlength, avglength = calcSentencesStats(numSentences)

	MAX_SEQUENCE_LENGTH = maxlength

	x_train = numSentences[:-1]
	y_train = numSentences[1:]

	print('[INFO] Zero padding...')
	X = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32')

	# create model

	model = hierarchyLanguage(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, w2v_contls_embed1)

	# train model in batch to avoid memory error

	_start = 1
	mini_batches = 100
	loss = []
	acc = []

	for k in range(_start, NB_EPOCH+1):
		i_loss = []
		i_acc = []
		for i in range(0, len(X), mini_batches):
			if i + mini_batches >= len(X):
				i_end = len(X)
			else:
				i_end = i + mini_batches


			yEncoded = sentenceMatrixVectorization(y_train[i:i_end],MAX_SEQUENCE_LENGTH,VOCAB_LENGTH)

			print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X)))

			model.fit(X[i:i_end], yEncoded, batch_size=BATCH_SIZE, nb_epoch=1, callbacks=[history])

			i_loss.append(history.losses)
			i_acc.append(history.acc)

		model.save_weights('ls_cont_weights_LM_{}.hdf5'.format(k))
		loss.append(i_loss)
		acc.append(i_acc)



	model.save('ls_cont_HLM.h5')
	
	savePickle(loss,'ls_cont_HLM_history.losses')
	savePickle(acc,'ls_cont_HLM_history.acc')

	encoderCont = Model(inputs=model.input, outputs=model.get_layer('lstm_enc_1').output)
	encodedCont = encoderCont.predict(X)
	savePickle(encodedCont,'ls_cont_encoded_HLM')


