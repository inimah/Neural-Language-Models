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

	
	# By NLM-Language Model
	####################
	
	#load pretrained embedding weight
	w2v_subjls_embed1 = readPickle(os.path.join(EMBED_PATH,'w2v_subjls_embed1'))
	VOCAB_LENGTH = len(subject_vocab)
	
	EMBEDDING_DIM = w2v_subjls_embed1.shape[1]

	minlength, maxlength, avglength = calcSentencesStats(numSentences)

	MAX_SEQUENCE_LENGTH = maxlength

	x_train = numSentences[:-1]
	y_train = numSentences[1:]

	print('[INFO] Zero padding...')
	X = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, dtype='int32')

	# create model

	yEncoded = sentenceMatrixVectorization(y_train,MAX_SEQUENCE_LENGTH,VOCAB_LENGTH)
	saveH5File('ls_subj_yEncoded.h5','yEncoded',yEncoded)

	model = seqMonoEncDec(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, w2v_subjls_embed1)

	model.fit(X, yEncoded, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[history])

	model.save('ls_subj_LM.h5')
	model.save_weights('ls_subj_weights_LM.hdf5')
	savePickle(history.losses,'ls_subj_LM_history.losses')
	savePickle(history.acc,'ls_subj_LM_history.acc')

	encoderSubj = Model(inputs=model.input, outputs=model.get_layer('lstm_enc_1').output)
	encoded_subj = encoderSubj.predict(X)
	savePickle(encoded_subj,'ls_subj_encoded_LM')
