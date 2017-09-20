# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "16.09.2017"
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
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical



import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-batch_size', type=int, default=100)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
args = vars(ap.parse_args())


BATCH_SIZE = args['batch_size']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']


PATH = '/home/inimah/git/Neural-Language-Models/scripts/prepdata/spamassasin'
EMBED_PATH = '/home/inimah/git/Neural-Language-Models/scripts/train_spamas/subj/w2v'
LM_PATH = '/home/inimah/git/Neural-Language-Models/scripts/train_spamas/subj/lm'

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


	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))
	xTrain = readPickle(os.path.join(LM_PATH,'xTrain'))
	yCategorical = readPickle(os.path.join(LM_PATH,'yCategorical'))
	numClasses = 3

	# skipgram
	w2v_subj_embed1 = readPickle(os.path.join(EMBED_PATH,'w2v_subj_embed1'))
	# CBOW
	w2v_subj_embed2 = readPickle(os.path.join(EMBED_PATH,'w2v_subj_embed2'))

	VOCAB_LENGTH = len(subject_vocab)
	EMBEDDING_DIM = w2v_subj_embed1.shape[1]

	# Maximum (approx.) number of words in subject title
	MAX_SEQUENCE_LENGTH = 25

	print('[INFO] Zero padding...')
	X = pad_sequences(xTrain, maxlen=MAX_SEQUENCE_LENGTH, dtype='int64')
	

	model = classificationModelBiGRUDense(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, numClasses, EMBEDDING_DIM, w2v_subj_embed1)
	model.fit(X, yCategorical, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[history])

	model.save('subj_CM1d.h5')
	model.save_weights('subj_weights_CM1d.hdf5')
	savePickle(history.losses,'subj_CM1d_loss')
	savePickle(history.acc,'subj_CM1d_accuracy')

	# embedding layer
	embedSubj = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
	sent_embed_CM1d = embedSubj.predict(X)
	#savePickle(sent_embed_CM1d,'sent_embed_CM1d')
	saveH5File('sent_embed_CM1d.h5','sent_embed_CM1d',sent_embed_CM1d)

	# encoder layer
	encoderSubj = Model(inputs=model.input, outputs=model.get_layer('bigru_encoder').output)
	doc_embed_CM1d = encoderSubj.predict(X)
	#savePickle(doc_embed_CM1d,'doc_embed_CM1d')
	saveH5File('doc_embed_CM1d.h5','doc_embed_CM1d',doc_embed_CM1d)

	# output layer
	output_pred = model.predict(X)
	saveH5File('embed_pred_CM1d.h5','output_pred',output_pred)