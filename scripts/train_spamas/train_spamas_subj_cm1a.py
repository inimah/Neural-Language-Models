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
	train_dat = readPickle(os.path.join(LM_PATH,'lm_traindat'))

	numClasses = 3
	xTrain = list(train_dat[:,1])
	# the following labels are still in nominal form ('spam', 'easy_ham', 'hard_ham')
	yTrainLabel = list(train_dat[:,0])
	# the following sklearn module will tranform nominal to numerical (0,1,2)
	numEncoder = LabelEncoder()
	numEncoder.fit(yTrainLabel)
	yNumerical = numEncoder.transform(yTrainLabel)
	# because our output is multiclass classification problems, 
	# we need to transform the class label into categorical encoding ([1,0,0],[0,1,0],[0,0,1])
	yCategorical = to_categorical(yNumerical, num_classes=numClasses)

	savePickle(yCategorical,'yCategorical')

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

	model = classificationModelLSTMDense(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, numClasses, EMBEDDING_DIM, w2v_subj_embed1)
	model.fit(X, yCategorical, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[history])

	model.save('subj_CM2a.h5')
	model.save_weights('subj_weights_CM2a.hdf5')
	savePickle(history.losses,'subj_CM2a_loss')
	savePickle(history.acc,'subj_CM2a_accuracy')

	# embedding layer
	embedSubj = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
	sent_embed_CM2a = embedSubj.predict(X)
	#savePickle(sent_embed_CM2a,'sent_embed_CM2a')
	saveH5File('sent_embed_CM2a.h5','sent_embed_CM2a',sent_embed_CM2a)

	# encoder layer
	encoderSubj = Model(inputs=model.input, outputs=model.get_layer('lstm_encoder').output)
	doc_embed_CM2a = encoderSubj.predict(X)
	#savePickle(doc_embed_CM2a,'doc_embed_CM2a')
	saveH5File('doc_embed_CM2a.h5','doc_embed_CM2a',doc_embed_CM2a)

	# output layer
	output_pred = model.predict(X)
	saveH5File('embed_pred_CM2a.h5','output_pred',output_pred)