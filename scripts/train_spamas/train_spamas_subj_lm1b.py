# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "16.09.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

############################################################
# this sequence (BIDIRECTIONAL LSTM) language model utilizes
# full encoder - decoder model 
# with TimeDistributed layer that also returns all time steps (sequence of words per subject title document) to output layer

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
import seaborn as sns


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

	# skipgram
	w2v_subj_embed1 = readPickle(os.path.join(EMBED_PATH,'w2v_subj_embed1'))
	# CBOW
	w2v_subj_embed2 = readPickle(os.path.join(EMBED_PATH,'w2v_subj_embed2'))

	VOCAB_LENGTH = len(subject_vocab)
	EMBEDDING_DIM = w2v_subj_embed1.shape[1]

	# Maximum (approx.) number of words in subject title
	MAX_SEQUENCE_LENGTH = 25


	X = readPickle(os.path.join(LM_PATH,'lm_paddedx_input'))
	Y = readPickle(os.path.join(LM_PATH,'lm_paddedy_output'))
	Y_encoded = readH5File((os.path.join(LM_PATH,'lm_yencoded_output.h5')),'Y_encoded')
	

	model = languageModelBiLSTM(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, w2v_subj_embed1)
	model.fit(X, Y_encoded, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[history])

	model.save('subj_LM1b.h5')
	model.save_weights('subj_weights_LM1b.hdf5')
	savePickle(history.losses,'subj_LM1b_loss')
	savePickle(history.acc,'subj_LM1b_accuracy')



	# embedding layer
	embedSubj = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
	sent_embed_LM1b = embedSubj.predict(X)
	#savePickle(sent_embed_LM1b,'sent_embed_LM1b')
	saveH5File('sent_embed_LM1b.h5','sent_embed_LM1b',sent_embed_LM1b)

	# encoder layer
	encoderSubj = Model(inputs=model.input, outputs=model.get_layer('bilstm_encoder').output)
	doc_embed_LM1b = encoderSubj.predict(X)
	#savePickle(doc_embed_LM1b,'doc_embed_LM1b')
	saveH5File('doc_embed_LM1b.h5','doc_embed_LM1b',doc_embed_LM1b)

	# decoder
	decoderSubj = Model(inputs=model.input, outputs=model.get_layer('bilstm_decoder_2').output)
	doc_sent_embed_LM1b = decoderSubj.predict(X)
	#savePickle(doc_sent_embed_LM1b,'doc_sent_embed_LM1b')
	saveH5File('doc_sent_embed_LM1b.h5','doc_sent_embed_LM1b',doc_sent_embed_LM1b)

	# output layer
	output_pred = model.predict(X)
	saveH5File('embed_pred_LM1b.h5','output_pred',output_pred)



	