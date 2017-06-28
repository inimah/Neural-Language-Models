# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "20.05.2017"
#__version__ = "1.0.1"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import os
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from text_preprocessing import *
from language_models import *
from keras.callbacks import Callback


import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-batch_size', type=int, default=100)
ap.add_argument('-layer_num', type=int, default=3)
ap.add_argument('-hidden_dim', type=int, default=200)
ap.add_argument('-embedding_dim', type=int, default=200)
ap.add_argument('-nb_epoch', type=int, default=10)
ap.add_argument('-mode', default='train')
args = vars(ap.parse_args())


BATCH_SIZE = args['batch_size']
LAYER_NUM = args['layer_num']
EMBEDDING_DIM = args['embedding_dim']
HIDDEN_DIM = args['hidden_dim']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']


PATH = '../data/europarl_sample'




if __name__ == '__main__':


	
	# get list of data files
	filenames = listData(PATH)
	
		# grouped by class
	datadict = getClassLabel(filenames)

	# return tokenized subject and mail content 
	tokens, worddocs_freq, vocab, alltokens, alldocs = generatePairset(datadict)

	# length of vocab for each language
	X_vocab_len = len(vocab[0])
	y_vocab_len = len(vocab[1])

	# split document into sentences - taken from text-tokenized dictionary, not the numeric one
	sentences = getSentencesClass(alltokens)

	# transform into numeric-encoded sentences
	numSentences = sentToNumBi(sentences,vocab)

	# Finding the length of the longest sequence for both languages
	# statistics of sentences in document corpus
	nSentences, nWords, minSent, maxSent, sumSent, avgSent, minWords, maxWords, sumWords, avgWords = getStatClass(numSentences)

	# maximum length of word sequence in english sentences
	X_max_len = maxWords[0][0]

	# maximum length of word sequence in non-english (e.g. nl) sentences
	y_max_len = maxWords[1][0]

	print('[INFO] Zero padding...')
	X = pad_sequences(numSentences[0][0], maxlen=X_max_len, dtype='int32')
	y = pad_sequences(numSentences[1][0], maxlen=y_max_len, dtype='int32')

	print('[INFO] Compiling model...')
	model = seqEncDec(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM, HIDDEN_DIM, LAYER_NUM)
	encoder = Model(inputs=model.input, outputs=model.get_layer('lstm_enc_1').output)

	saved_weights = findWeights('../weights')

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

	#_N = X_max_len
	_N = 100
	if MODE == 'train':
		_start = 1
		# If any trained weight was found, then load them into the model
		if len(saved_weights) != 0:
			print('[INFO] Saved weights found, loading...')
			epoch = saved_weights[saved_weights.rfind('_')+1:saved_weights.rfind('.')]
			model.load_weights(saved_weights)
			_start = int(epoch) + 1

		_end = 0
		loss = []
		acc = []
		for k in range(_start, NB_EPOCH+1):
			

			i_loss = []
			i_acc = []

			# Training n sequences 
			for i in range(0, len(X), _N):
				if i + _N >= len(X):
					i_end = len(X)
				else:
					i_end = i + _N

				# encode y output to one-hot vector (categorical word classes)
				yEncoded = sentenceMatrixVectorization(y[i:i_end], y_max_len, y_vocab_len)

				print('[INFO] Training model: epoch {}th {}/{} samples'.format(k, i, len(X)))

				model.fit(X[i:i_end], y, batch_size=BATCH_SIZE, nb_epoch=1, verbose=2, callbacks=[history])

				i_loss.append(history.losses)
				i_acc.append(history.acc)

				model.save_weights('sentweights_noshuffle_{}_{}.hdf5'.format(k,i))
			model.save_weights('weights_noshuffle_{}.hdf5'.format(k))

			loss.append(i_loss)
			acc.append(i_acc)

		model.save('bi_model_noshuffle.h5')
		savePickle(loss,'err_loss')
		savePickle(acc,'accuracy')

