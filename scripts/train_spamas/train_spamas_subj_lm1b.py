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

def _encodeText(tokenized_docs, vocab):

	# encode tokenized of words in document into its index/numerical value in vocabulary list
	# the input is in array list tokenized documents

	encoded_docs = []

	for i, arrTokens in enumerate(tokenized_docs):
		encoded_docs.append(wordsToIndex(vocab,arrTokens))

	return encoded_docs

def _encodeLabelledText(tokenized_docs, vocab):

	# encode tokenized of words in document into its index/numerical value in vocabulary list
	# the input is in array list tokenized documents

	encoded_docs = []

	for i, data in enumerate(tokenized_docs):
		encoded_docs.append((data[0],wordsToIndex(vocab,data[1])))

	return encoded_docs

# from labelled tokenized documents
def _countWord(tokenized_docs):
	count_words = []
	for i,data in enumerate(tokenized_docs):
		count_ = len(data[1])
		count_words.append((data[0],count_))
	return count_words



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

	# Vocabulary
	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))

	'''
	# Labelled tokenized documents
	labelled_subj = readPickle(os.path.join(EMBED_PATH,'spamas_w2v_labelled_subj'))
	tokenized_docs = readPickle(os.path.join(EMBED_PATH,'spamas_w2v_tokenized_docs'))
	class_labels = readPickle(os.path.join(EMBED_PATH,'spamas_w2v_class_labels'))

	# discard subjects with number of words > 25 (as being seen in statistics of subject title)
	subjects = []
	for i, data in enumerate(labelled_subj):
		if len(data[1]) <= 25:
			subjects.append((data[0],data[1]))

	# save reduced versioned of labelled tokenized documents
	savePickle(subjects,'spamas_fin_labelled_subj')

	# Encode text into numerical tokenized format
	encoded_docs = _encodeLabelledText(subjects,subject_vocab)
	savePickle(encoded_docs,'spamas_fin_encoded_subj')

	# check statistic of each class (maximum - average number of words per class)
	count_words = _countWord(encoded_docs)
	savePickle(count_words,'spamas_count_words')

	
	
	# w1 = 'spam'
	w1 = []
	# w2 = 'easy_ham'
	w2 = []
	# w3 = 'hard_ham'
	w3 = []
	for i, data in enumerate(count_words):
		if data[0] == 'spam':
			w1.append(data[1])
		elif data[0] == 'easy_ham':
			w2.append(data[1])
		elif data[0] == 'hard_ham':
			w3.append(data[1])

	# for spam
	max_spam = max(w1)
	avg_spam = sum(w1)/len(w1)

	# max_spam = 2621
	# avg_spam = 6

	# for easy ham
	max_easyham = max(w2)
	avg_easyham = sum(w2)/len(w2)
	# max_easyham = 22
	# avg_easyham = 6

	# for hard ham
	max_hardham = max(w3)
	avg_hardham = sum(w3)/len(w3)
	# max_hardham = 13
	# avg_hardham = 5


	

	aw1 = sns.distplot(w1)
	fig_w1 = aw1.get_figure()
	fig_w1.savefig('spam_words_per_subj.png')
	fig_w1.clf()
	aw2 = sns.distplot(w2)
	fig_w2 = aw2.get_figure()
	fig_w2.savefig('easyham_words_per_subj.png')
	fig_w2.clf()
	aw3 = sns.distplot(w3)
	fig_w3 = aw3.get_figure()
	fig_w3.savefig('hardham_words_per_subj.png')
	fig_w3.clf()

	



	# randomly shuffling data
	xy_train = np.array(encoded_docs,dtype=object)
	ind_rand = np.arange(len(xy_train))
	np.random.shuffle(ind_rand)
	traindat = xy_train[ind_rand]

	xTrain = list(traindat[:,1])
	yTrainLabel = list(traindat[:,0])
	n_samples = len(xTrain)

	'''
	
	# By NLM-Language Model
	####################
	
	#load pretrained embedding weight (word2vec) from the same data sets

	# skipgram
	w2v_subj_embed1 = readPickle(os.path.join(EMBED_PATH,'w2v_subj_embed1'))
	# CBOW
	w2v_subj_embed2 = readPickle(os.path.join(EMBED_PATH,'w2v_subj_embed2'))


	VOCAB_LENGTH = len(subject_vocab)
	EMBEDDING_DIM = w2v_subj_embed1.shape[1]

	# Maximum (approx.) number of words in subject title
	MAX_SEQUENCE_LENGTH = 25

	# for language model x input is sequence of tokens (text or numeric)
	# which is started by 'SOF' token
	# and ended by 'EOF' token

	'''
	revertVocab['SOF'] = 0
	revertVocab['EOF'] = 2812
	revertVocab['UNK'] = 2813


	
	x_train = []
	y_train = []
	for i, tokens in enumerate(xTrain):	
		x_tokens = list(tokens)
		y_tokens = list(tokens)
		# insert 'SOF' token which is encoded as 0 for x input sequence
		x_tokens.insert(0,0)
		x_train.append(x_tokens)
		# append 'EOF' token which is encoded as 2812 for y output sequence
		y_tokens.append(2812)
		y_train.append(y_tokens)

	print('[INFO] Zero padding...')
	X = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH, dtype='int64')
	Y = pad_sequences(y_train, maxlen=MAX_SEQUENCE_LENGTH, dtype='int64')
	# encoding y output as one hot vector with dimension size of vocabulary
	Y_encoded = sentenceMatrixVectorization(Y,MAX_SEQUENCE_LENGTH,VOCAB_LENGTH)

	# saving data into pickle ...
	savePickle(traindat,'lm1a_traindat')
	savePickle(x_train,'lm_x_input')
	savePickle(y_train,'lm_y_output')
	savePickle(X,'lm_paddedx_input')
	savePickle(Y,'lm_paddedy_output')
	savePickle(Y_encoded,'lm_yencoded_output')

	'''

	X = readPickle(os.path.join(LM_PATH,'lm_paddedx_input'))
	Y = readPickle(os.path.join(LM_PATH,'lm_paddedy_output'))
	Y_encoded = sentenceMatrixVectorization(Y,MAX_SEQUENCE_LENGTH,VOCAB_LENGTH)
	

	model = languageModel3(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, w2v_subj_embed1)
	model.fit(X, Y_encoded, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[history])

	model.save('subj_LM1b.h5')
	model.save_weights('subj_weights_LM1b.hdf5')
	savePickle(history.losses,'subj_LM1b_loss')
	savePickle(history.acc,'subj_LM1b_accuracy')



	# embedding layer
	embedSubj = Model(inputs=model.input, outputs=model.get_layer('embedding_layer').output)
	word_embed_LM1b = embedSubj.predict(X)
	savePickle(word_embed_LM1b,'word_embed_LM1b')

	# encoder layer
	encoderSubj = Model(inputs=model.input, outputs=model.get_layer('bilstm_enc').output)
	doc_embed_LM1b = encoderSubj.predict(X)
	savePickle(doc_embed_LM1b,'doc_embed_LM1b')

	# decoder
	decoderSubj = Model(inputs=model.input, outputs=model.get_layer('bilstm_dec_2').output)
	doc_word_embed_LM1b = decoderSubj.predict(X)
	savePickle(doc_word_embed_LM1b,'doc_word_embed_LM1b')

	# output layer
	output_pred = model.predict(X)
	savePickle(output_pred,'embed_pred_LM1b')



	