# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "14.06.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import string
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
	label = ls_classLabel[:500]
	binEncoder = LabelEncoder()
	binEncoder.fit(label)
	yEncoded = binEncoder.transform(label)

	
	## For mail contents
	#######################################################
	
	'''
	mailSentences = readPickle(os.path.join(PATH,'ls_mailSentences'))
	numSentences = readPickle(os.path.join(PATH,'ls_mailNumSentences'))

	# splitting documents into sentences
	sentences = []
	for i in range(len(mailSentences)):
		sentences.append(splitSentences(mailSentences[i]))

	# cleaning from punctuations - including consecutive duplicates of punctuation
	# sentences[2347][91]
	newSent=[]
	for sent in sentences:
		tmp1 = []
		for words in sent:
			tmp2 = []
			for word in words:
				if word not in string.punctuation:
					tmp2.append(word)
			tmp1.append(tmp2)
		newSent.append(tmp1)


	savePickle(sentences,'ls_mailSplitSent')
	savePickle(newSent,'ls_mailCleanSent')

	'''

	numSentences = readPickle('ls_mail500Num')

	n_samples = len(numSentences)

	# max number of sentences in documents
	MAX_SENTENCES = max([len(sent) for sent in numSentences])
	# sentences.index(max(sent for sent in sentences)) --> the index of document with the longest sentence
	# max length of words per sentence
	MAX_SEQUENCE_LENGTH = max([max(len(s) for s in sent) for sent in numSentences])

	'''

	In [4]: tmp = []
   ...: for sent in newSent:
   ...:     tmp1=[]
   ...:     for words in sent:
   ...:         tmp1.append(len(words))        
   ...:     tmp.append(tmp1)


	In [12]: for i in range(len(tmp)):
	...:     for j in range(len(tmp[i])):
	...:         if tmp[i][j]==394:
	...:             print(i,j)
	...:             
	(807, 11)

	In [15]: tmpSent = newSent[:500]

	In [16]: numSent = []

	In [17]: for i in range(len(tmpSent)):
		...:     tmp = []
		...:     for j in range(len(tmpSent[i])):
		...:         tmp.append(wordsToIndex(mail_vocab,tmpSent[i][j]))
		...:     numSent.append(tmp)
		...: 



	In [18]: 

	In [18]: 

	In [18]: len(numSent)
	Out[18]: 500


	# transform into numerical sequence values
	numSent = []
	for i in range(len(newSent)):
		tmp = []
		for j in range(len(newSent[i])):
			tmp.append(wordsToIndex(mail_vocab,newSent[i][j]))
		numSent.append(tmp)

	'''

	#load pretrained embedding weight
	w2v_contls_embed1 = readPickle(os.path.join(EMBED_PATH,'w2v_contls_embed1'))
	VOCAB_LENGTH = len(mail_vocab)
	
	EMBEDDING_DIM = w2v_contls_embed1.shape[1]

	# Get maximum number of sentences in document and maximum word length per sentence

	print('[INFO] Zero padding...')
	X = []
	for words in numSentences:
		X.append(pad_sequences(words, maxlen=MAX_SEQUENCE_LENGTH, dtype='int64'))

	X_train=np.zeros((n_samples,MAX_SENTENCES,MAX_SEQUENCE_LENGTH))   
	for i in range(len(X)):
		for j in range(len(X[i])):
			X_train[i][j]=X[i][j]



	# create model
	model = hierarchyClassifier1(MAX_SENTENCES,MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, w2v_contls_embed1,2)

	model.fit(X_train, yEncoded, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCH, callbacks=[history])

	model.save('ls_cont_HCM.h5')
	model.save_weights('ls_cont_weights_HCM.hdf5')
	savePickle(history.losses,'ls_cont_HCM_history.losses')
	savePickle(history.acc,'ls_cont_HCM_history.acc')

	encoderSubj = Model(inputs=model.input, outputs=model.get_layer('lstm_enc_1').output)
	encoded_subj = encoderSubj.predict(X)
	savePickle(encoded_subj,'ls_cont_encoded_HCM')

	


