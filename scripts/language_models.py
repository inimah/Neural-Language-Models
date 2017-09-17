# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "01.07.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import numpy as np
import nltk
import math
from gensim.models import Word2Vec, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential, Model
from keras.layers import *
from keras.preprocessing import sequence
from text_preprocessing import *
from functools import partial
from keras.optimizers import Adam, RMSprop, SGD
from keras import optimizers
from keras.layers.merge import Concatenate


# since we preserve all possible words/characters/information, we need to add regex in tokenizer of sklearn TfIdfVectorizer 
'''
pattern = r"""
 (?x)                   # set flag to allow verbose regexps
 (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
 |\$?\d+(?:\.?,\d+)?%?       # numbers, incl. currency and percentages
 |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
 |(?:[`'^~\":;,.?(){}/\\/+/\-|=#$%@&*\]\[><!])         # special characters with meanings
 """
 '''


################################################
# word2vec models
################################################
def wordEmbedding(documents, vocab, argsize, argiter):

	
	# description of parameters in gensim word2vec model

	#`sg` defines the training algorithm. By default (`sg=0`), CBOW is used. Otherwise (`sg=1`), skip-gram is employed.
	#`min_count` = ignore all words with total frequency lower than this
	#`max_vocab_size` = limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types
	# need about 1GB of RAM. Set to `None` for no limit (default).
	#`workers` = use this many worker threads to train the model (=faster training with multicore machines).
	#`hs` = if 1, hierarchical softmax will be used for model training. If set to 0 (default), and `negative` is non-zero, negative sampling will be used.
	#`negative` = if > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20).
	# Default is 5. If set to 0, no negative samping is used.
	

	word2vec_models = [
	# skipgram model with hierarchical softmax and negative sampling
	Word2Vec(size=argsize, min_count=0, window=5, sg=1, hs=1, negative=5, iter=argiter),
	# cbow model with hierarchical softmax and negative sampling
	Word2Vec(size=argsize, min_count=0, window=5, sg=0, hs=1, negative=5, iter=argiter)

	]

	# model 1 = skipgram 
	model1 = word2vec_models[0]
	# model 2 = cbow
	model2 = word2vec_models[1]
 

	# building vocab for each model (creating 2, in case one model subsampling word vocabulary differently)
	model1.build_vocab(documents)
	model2.build_vocab(documents)

	word2vec_vocab1 = dict([(k, v.index) for k, v in model1.wv.vocab.items()])   
	word2vec_vocab2 = dict([(k, v.index) for k, v in model2.wv.vocab.items()])

	embedding1 = np.zeros(shape=(len(vocab), argsize), dtype='float32')
	embedding2 = np.zeros(shape=(len(vocab), argsize), dtype='float32')

	print('Training word2vec model...')

	# number of tokens
	n_tokens = sum([len(sent) for sent in documents])
	# number of sentences/documents
	n_examples = len(documents)
	model1.train(documents, total_words=n_tokens, total_examples=n_examples, epochs=argiter)
	model2.train(documents, total_words=n_tokens, total_examples=n_examples, epochs=argiter)
	

	word2vec_weights1 = model1.wv.syn0
	word2vec_weights2 = model2.wv.syn0    

	for i, w in vocab.items():

		if w not in word2vec_vocab1:
			continue
		embedding1[i, :] = word2vec_weights1[word2vec_vocab1[w], :]

		if w not in word2vec_vocab2:
			continue
		embedding2[i, :] = word2vec_weights2[word2vec_vocab2[w], :]

	
	return model1, model2, embedding1, embedding2

################################################
# doc2vec models
################################################
def docEmbedding(documents, vocab, argsize, argiter):


	
	# doc2vec models
	doc2vec_models = [
	# PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
	Doc2Vec(dm=1, dm_concat=1, size=argsize, window=5, negative=5, hs=1, min_count=0, alpha=0.025, min_alpha=0.025),
	# PV-DBOW 
	Doc2Vec(dm=0, size=argsize, negative=5, hs=1, min_count=0, alpha=0.025, min_alpha=0.025),
	# PV-DM w/average
	Doc2Vec(dm=1, dm_mean=1, size=argsize, window=5, negative=5, hs=1, min_count=0, alpha=0.025, min_alpha=0.025),
	]

	model1 = doc2vec_models[0]
	model2 = doc2vec_models[1]
	model3 = doc2vec_models[2]

	model1.build_vocab(documents)
	model2.build_vocab(documents)
	model3.build_vocab(documents)


	doc2vec_vocab1 = dict([(k, v.index) for k, v in model1.wv.vocab.items()])   
	doc2vec_vocab2 = dict([(k, v.index) for k, v in model2.wv.vocab.items()])
	doc2vec_vocab3 = dict([(k, v.index) for k, v in model3.wv.vocab.items()])

	print('Training doc2vec model...')

	# number of tokens
	n_tokens = sum([len(sent) for sent in documents])
	# number of sentences/documents
	n_examples = len(documents)

	model1.train(documents, total_examples=n_examples, epochs=argiter)
	model2.train(documents, total_examples=n_examples, epochs=argiter)
	model3.train(documents, total_examples=n_examples, epochs=argiter)


	doc2vec_weights1 = np.array(model1.docvecs)
	doc2vec_weights2 = np.array(model2.docvecs)
	doc2vec_weights3 = np.array(model3.docvecs)

	return model1, model2, model3, doc2vec_weights1, doc2vec_weights2, doc2vec_weights3


################################################
# generating sentence-level / document embedding by averaging word2vec
# document here is sentence - or sequence of words
################################################
def averageWE(w2v_weights, vocab, documents):

	#w2v_vocab = word2vec_model.wv.index2word
	#w2v_weights = word2vec_model.wv.syn0
	w2v_vocab = vocab.values()
	w2v = dict(zip(w2v_vocab, w2v_weights))
	dim = len(w2v.itervalues().next())

	doc_embedding = []

	for i,text in enumerate(documents):
		embedding = np.mean([w2v[w] for w in text if w in w2v]
			or [np.zeros(dim)], axis=0)
		
		doc_embedding.append(embedding)

	return np.array(doc_embedding)

################################################
# generating sentence-level / document embedding by averaging word2vec and Tf-Idf penalty
# document here is sentence - or sequence of words
################################################

def countFrequency(word, doc):
	return doc.count(word)

def docFrequency(word, list_of_docs):
	count = 0
	for document in list_of_docs:
		if countFrequency(word, document) > 0:
			count += 1
	return 1 + count

def computeIDF(word, list_of_docs):

	# idf(term) = ( log ((1 + nd)/(1 + df(doc,term))) ) 
	# where nd : number of document in corpus; 
	# df : doc frequency (number of documents containing term)

	idf = math.log( (1 + len(list_of_docs)) / float(docFrequency(word, list_of_docs)))

	return idf

def averageIdfWE(w2v_weights, vocab, documents):

	#w2v_vocab = word2vec_model.wv.index2word
	#w2v_weights = word2vec_model.wv.syn0


	
	print('calculating Tf-Idf weights...')


	words = []
	for k,v in vocab.iteritems():
		words.append(v)

	w2v_vocab = words
	w2v = dict(zip(w2v_vocab, w2v_weights))
	dim = len(w2v.itervalues().next())

	wordIdf = {}

	for i,txt in enumerate(words): 
		wordIdf[txt] = computeIDF(txt, documents)

	doc_embedding = []
	for i,text in enumerate(documents):
		embedding = np.mean([w2v[w] * wordIdf[w]
				for w in text if w in w2v] or
				[np.zeros(dim)], axis=0)
		
		doc_embedding.append(embedding)

	return np.array(doc_embedding)


################################################
# Neural Language Model (Sequence prediction)
# encoder - decoder architecture
# objective: predicts the next word sequence (target word) based on previous context words 
# sentence / document embedding is retrieved from the encoder part

################################################

# with keras sequential model
# full encoder - decoder model 
def languageModel1(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 50
	num_layers = 3

	model = Sequential()
	
	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable = True, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))

	# Creating encoder network
	# encoding text input (sequence of words) into sentence embedding
	model.add(LSTM(hidden_size,name='lstm_enc'))
	model.add(RepeatVector(MAX_SEQUENCE_LENGTH))

	# Creating decoder network
	# objective function: predicting next words (language model)
	for i in range(num_layers):
		model.add(LSTM(hidden_size, name='lstm_dec%s'%(i+1), return_sequences=True))
	model.add(TimeDistributed(Dense(VOCAB_LENGTH), name='td_output'))
	model.add(Activation('softmax', name='last_output'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model

# with keras functional API
# full encoder - decoder model 
# apply for TimeDistributed layer

def languageModel2(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 50
	num_layers = 3

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	lstm_layer = LSTM(hidden_size, name='lstm_enc')(embedded_layer)
	encoder_output = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(lstm_layer)

	# Creating decoder network
	# objective function: predicting next words (language model)
	for i in range(num_layers):
		decoder_layer = LSTM(hidden_size, return_sequences=True,name='lstm_dec_%s'%(i))(encoder_output)
	prediction = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax', name='dense_output'))(decoder_layer)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_1 (InputLayer)         (None, 1000)              0         
	_________________________________________________________________
	embedding_1 (Embedding)      (None, 1000, 100)         8056300    -
	_________________________________________________________________
	lstm_1 (LSTM)                (None, 100)               80400      -> doc vector
	_________________________________________________________________
	encoder_repeat (RepeatVector (None, 1000, 100)         0         
	_________________________________________________________________
	lstm_dec_3 (LSTM)            (None, 1000, 100)         80400     
	_________________________________________________________________
	time_distributed_1 (TimeDist (None, 1000, 2)           202       
	=================================================================
	Total params: 8,217,302
	Trainable params: 8,217,302
	Non-trainable params: 0
	_________________________________________________________________

	'''

	return model

# with Bidirectional LSTM
def languageModel3(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 25
	num_layers = 3

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	lstm_layer = Bidirectional(LSTM(hidden_size, name='lstm_enc_1'))(embedded_layer)
	encoder_output = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(lstm_layer)

	# Creating decoder network
	# objective function: predicting next words (language model)
	for i in range(num_layers):
		decoder_layer = Bidirectional(LSTM(hidden_size, return_sequences=True,name='lstm_dec_%s'%(i+2)))(encoder_output)
	prediction = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax', name='dense_output'))(decoder_layer)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_1 (InputLayer)         (None, 1000)              0         
	_________________________________________________________________
	embedding_1 (Embedding)      (None, 1000, 100)         8056300    -
	_________________________________________________________________
	lstm_1 (LSTM)                (None, 100)               80400      -> doc vector
	_________________________________________________________________
	encoder_repeat (RepeatVector (None, 1000, 100)         0         
	_________________________________________________________________
	lstm_dec_3 (LSTM)            (None, 1000, 100)         80400     
	_________________________________________________________________
	time_distributed_1 (TimeDist (None, 1000, 2)           202       
	=================================================================
	Total params: 8,217,302
	Trainable params: 8,217,302
	Non-trainable params: 0
	_________________________________________________________________

	'''

	return model

# with keras functional API
# only encoder
# apply for Dense layer

def languageModel4(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 50
	num_layers = 3

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	lstm_layer = LSTM(hidden_size, name='lstm_enc')(embedded_layer)

	prediction = Dense(VOCAB_LENGTH, activation='softmax', name='dense_output')(lstm_layer)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_1 (InputLayer)         (None, 1000)              0         
	_________________________________________________________________
	embedding_1 (Embedding)      (None, 1000, 100)         8056300   
	_________________________________________________________________
	lstm_1 (LSTM)                (None, 100)               80400     
	_________________________________________________________________
	dense_output (Dense)         (None, 2)                 202       
	=================================================================
	Total params: 8,136,902
	Trainable params: 8,136,902
	Non-trainable params: 0

	'''

	return model

# with bidirectional LSTM
def languageModel5(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 25
	num_layers = 3

	sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
	embedded_layer = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sequence_input)
	lstm_layer = Bidirectional(LSTM(hidden_size, name='lstm_enc'))(embedded_layer)

	prediction = Dense(VOCAB_LENGTH, activation='softmax', name='dense_output')(decoder_layer)
	model = Model(sequence_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	________________________________________________________________
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_1 (InputLayer)         (None, 1000)              0         
	_________________________________________________________________
	embedding_1 (Embedding)      (None, 1000, 100)         8056300   
	_________________________________________________________________
	lstm_1 (LSTM)                (None, 100)               80400     
	_________________________________________________________________
	dense_output (Dense)         (None, 2)                 202       
	=================================================================
	Total params: 8,136,902
	Trainable params: 8,136,902
	Non-trainable params: 0

	'''

	return model

# with keras functional API

################################################
# Neural Classification Model 
# encoder - decoder architecture
# objective: predicts class label of input
# sentence / document embedding is retrieved from the encoder part

################################################

def classificationModel(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 200

	model = Sequential()
	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable = True, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))
	model.add(Dropout(0.25))
	model.add(LSTM(hidden_size, name='lstm_enc'))
	model.add(RepeatVector(MAX_SEQUENCE_LENGTH))
	model.add(LSTM(hidden_size, name='lstm_dec'))
	model.add(Dense(1, name='dense_output'))
	model.add(Activation('sigmoid', name='last_output'))
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())

	return model


def classificationModel2(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 200

	model = Sequential()
	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable = True, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))
	model.add(Dropout(0.25))
	model.add(LSTM(hidden_size, name='lstm_enc'))
	model.add(RepeatVector(MAX_SEQUENCE_LENGTH))
	model.add(LSTM(hidden_size, name='lstm_dec', return_sequences=True))
	model.add(TimeDistributed(Dense(1), name='td_output'))
	model.add(Activation('sigmoid', name='last_output'))
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())

	return model

################################################
# Neural Translation Model 
# encoder - decoder architecture
# objective: predicts words in target language given sequence of words in source language
# sentence / document embedding is retrieved from the encoder part

################################################
def translationModel(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM, embedding_weights):

	hidden_size = 200
	num_layers = 3

	model = Sequential()
	
	model.add(Embedding(X_vocab_len, EMBEDDING_DIM, input_length=X_max_len, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))
	# Creating encoder network
	model.add(LSTM(hidden_size,name='lstm_enc_1'))
	model.add(RepeatVector(y_max_len))

	# Creating decoder network
	for i in range(num_layers):
		model.add(LSTM(hidden_size, name='lstm_%s'%(i+2), return_sequences=True))
	model.add(TimeDistributed(Dense(y_vocab_len,name='dense')))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())


	return model


################################################
# Hierarchical Neural Language Model 
# with encoder - decoder type architecture
#
# input array for this model is in 4D shape
# ( rows,       cols,     time_steps,    n_dim     )
# ( _____   ___________   _________   ____________ )
# ( n_docs, n_sentences,   n_words    dim_embedding)
################################################

# full encoder - decoder
def hierarchyLanguage1(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	row_hidden_size = 100
	col_hidden_size = 100
	num_layers = 3


	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int64')
	# embedding layer
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	# Encoder model
	# Encodes sentences
	sent_lstm = LSTM(row_hidden_size, name='lstm_enc_1')(embedded_sentences)
	sent_model = Model(sentences_input, sent_lstm)

	doc_input = Input(shape=(MAX_SENTS,MAX_SEQUENCE_LENGTH), dtype='int64')

	encoded_sentences = TimeDistributed(sent_model)(doc_input)
	
	# Encodes documents
	encoded_docs = LSTM(col_hidden_size, name='lstm_enc_2')(encoded_sentences)

	# Decoder model
	encoder_output = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(encoded_docs)

	# Creating decoder network
	# objective function: predicting next words (language model)
	for i in range(num_layers):
		decoder_layer = LSTM(col_hidden_size, return_sequences=True,name='lstm_dec_%s'%(i+2))(encoder_output)

	decoder = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax', name='dense_output'))(decoder_layer)

	model = Model(doc_input, decoder)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	input_2 (InputLayer)         (None, 15, 100)           0
	_________________________________________________________________
	time_distributed_1 (TimeDist (None, 15, 100)           8136700
	_________________________________________________________________
	lstm_enc_2 (LSTM)            (None, 100)               80400
	_________________________________________________________________
	encoder_repeat (RepeatVector (None, 100, 100)          0
	_________________________________________________________________
	lstm_dec_3 (LSTM)            (None, 100, 100)          80400
	_________________________________________________________________
	time_distributed_2 (TimeDist (None, 100, 80563)        8136863
	=================================================================
	Total params: 16,434,363
	Trainable params: 16,434,363
	Non-trainable params: 0
	_________________________________________________________________
	None


	'''


	return model


# USE THIS
# only encoder (no decoder layer)
# and dense layer
def hierarchyLanguage2(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 100

	# this will process sequence of words in a current sentence 
	# LSTM layer for this input --> predicts word based on preceding words --> this will represent our sentence vector
	# P(wt | w1...wt-1) or in another words P(sentence) = P(w1...wt-1)
	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int64')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	sent_lstm = LSTM(hidden_size, name='lstm_enc_1')(embedded_sentences)
	sent_model = Model(sentences_input, sent_lstm)

	# this will process sequence of sentences as part of a document
	# representing document vector
	# P(St | S1..St-1)
	doc_input = Input(shape=(MAX_SENTS,MAX_SEQUENCE_LENGTH), dtype='int64')
	# the following TimeDistributed layer will link LSTM learning of sentence model (as joint distribution of words in sentence) to document model (as joint distribution of sentences)
	encoded_sentences = TimeDistributed(sent_model)(doc_input)
	encoded_docs = LSTM(hidden_size, name='lstm_enc_2')(encoded_sentences)
	
	# the class label here is words in fixed vocabulary list
	prediction = Dense(VOCAB_LENGTH, activation='softmax', name='dense_output')(encoded_docs)
	model = Model(doc_input, prediction)

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	input_2 (InputLayer)         (None, 15, 100)           0
	_________________________________________________________________
	time_distributed_1 (TimeDist (None, 15, 100)           8136700
	_________________________________________________________________
	lstm_enc_2 (LSTM)            (None, 100)               80400
	_________________________________________________________________
	dense_output (Dense)         (None, 80563)             8136863
	=================================================================
	Total params: 16,353,963
	Trainable params: 16,353,963
	Non-trainable params: 0
	_________________________________________________________________
	None


	'''


	return model

################################################
# Hierarchical Neural Classification Model 
#
# input array for this model is in 4D shape (after embedding layer)
# ( cols,       rows,     time_steps,    n_dim     )
# ( _____   ___________   _________   ____________ )
# ( n_docs, n_sentences,   n_words    dim_embedding)
################################################


# !!!!NOT BEING USED!!!!
# with time distributed layer to get the projection of vector on class labels 
# include latent vector dimension in the last layer (, ndim, class size) instead of just (, class size)
def hierarchyClassifier1(MAX_SENTENCES, MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):

	
	row_hidden_size = 100
	col_hidden_size = 100

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	sentences_input = Input(shape=(MAX_SENTENCES,MAX_SEQUENCE_LENGTH,), dtype='int64')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	lstm_sentence = TimeDistributed(LSTM(row_hidden_size,name='lstm_enc_1'))(embedded_sentences)
	
	docs_model = LSTM(col_hidden_size,name='lstm_enc_2', return_sequences=True)(lstm_sentence)

	# Prediction
	prediction = TimeDistributed(Dense(num_classes, activation='softmax', name='dense_out'))(docs_model)
	model = Model(sentences_input, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	Layer (type)                 Output Shape              Param #   
	=================================================================
	input_2 (InputLayer)         (None, 15, 100)           0         
	_________________________________________________________________
	time_distributed_1 (TimeDist (None, 15, 100)           8136700   
	_________________________________________________________________
	lstm_enc_2 (LSTM)            (None, 15, 100)           80400     
	_________________________________________________________________
	time_distributed_2 (TimeDist (None, 15, 80563)         8136863   
	=================================================================
	Total params: 16,353,963
	Trainable params: 16,353,963
	Non-trainable params: 0
	_________________________________________________________________
	None

	'''


	return model


# USE THIS
# with dense layer
def hierarchyClassifier2(MAX_SENTENCES, MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):


	row_hidden_size = 100
	col_hidden_size = 100

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	lstm_sentence = LSTM(row_hidden_size,name='lstm_enc_1')(embedded_sentence)
	sentences_model = Model(sentences_input, lstm_sentence)

	docs_input = Input(shape=(MAX_SENTENCES, MAX_SEQUENCE_LENGTH), dtype='int64')
	docs_encoded = TimeDistributed(sentences_model)(docs_input)
	#dropout = Dropout(0.2)(docs_encoded)
	docs_model = LSTM(col_hidden_size,name='lstm_enc_2')(docs_encoded)

	# Prediction
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(docs_model)
	model = Model(docs_input, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	'''
	in this summary, number of classes = vocab size
	_________________________________________________________________
		Layer (type)                 Output Shape              Param #
	=================================================================
	input_2 (InputLayer)         (None, 15, 100)           0
	_________________________________________________________________
	time_distributed_1 (TimeDist (None, 15, 100)           8136700
	_________________________________________________________________
	lstm_enc_2 (LSTM)            (None, 100)               80400
	_________________________________________________________________
	dense_out (Dense)            (None, 80563)             8136863
	=================================================================
	Total params: 16,353,963
	Trainable params: 16,353,963
	Non-trainable params: 0
	_________________________________________________________________
	None

	'''


	return model



def seqTDEncDec(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	row_hidden_size = 200
	col_hidden_size = 200
	num_layers = 3

	model = Sequential()	
	# input captured here is sequence of sentences in shape (rows, time_steps, n_dim)
	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,), mask_zero=True, trainable = True, weights=[embedding_weights], name='embedding_layer'))
	# encoding rows (sentences)
	model.add(TimeDistributed(LSTM(row_hidden_size,name='lstm_enc_1')))

	model.add(RepeatVector(y_max_len))

	# Creating decoder network
	for i in range(num_layers):
		model.add(LSTM(hidden_size, name='lstm_%s'%(i+2), return_sequences=True))
	model.add(TimeDistributed(Dense(y_vocab_len,name='dense')))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	# encoding cols (documents)
	model.add(LSTM(col_hidden_size,name='lstm_enc_2'))
	model.add(Dense(num_classes, activation='softmax', name='dense_out'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())

	return model

def seqTDClassifier(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):

	row_hidden_size = 200
	col_hidden_size = 200

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	model = Sequential()	
	# input captured here is sequence of sentences in shape (rows, time_steps, n_dim)
	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,), mask_zero=True, trainable = True, weights=[embedding_weights], name='embedding_layer'))
	# encoding rows (sentences)
	model.add(TimeDistributed(LSTM(row_hidden_size,name='lstm_enc_1')))

	# encoding cols (documents)	model.add(TimeDistributed(LSTM(row_hidden_size,name='lstm_enc_1')))
	
	model.add(LSTM(col_hidden_size,name='lstm_enc_2'))
	model.add(Dense(num_classes, activation='softmax', name='dense_out'))
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())


	return model


################################################
# Hierarchical LSTM encoder - decoder type 2
# with keras functional API vs. sequential model





def seqClassifier(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):

	n_docs = 1
	row_hidden_size = 200
	col_hidden_size = 200

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'


	sentences_model = Sequential()
	# input captured here is sequence of sentences in shape (rows, time_steps, n_dim)
	sentences_model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,), mask_zero=True, trainable = True, weights=[embedding_weights], name='embedding_layer'))
	# Creating encoder for capturing sentence embedding
	sentences_model.add(LSTM(row_hidden_size,name='lstm_enc_1'))

	# Creating encoder for capturing document embedding
	docs_model = Sequential()
	docs_model.add(TimeDistributed(Input(shape=(n_docs, MAX_SEQUENCE_LENGTH), dtype='int64', name='td_input_docs')))
	

	model = Sequential()
	model.add(Merge([sentences_model, docs_model], mode='concat'))
	model.add(LSTM(col_hidden_size,name='lstm_enc_2'))

	# Prediction
	model.add(Dense(num_classes, activation='softmax', name='dense_out'))
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model

# with time-distributed layer
# return all time steps in previous layer
def simpleSeqClassifier2(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 200

	model = Sequential()

	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable = True, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))
	model.add(Dropout(0.25))

	model.add(LSTM(hidden_size, name='lstm_enc'))
	model.add(RepeatVector(MAX_SEQUENCE_LENGTH))
	#model.add(LSTM(hidden_size, name='lstm_dec',return_sequences=True))

	model.add(LSTM(hidden_size, name='lstm_dec',return_sequences=True))
	model.add(TimeDistributed(Dense(1)))

	#model.add(TimeDistributed(Dense(1)))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())

	return model


# old model without pre-trained embedding matrix
def seqEncDec_old(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM):

	hidden_size = 200
	num_layers = 3

	model = Sequential()
	
	model.add(Embedding(X_vocab_len, EMBEDDING_DIM, input_length=X_max_len, mask_zero=True, name='embedding_layer'))
	# Creating encoder network
	model.add(LSTM(hidden_size,name='lstm_enc_1'))
	model.add(RepeatVector(y_max_len))

	# Creating decoder network
	for i in range(num_layers):
		model.add(LSTM(hidden_size, name='lstm_%s'%(i+2), return_sequences=True))
	model.add(TimeDistributed(Dense(y_vocab_len,name='dense')))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())


	return model

################################################
# learning non-aligned bi-lingual input and output
# by merging 2 sequential models

################################################
def seqParallelEnc(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM, X_embedding_weights, y_embedding_weights):

	hidden_size = 200
	nb_classes = 2

	encoder_a = Sequential()
	encoder_a.add(Embedding(X_vocab_len, EMBEDDING_DIM, input_length=X_max_len, mask_zero=True, weights=[X_embedding_weights], name='X_embedding_layer'))
	encoder_a.add(LSTM(hidden_size,name='lstm_a'))

	encoder_b = Sequential()
	encoder_b.add(Embedding(y_vocab_len, EMBEDDING_DIM, input_length=y_max_len, mask_zero=True, weights=[y_embedding_weights], name='y_embedding_layer'))
	encoder_b.add(LSTM(hidden_size,name='lstm_b'))

	decoder = Sequential()
	decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
	decoder.add(Dense(hidden_size, activation='relu'))
	decoder.add(Dense(nb_classes, activation='softmax'))

	decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	print(decoder.summary())


	return decoder



	
