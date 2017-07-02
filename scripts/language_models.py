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
pattern = r"""
 (?x)                   # set flag to allow verbose regexps
 (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
 |\$?\d+(?:\.?,\d+)?%?       # numbers, incl. currency and percentages
 |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
 |(?:[`'^~\":;,.?()+/\-|=#$%@&*\]\[><!])         # special characters with meanings
 """


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

	model1.train(documents, total_words=n_tokens, total_examples=n_examples, epochs=argiter)
	model2.train(documents, total_words=n_tokens, total_examples=n_examples, epochs=argiter)
	model3.train(documents, total_words=n_tokens, total_examples=n_examples, epochs=argiter)


	doc2vec_weights1 = model1.wv.syn0
	doc2vec_weights2 = model2.wv.syn0
	doc2vec_weights3 = model3.wv.syn0

	embedding1 = np.zeros(shape=(len(vocab), argsize), dtype='float32')
	embedding2 = np.zeros(shape=(len(vocab), argsize), dtype='float32')
	embedding3 = np.zeros(shape=(len(vocab), argsize), dtype='float32')

	for i, w in vocab.items():

		if w not in doc2vec_vocab1:
			continue
		embedding1[i, :] = doc2vec_weights1[doc2vec_vocab1[w], :]


		if w not in doc2vec_vocab2:
			continue
		embedding2[i, :] = doc2vec_weights2[doc2vec_vocab2[w], :]

		if w not in doc2vec_vocab3:
			continue
		embedding3[i, :] = doc2vec_weights3[doc2vec_vocab3[w], :]
	
	
	return model1, model2, model3, embedding1, embedding2, embedding3


################################################
# generating sentence-level / document embedding by averaging word2vec
# document here is sentence - or sequence of words
################################################
def averageWE(word2vec_model, documents):

	w2v_vocab = word2vec_model.wv.index2word
	w2v_weights = word2vec_model.wv.syn0
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
def averageIdfWE(word2vec_model, documents):

	w2v_vocab = word2vec_model.wv.index2word
	w2v_weights = word2vec_model.wv.syn0
	w2v = dict(zip(w2v_vocab, w2v_weights))
	dim = len(w2v.itervalues().next())

	# calculating Tf-Idf weights
	
	# transforming tokenized documents into string format - following the input format of TfidfVectorizer
	strSentences = sequenceToStr(documents)

	tfidf = TfidfVectorizer(analyzer=partial(nltk.regexp_tokenize, pattern=pattern), use_idf=True, smooth_idf= True, norm=None, stop_words=None)
	tfidf_matrix = tfidf.fit_transform(strSentences)
	# the resulting document term matrix (tf-idf doc-term matrix)
	arrTfIdf = tfidf_matrix.todense()

	# sklearn library use the following formula of idf
	# idf(term) = ( log ((1 + nd)/(1 + df(doc,term))) ) + 1
	# where nd : number of document in corpus; 
	# df : doc frequency (number of documents containing term)

	# use idf weights as the weight of word vectors
	idf = tfidf.idf_ 
	wordIdf = dict(zip(tfidf.get_feature_names(), idf))

	# if a word was never seen - it must be at least as infrequent
	# as any of the known words - so the default idf is the max of 
	# known idf's
	# max_idf = max(tfidf.idf_)

	doc_embedding = []
	for i,text in enumerate(strSentences):
		embedding = np.mean([w2v[w] * wordIdf[w]
				for w in text if w in w2v] or
				[np.zeros(dim)], axis=0)
		
		doc_embedding.append(embedding)

	return np.array(doc_embedding)


################################################
# LSTM sequential encoder - decoder for language model
# bi-lingual learning - translation task 
# with parallel aligned document input output 

# for single document with multiple sentences
# encoding sentences -> create sentence embedding
################################################
def seqEncDec(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM, embedding_weights):

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


################################################
# Encoder - decoder for language model
# train on short text monolingual data (single sentence)

################################################
def seqMonoEncDec(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	hidden_size = 200
	num_layers = 3

	model = Sequential()
	
	model.add(Embedding(VOCAB_LENGTH, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable = True, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))

	# Creating encoder network
	# encoding text input (sequence of words) into sentence embedding
	model.add(LSTM(hidden_size,name='lstm_enc_1'))
	model.add(RepeatVector(MAX_SEQUENCE_LENGTH))

	# Creating decoder network
	# objective function: predicting next words (language model)
	for i in range(num_layers):
		model.add(LSTM(hidden_size, name='lstm_%s'%(i+2), return_sequences=True))
	model.add(TimeDistributed(Dense(VOCAB_LENGTH,name='dense_output')))
	model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model


################################################
# Hierarchical LSTM encoder - decoder type 1
# with keras functional API vs. sequential model


# input array for this model is in 4D shape
# ( rows,       cols,     time_steps,    n_dim     )
# ( _____   ___________   _________   ____________ )
# ( n_docs, n_sentences,   n_words    dim_embedding)
################################################

def hierarchyTDEncDec(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	row_hidden_size = 200
	col_hidden_size = 200
	num_layers = 3


	x = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int64')
	# embedding layer
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(x)
	# Encoder model
	# Encodes sentences
	encoded_sentences = TimeDistributed(LSTM(row_hidden_size, name='lstm_enc_1'))(embedded_sentences)
	
	# Encodes documents
	encoded_docs = LSTM(col_hidden_size, name='lstm_enc_2')(encoded_sentences)

	# Decoder model
	encoder_output = RepeatVector(MAX_SEQUENCE_LENGTH,name='encoder_repeat')(encoded_docs)

	# Creating decoder network
	# objective function: predicting next words (language model)
	for i in range(num_layers):
		decoder_layer = LSTM(col_hidden_size, return_sequences=True,name='lstm_dec_%s'%(i+2))(encoder_output)

	decoder = TimeDistributed(Dense(VOCAB_LENGTH, activation='softmax', name='dense_output'))(decoder_layer)

	language_model = Model(x, decoder)

	language_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(language_model.summary())

	return language_model


def hierarchyTDClassifier(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights,num_classes):

	row_hidden_size = 200
	col_hidden_size = 200

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'


	x = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int64')
	# embedding layer
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(x)
	# Encodes sentences
	encoded_sentences = TimeDistributed(LSTM(row_hidden_size, name='lstm_enc_1'))(embedded_sentences)
	
	# Encodes documents
	encoded_docs = LSTM(col_hidden_size, name='lstm_enc_2')(encoded_sentences)

	# Final predictions and model.
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(encoded_docs)
	model = Model(x, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

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

	# encoding cols (documents)
	model.add(LSTM(col_hidden_size,name='lstm_enc_2'))
	model.add(Dense(num_classes, activation='softmax', name='dense_out'))
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])

	print(model.summary())

	return model


################################################
# Hierarchical LSTM encoder - decoder type 2
# with keras functional API vs. sequential model



# input array for this model is in 4D shape
# ( cols,       rows,     time_steps,    n_dim     )
# ( _____   ___________   _________   ____________ )
# ( n_docs, n_sentences,   n_words    dim_embedding)
################################################
def hierarchyClassifier(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights, num_classes):

	n_docs = 1
	row_hidden_size = 200
	col_hidden_size = 200

	if num_classes > 2:
		loss_function = 'categorical_crossentropy'
	else:
		loss_function = 'binary_crossentropy'

	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
	embedded_sentences = Embedding(VOCAB_LENGTH, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, mask_zero=True, name='embedding_layer')(sentences_input)
	lstm_sentence = LSTM(row_hidden_size,name='lstm_enc_1')(embedded_sentence)
	sentences_model = Model(sentences_input, lstm_sentence)

	docs_input = Input(shape=(n_docs, MAX_SEQUENCE_LENGTH), dtype='int64')
	docs_encoded = TimeDistributed(sentences_model)(docs_input)
	#dropout = Dropout(0.2)(docs_encoded)
	docs_model = LSTM(col_hidden_size,name='lstm_enc_2')(docs_encoded)

	# Prediction
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(docs_model)
	model = Model(docs_input, prediction)
	model.compile(loss=loss_function, optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())


	return model

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





	
