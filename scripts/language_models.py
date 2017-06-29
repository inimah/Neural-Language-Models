# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "14.06.2017"
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

	for epoch in range(argiter):
		model1.train(documents)
		model2.train(documents)

	word2vec_weights1 = model1.wv.syn0
	word2vec_weights2 = model2.wv.syn0    

	for i, w in vocab.items():

		if w not in word2vec_vocab1:
			continue
		embedding1[i, :] = word2vec_weights1[word2vec_vocab1[w], :]

		if w not in word2vec_vocab2:
			continue
		embedding2[i, :] = word2vec_weights2[word2vec_vocab2[w], :]

	savePickle(embedding1,'w2v_embedding1')
	# alternative - saving as h5 file
	saveH5File('w2v_embedding1.hdf5','w2v_embedding1',embedding1)

	savePickle(embedding2,'w2v_embedding2')
	# alternative - saving as h5 file
	saveH5File('w2v_embedding2.hdf5','w2v_embedding2',embedding2)

	# save model
	model1.save('word2vec_model1')
	savePickle(model1,'w2v_model1_pickle')

	model2.save('word2vec_model2')
	savePickle(model2,'w2v_model2_pickle')

	# save vocab built from word2vec model
	savePickle(word2vec_vocab1,'word2vec_vocab1')
	saveH5Dict('word2vec_vocab1.hdf5',word2vec_vocab1)

	savePickle(word2vec_vocab2,'word2vec_vocab2')
	saveH5Dict('word2vec_vocab2.hdf5',word2vec_vocab2)


	# save original weights from word2vec model 
	savePickle(word2vec_weights1,'word2vec_weights1')
	saveH5File('word2vec_weights1.hdf5','word2vec_weights1',word2vec_weights1)

	savePickle(word2vec_weights2,'word2vec_weights2')
	saveH5File('word2vec_weights2.hdf5','word2vec_weights2',word2vec_weights2)


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

	for epoch in range(argiter):
		model1.train(documents)
		model2.train(documents)
		model3.train(documents)


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
	


	model1.save('doc2vec_model1')
	model2.save('doc2vec_model2')
	model3.save('doc2vec_model3')

	savePickle(embedding1,'doc2vec_embedding1')
	# alternative - saving as h5 file
	saveH5File('doc2vec_embedding1.hdf5','doc2vec_embedding1',embedding1)

	savePickle(embedding2,'doc2vec_embedding2')
	# alternative - saving as h5 file
	saveH5File('doc2vec_embedding2.hdf5','doc2vec_embedding2',embedding2)

	savePickle(embedding3,'doc2vec_embedding3')
	# alternative - saving as h5 file
	saveH5File('doc2vec_embedding3.hdf5','doc2vec_embedding3',embedding3)


	
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
	
	model.add(Embedding(X_vocab_len+1, EMBEDDING_DIM, input_length=X_max_len, mask_zero=True, weights=[embedding_weights], name='embedding_layer'))
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


def seqEncDec2(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM):

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
	encoder_a.add(Embedding(X_vocab_len+1, EMBEDDING_DIM, input_length=X_max_len, mask_zero=True, weights=[X_embedding_weights], name='X_embedding_layer'))
	encoder_a.add(LSTM(hidden_size,name='lstm_a'))

	encoder_b = Sequential()
	encoder_b.add(Embedding(y_vocab_len+1, EMBEDDING_DIM, input_length=y_max_len, mask_zero=True, weights=[y_embedding_weights], name='y_embedding_layer'))
	encoder_b.add(LSTM(hidden_size,name='lstm_b'))

	decoder = Sequential()
	decoder.add(Merge([encoder_a, encoder_b], mode='concat'))
	decoder.add(Dense(hidden_size, activation='relu'))
	decoder.add(Dense(nb_classes, activation='softmax'))

	decoder.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	print(decoder.summary())


	return decoder

'''
def seqSharedEnc(X_vocab_len, X_max_len, y_vocab_len, y_max_len, EMBEDDING_DIM):

	hidden_size = 200
	nb_classes = 2

	encoder = Sequential()
	encoder.add(Embedding(X_vocab_len+1, EMBEDDING_DIM, input_length=X_max_len, mask_zero=True, weights=[joint_embedding], name='shared_embedding'))
	encoder.add(LSTM(hidden_size, name='shared_lstm'))

	model = Graph()
	model.add_input(name='input_a', input_shape=(X_max_len, EMBEDDING_DIM))
	model.add_input(name='input_b', input_shape=(y_max_len, EMBEDDING_DIM))
	model.add_shared_node(encoder, name='shared_encoder', inputs=['input_a', 'input_b'], merge_mode='concat')
	model.add_node(Dense(hidden_size, activation='relu'), name='fc1', input='shared_encoder')
	model.add_node(Dense(nb_classes, activation='softmax'), name='output', input='fc1', create_output=True)

	model.compile(optimizer='adam', loss={'output': 'categorical_crossentropy'})

	print(model.summary())

	return model
'''
################################################
# Hierarchical LSTM encoder - decoder type 1
# with keras functional API vs. sequential model


# input array for this model is in 4D shape
# ( rows,       cols,     time_steps,    n_dim     )
# ( _____   ___________   _________   ____________ )
# ( n_docs, n_sentences,   n_words    dim_embedding)
################################################
def apiHierarchical1(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	row_hidden_size = 200
	col_hidden_size = 200
	num_classes = 2

	x = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int64')
	# embedding layer
	embedded_sentences = Embedding(VOCAB_LENGTH + 1, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, name='embedding_layer')(x)
	# Encodes sentences
	encoded_sentences = TimeDistributed(LSTM(row_hidden_size, name='lstm_enc_1'))(embedded_sentences)
	#encoded_sentences = LSTM(row_hidden_size, name='lstm_enc_1', return_sequences=True)(embedded_sentences)
	# Encodes documents
	encoded_docs = LSTM(col_hidden_size, name='lstm_enc_2')(encoded_sentences)

	# Final predictions and model.
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(encoded_docs)
	model = Model(x, prediction)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())


	return model

def seqHierarchical1(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	row_hidden_size = 200
	col_hidden_size = 200
	num_classes = 2

	model = Sequential()	
	# input captured here is sequence of sentences in shape (rows, time_steps, n_dim)
	model.add(Embedding(VOCAB_LENGTH + 1, EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,), mask_zero=True, weights=[embedding_weights], name='embedding_layer'))
	# encoding rows (sentences)
	model.add(TimeDistributed(LSTM(row_hidden_size,name='lstm_enc_1')))
	#model.add(LSTM(row_hidden_size,name='lstm_enc_1'))

	# encoding cols (documents)
	model.add(LSTM(col_hidden_size,name='lstm_enc_2'))
	model.add(Dense(num_classes, activation='softmax', name='dense_out'))
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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
def apiHierarchical2(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	n_docs = 1
	row_hidden_size = 200
	col_hidden_size = 200
	num_classes = 2

	sentences_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
	embedded_sentences = Embedding(VOCAB_LENGTH + 1, EMBEDDING_DIM, weights=[embedding_weights], trainable = True, name='embedding_layer')(sentences_input)
	lstm_sentence = LSTM(row_hidden_size,name='lstm_enc_1')(embedded_sentence)
	sentences_model = Model(sentences_input, lstm_sentence)

	docs_input = Input(shape=(n_docs, MAX_SEQUENCE_LENGTH), dtype='int64')
	docs_encoded = TimeDistributed(sentences_model)(docs_input)
	#dropout = Dropout(0.2)(docs_encoded)
	docs_model = LSTM(col_hidden_size,name='lstm_enc_2')(docs_encoded)

	# Prediction
	prediction = Dense(num_classes, activation='softmax', name='dense_out')(docs_model)
	model = Model(docs_input, prediction)
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())


	return model

def seqHierarchical2(MAX_SEQUENCE_LENGTH, VOCAB_LENGTH, EMBEDDING_DIM, embedding_weights):

	n_docs = 1
	row_hidden_size = 200
	col_hidden_size = 200
	num_classes = 2

	sentences_model = Sequential()
	# input captured here is sequence of sentences in shape (rows, time_steps, n_dim)
	sentences_model.add(Embedding(VOCAB_LENGTH + 1, EMBEDDING_DIM, input_shape=(MAX_SEQUENCE_LENGTH,), mask_zero=True, weights=[embedding_weights], name='embedding_layer'))
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
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	print(model.summary())

	return model





	
