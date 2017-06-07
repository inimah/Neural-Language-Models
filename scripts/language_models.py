# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "06.06.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import numpy as np
from gensim.models import Word2Vec, Doc2Vec
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing import sequence


def wordEmbedding(documents, vocab, argsize, argiter):
    # use skipgram model
    model = Word2Vec(documents, size=argsize, min_count=5, window=5, sg=1, iter=argiter)
    weights = model.wv.syn0
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    embedding = np.zeros(shape=(len(vocab)+1, argsize), dtype='float32')

    for i, w in vocab.items():
        if w not in d:continue
        embedding[i, :] = weights[d[w], :]
    savePickle(embedding,'embedding')
    # alternative - saving as h5 file
    saveH5File('embedding.h5','embedding',embedding)
    return embedding

## to-be-added
def docEmbedding(documents, vocab, argsize, argiter):

    model = Doc2Vec(size=argsize, window=10, min_count=5, workers=10, alpha=0.025, min_alpha=0.025)
    embedding = np.zeros(shape=(len(documents), argsize), dtype='float32')
    
    for epoch in range(argiter):
        model.train(documents)
        model.alpha -= 0.002  # decrease the learning rate
        model.min_alpha = model.alpha  # fix the learning rate, no decay
    
    return embedding


def seqEncoderDecoder(X_vocab_len, X_max_len, y_vocab_len, y_max_len, embedding_dim, hidden_size, num_layers):

    model = Sequential()
    # Creating encoder network
    model.add(Embedding(X_vocab_len, embedding_dim, input_length=X_max_len, mask_zero=True))
    model.add(LSTM(hidden_size))
    model.add(RepeatVector(y_max_len))

    # Creating decoder network
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(y_vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print(model.summary())
    return model

def seqBinClassifier(X_vocab_len,embedding_dim,X_max_len,hidden_size):

    model = Sequential()
    model.add(Embedding(X_vocab_len, embedding_dim, input_length=X_max_len, mask_zero=True))
    model.add(LSTM(hidden_size))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

	
