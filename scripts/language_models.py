# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "14.06.2017"
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
from text_preprocessing import *


def wordEmbedding(documents, vocab, argsize, argiter):

    
    print('Training word2vec model...')

    #`sg` defines the training algorithm. By default (`sg=0`), CBOW is used. Otherwise (`sg=1`), skip-gram is employed.
    #`min_count` = ignore all words with total frequency lower than this
    #`max_vocab_size` = limit RAM during vocabulary building; if there are more unique words than this, then prune the infrequent ones. Every 10 million word types
    # need about 1GB of RAM. Set to `None` for no limit (default).
    #`workers` = use this many worker threads to train the model (=faster training with multicore machines).
    #`hs` = if 1, hierarchical softmax will be used for model training. If set to 0 (default), and `negative` is non-zero, negative sampling will be used.
    #`negative` = if > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20).
    # Default is 5. If set to 0, no negative samping is used.
    

    model = Word2Vec(documents, size=argsize, min_count=0, window=5, sg=1, hs=1, negative=5, iter=argiter)
    weights = model.wv.syn0
    word2vec_vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    embedding = np.zeros(shape=(len(vocab), argsize), dtype='float32')

    for i, w in vocab.items():

        if w not in word2vec_vocab:
            continue
        embedding[i, :] = weights[word2vec_vocab[w], :]

    savePickle(embedding,'embedding')
    # alternative - saving as h5 file
    saveH5File('embedding.h5','embedding',embedding)

    # also save model
    model.save('word2vec_model')
    savePickle(model,'model_pickle')

    # also vocab and weights from word2vec
    savePickle(word2vec_vocab,'word2vec_vocab')
    saveH5Dict('word2vec_vocab.h5',word2vec_vocab)

    savePickle(weights,'word2vec_weights')
    saveH5File('word2vec_weights.h5','word2vec_weights',weights)



    return model, embedding, d, weights

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

	
