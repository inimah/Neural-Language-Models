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

    model1 = word2vec_models[0]
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



    savePickle(embedding1,'embedding1')
    # alternative - saving as h5 file
    saveH5File('embedding1.h5','embedding1',embedding1)

    savePickle(embedding2,'embedding2')
    # alternative - saving as h5 file
    saveH5File('embedding2.h5','embedding2',embedding2)

    # save model
    model1.save('word2vec_model1')
    savePickle(model1,'model1_pickle')

    model2.save('word2vec_model2')
    savePickle(model2,'model2_pickle')

    # save vocab built from word2vec model
    savePickle(word2vec_vocab1,'word2vec_vocab1')
    saveH5Dict('word2vec_vocab1.h5',word2vec_vocab1)

    savePickle(word2vec_vocab2,'word2vec_vocab2')
    saveH5Dict('word2vec_vocab2.h5',word2vec_vocab2)


    # save original weights from word2vec model 
    savePickle(word2vec_weights1,'word2vec_weights1')
    saveH5File('word2vec_weights1.h5','word2vec_weights1',word2vec_weights1)

    savePickle(word2vec_weights2,'word2vec_weights2')
    saveH5File('word2vec_weights2.h5','word2vec_weights2',word2vec_weights2)



    return model1, model2, embedding1, embedding2


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
    saveH5File('doc2vec_embedding1.h5','embedding1',embedding1)

    savePickle(embedding2,'doc2vec_embedding2')
    # alternative - saving as h5 file
    saveH5File('doc2vec_embedding2.h5','embedding2',embedding2)

    savePickle(embedding3,'doc2vec_embedding3')
    # alternative - saving as h5 file
    saveH5File('doc2vec_embedding3.h5','embedding3',embedding3)


    
    return model1, model2, model3, embedding1, embedding2, embedding3


def averageWE():

    return 0


def averageTfIdfWE():

    return 0 


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

	
