# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "14.06.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import numpy as np
from text_preprocessing import *
from language_models import *

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-batch_size', type=int, default=100)
ap.add_argument('-layer_num', type=int, default=3)
ap.add_argument('-hidden_dim', type=int, default=200)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
args = vars(ap.parse_args())


BATCH_SIZE = args['batch_size']
LAYER_NUM = args['layer_num']
HIDDEN_DIM = args['hidden_dim']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']


LINGSPAM_PATH = '../data/lingspam'


if __name__ == '__main__':

	# reading stored pre-processed (in pickle format)

	subject_vocab = readPickle(os.path.join(LINGSPAM_PATH,'subject_vocabulary'))
	mail_vocab = readPickle(os.path.join(LINGSPAM_PATH,'mail_vocabulary'))
	allSubjects = readPickle(os.path.join(LINGSPAM_PATH, 'allSubjects'))
	allMails = readPickle(os.path.join(LINGSPAM_PATH,'allMails'))

	## For mail subject (short text part of mail)
	#######################################################
	# create WE version of subject using gensim word2vec model
	# put all subjects into one single document

	# this sentence/document list is in numerical format

	subjNumSentences = []
	for i in allSubjects:
		subjNumSentences += allSubjects[i]


	# revert numerical format of sentence / document list into sequence of words format

	subjSentences = []
	for i in range(len(subjNumSentences)):
		subjSentences += [indexToWords(subject_vocab,subjNumSentences[i])]

	# word2vec model of mail subjects
	model1, model2, embedding1, embedding2 = wordEmbedding(subjSentences, subject_vocab, 200, 50)


	# create document representation of word vectors

	# By averaging word vectors

	

	# By averaging Tf-Idf of word vectors 

	# doc2vec model of mail subject
	# labelling sentences with tag sent_id - since gensim doc2vec has different format of input as follows:
    # sentences = [
    #             TaggedDocument(words=[u're', u':', u'2', u'.', u'882', u's', u'-', u'>', u'np', u'np'], tags=['sent_0']),
    #             TaggedDocument(words=[u'job', u'-', u'university', u'of', u'utah'], tags=['sent_1']),
    #             ...
    #             ]

    # sentences here can also be considered as document
    # for document with > 1 sentence, the input is the sequence of words in document
    labelledSentences = createLabelledSentences(subjSentences)


    # doc2vec model
    model1, model2, model3, embedding1, embedding2, embedding3 = docEmbedding(labelledSentences)


	## For mail contents
	#######################################################



