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

	# specifically for subject title part (short text)
	# create WE version of subject
	# first, put all subjects into one single document

	allSentences = []
	for i in allSubjects:
		allSentences += allSubjects[i]


	# for training on pre-processed data
	# variable "allSentences" here is in numeric format - different with the resulting from reading raw data above
	wordSentences = []
	for i in range(len(allSentences)):
		wordSentences += [indexToWords(subject_vocab,allSentences[i])]



	model, embedding, d, weights = wordEmbedding(wordSentences, subject_vocab, 200, 50)

	# create doc embedding for mail content




#   splitting training, validation, test sets (in dictionary format)
#	train, validate, test = splitDataDict(numericdict, train_percent=.6, validate_percent=.2, seed=None)
	

#	xTrain, yTrain = dictToArray(train)
#	xValidate, yValidate = dictToArray(validate)
#	xTest, yTest = dictToArray(test)