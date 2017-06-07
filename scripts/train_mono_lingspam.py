# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "20.05.2017"
#__version__ = "1.0.1"
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


LINGSPAM_PATH = 'data/maildata/lingspam'


if __name__ == '__main__':
	# get list of data files
	filenames = listData(LINGSPAM_PATH)
    # grouped by class
	datadict = getClassLabel(filenames)

	# return tokenized subject and mail content 
	subjVocab, contVocab, subjAllTokens, contentAllTokens, allSubj, allCont = generateLingSpam(datadict)

	# specifically for the content of mail
	# return splitted sentences in a form of tokenized words 
	sentencesMail = getSentencesClass(allCont)
	# try printing a sample of sentence
	# printSentencesClass(0, 0, 0, sentences)

	# transform to numerical sentences
	numSentencesMail = sentToNum(sentencesMail,contVocab)

    # specifically for subject title part (short text)
    # create WE version of subject
    subjWE = wordEmbedding(mergedTokens, subjVocab, 200, 50)

    # create doc embedding for mail content




#   splitting training, validation, test sets (in dictionary format)
#	train, validate, test = splitDataDict(numericdict, train_percent=.6, validate_percent=.2, seed=None)
	

#	xTrain, yTrain = dictToArray(train)
#	xValidate, yValidate = dictToArray(validate)
#	xTest, yTest = dictToArray(test)