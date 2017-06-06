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

ENRON_PATH = 'data/maildata/enron'
LINGSPAM_PATH = 'data/maildata/lingspam'
SPAMASSASIN_PATH = 'data/maildata/spamassasin'
POLYGOT_PATH = 'data/multilingual/polygot'
EUROPARL1_PATH = 'data/multilingual/europarl/txt'
EUROPARL2_PATH = 'data/multilingual/europarl/nl-en'
TED_PATH = 'data/multilingual/ted'
TEST_PATH = 'data/multilingual/test'

if __name__ == '__main__':
	# get list of data files
	filenames = listData(EUROPARL2_PATH)
    # grouped by class
	datadict = getClassLabel(filenames)

	# return tokenized subject and mail content 
	tokens, worddocs_freq, vocab, alltokens, alldocs = generatePairset(datadict)

	