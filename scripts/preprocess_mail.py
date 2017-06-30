# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "28.06.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import os
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from text_preprocessing import *
from language_models import *
from keras.callbacks import Callback

PATH = '/opt/data/phishing_mails/content'

if __name__ == '__main__':

	# get list of data files
	filenames = listData(PATH)

	data={}

	for path in filenames:
		filepath = os.path.basename(path)
		fileName, fileExtension = os.path.splitext(filepath)
		fname, fdate = split_at(fileName, '_', 2)
		data[fname] = extractData(path)		
