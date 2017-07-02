# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "20.05.2017"
#__update__ = "14.06.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import numpy as np
from text_preprocessing import *



ENRON_PATH = '~/exp/data/maildata/enron/preprocessed'


if __name__ == '__main__':
	# get list of data files
	filenames = listData(ENRON_PATH)
	# grouped by class
	datadict = getClassLabel(filenames)

	# return tokenized subject and mail content 
	subjVocab, contVocab, subject, content, numSubj, numCont = generateMailVocab(datadict)
	# save vocabulary list
	savePickle(subjVocab,'enron_subjVocab')
	savePickle(contVocab,'enron_contVocab')
	savePickle(subject,'allSubjects')
	savePickle(numSubj,'allNumSubjects')
	savePickle(content,'allMails')
	savePickle(numCont,'allNumMails')

