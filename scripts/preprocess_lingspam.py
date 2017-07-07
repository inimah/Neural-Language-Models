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



LINGSPAM_PATH = '../data/lingspam/raw'


if __name__ == '__main__':
	# get list of data files
	filenames = listData(LINGSPAM_PATH)
	# grouped by class
	datadict = getClassLabel(filenames)

	# return tokenized subject and mail content 
	subjVocab, contVocab, subject, content, numSubj, numCont = generateMailVocab(datadict)
	# save vocabulary list
	savePickle(subjVocab,'lingspam_subjVocab')
	savePickle(contVocab,'lingspam_contVocab')
	savePickle(subject,'allSubjects')
	savePickle(numSubj,'allNumSubjects')
	savePickle(content,'allMails')
	savePickle(numCont,'allNumMails')

	## For mail subject (short text part of mail)
	#######################################################
	# create WE version of subject using gensim word2vec model
	# put all subjects into one single array list, with separated class label array

	classLabel=[]
	subjSentences = []
	for i in subject:
		nclass = len(subject[i])
		for _j in range(nclass):
			classLabel.append(i)
		subjSentences += subject[i]

	savePickle(subjSentences,'ls_subjSentences')
	savePickle(classLabel,'ls_classLabel')

	subjNumSentences = []
	for i in subject:
		subjNumSentences += numSubj[i]

	savePickle(subjNumSentences,'ls_subjNumSentences')


	## For mail contents
	#######################################################

	
	mailSentences = []
	for i in content:
		mailSentences += content[i]

	savePickle(mailSentences,'ls_mailSentences')

	mailNumSentences = []
	for i in numCont:
		mailNumSentences += numCont[i]

	savePickle(mailNumSentences,'ls_mailNumSentences')


	

