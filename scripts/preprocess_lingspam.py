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
	subjVocab, contVocab, subject, content, numSubj, numCont = generateLingSpam(datadict)
	# save vocabulary list
	savePickle(subjVocab,'lingspam_subjVocab')
	savePickle(contVocab,'lingspam_contVocab')
	savePickle(subject,'allSubjects')
	savePickle(numSubj,'allNumSubjects')
	savePickle(content,'allMails')
	savePickle(numCont,'allNumMails')

    '''
	# specifically for the content of mail
	# return splitted sentences in a form of tokenized words 
	sentencesMail = getSentencesClass(allCont)
	# try printing a sample of sentence
	# printSentencesClass(0, 0, 0, sentences)

	# transform to numerical sentences
	numSentencesMail = sentToNum(sentencesMail,contVocab)

	# check minimum and maximum length of sequence words - dictionary is in numeric format
	minlength, maxlength, avglength=getMinMaxAvgLength(allSubj)

    # specifically for subject title part (short text)
    # create WE version of subject
    # first, put all subjects into one single document
    allSentences = []
    for i in allSubj:
    	allSentences += allSubj[i]


    # for training on pre-processed data
    # variable "allSentences" here is in numeric format - different with the resulting from reading raw data above
    wordSentences = []
    for i in range(len(allSentences)):
    	wordSentences += [indexToWords(subject_vocab,allSentences[i])]
    '''

