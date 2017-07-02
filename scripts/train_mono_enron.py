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


PATH = 'prepdata/enron'


if __name__ == '__main__':

	# reading stored pre-processed (in pickle format)

	subject_vocab = readPickle(os.path.join(PATH,'enron_subjVocab'))
	mail_vocab = readPickle(os.path.join(PATH,'enron_contVocab'))
	allSubjects = readPickle(os.path.join(PATH, 'allSubjects'))
	allMails = readPickle(os.path.join(PATH,'allMails'))
	allNumSubjects = readPickle(os.path.join(PATH, 'allNumSubjects'))
	allNumMails = readPickle(os.path.join(PATH,'allNumMails'))

	

	## For mail subject (short text part of mail)
	#######################################################
	# create WE version of subject using gensim word2vec model
	# put all subjects into one single array list, with separated class label array


	classLabel=[]
	subjSentences = []
	for i in allSubjects:
		nclass = len(allSubjects[i])
		for _j in range(nclass):
			classLabel.append(i)
		subjSentences += allSubjects[i]

	#savePickle(subjSentences,'lingspam_subjSentences')

	subjNumSentences = []
	for i in allSubjects:
		subjNumSentences += allNumSubjects[i]

	#savePickle(subjNumSentences,'lingspam_subjNumSentences')

	

	# word2vec model of mail subjects
	w2v_subj_enr1, w2v_subj_enr2, w2v_subjenr_embed1, w2v_subjenr_embed2 = wordEmbedding(subjSentences, subject_vocab, 200, 50)

	w2v_subj_enr1.save('w2v_subj_enr1')
	w2v_subj_enr2.save('w2v_subj_enr2')
	savePickle(w2v_subjenr_embed1,'w2v_subjenr_embed1')
	savePickle(w2v_subjenr_embed2,'w2v_subjenr_embed2')


	# create document representation of word vectors

	# By averaging word vectors
	avg_subjenr_embed1 = averageWE(w2v_subj_enr1, subjSentences)
	avg_subjenr_embed2 = averageWE(w2v_subj_enr2, subjSentences)

	savePickle(avg_subjenr_embed1,'avg_subjenr_embed1')
	savePickle(avg_subjenr_embed2,'avg_subjenr_embed2')


	# By averaging and idf weights of word vectors
	avgIDF_subjenr_embed1 = averageIdfWE(w2v_subj_enr1, subjSentences)
	avgIDF_subjenr_embed2 = averageIdfWE(w2v_subj_enr2, subjSentences)

	savePickle(avgIDF_subjenr_embed1,'avgIDF_subjenr_embed1')
	savePickle(avgIDF_subjenr_embed2,'avgIDF_subjenr_embed2')



	# doc2vec model of mail subject
	# labelling sentences with tag sent_id - since gensim doc2vec has different format of input as follows:
	# sentences = [
	#             labelledSentences(words=[u're', u':', u'2', u'.', u'882', u's', u'-', u'>', u'np', u'np'], tags=['sent_0']),
	#             labelledSentences(words=[u'job', u'-', u'university', u'of', u'utah'], tags=['sent_1']),
	#             ...
	#             ]

	# sentences here can also be considered as document
	# for document with > 1 sentence, the input is the sequence of words in document
	labelledSentences = createLabelledSentences(subjSentences)

	# doc2vec model
	d2v_subj_enr1, d2v_subj_enr2, d2v_subj_enr3, d2v_subj_enr_embed1, d2v_subj_enr_embed2, d2v_subj_enr_embed3 = docEmbedding(labelledSentences, subject_vocab, 200, 50)

	d2v_subj_enr1.save('d2v_subj_enr1')
	d2v_subj_enr2.save('d2v_subj_enr2')
	d2v_subj_enr3.save('d2v_subj_enr3')
	savePickle(d2v_subj_enr_embed1,'d2v_subj_enr_embed1')
	savePickle(d2v_subj_enr_embed2,'d2v_subj_enr_embed2')
	savePickle(d2v_subj_enr_embed3,'d2v_subj_enr_embed3')


	## For mail contents
	#######################################################
	



