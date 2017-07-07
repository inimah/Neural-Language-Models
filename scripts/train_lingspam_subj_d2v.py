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
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-batch_size', type=int, default=100)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
args = vars(ap.parse_args())


BATCH_SIZE = args['batch_size']
NB_EPOCH = args['nb_epoch']
MODE = args['mode']


PATH = 'prepdata/lingspam'
EMBED_PATH = 'embedding/lingspam'


if __name__ == '__main__':


	class TrainingHistory(Callback):
		
		def on_train_begin(self, logs={}):
			self.losses = []
			self.acc = []
			self.i = 0
			self.save_every = 50
		def on_batch_end(self, batch, logs={}):
			self.losses.append(logs.get('loss'))
			self.acc.append(logs.get('acc'))
			self.i += 1
		
	history = TrainingHistory()

	# reading stored pre-processed (in pickle format)

	subject_vocab = readPickle(os.path.join(PATH,'lingspam_subjVocab'))
	mail_vocab = readPickle(os.path.join(PATH,'lingspam_contVocab'))
	allSubjects = readPickle(os.path.join(PATH, 'allSubjects'))
	allMails = readPickle(os.path.join(PATH,'allMails'))
	allNumSubjects = readPickle(os.path.join(PATH, 'allNumSubjects'))
	allNumMails = readPickle(os.path.join(PATH,'allNumMails'))

	ls_classLabel = readPickle(os.path.join(PATH,'ls_classLabel'))
	binEncoder = LabelEncoder()
	binEncoder.fit(ls_classLabel)
	yEncoded = binEncoder.transform(ls_classLabel)

	subjSentences = readPickle(os.path.join(PATH,'ls_subjSentences'))
	numSentences = readPickle(os.path.join(PATH,'ls_subjNumSentences'))

	# doc2vec model of mail subject
	# labelling sentences with tag sent_id - since gensim doc2vec has different format of input as follows:
	# sentences = [
	#             labelledSentences(words=[u're', u':', u'2', u'.', u'882', u's', u'-', u'>', u'np', u'np'], tags=['sent_0']),
	#             labelledSentences(words=[u'job', u'-', u'university', u'of', u'utah'], tags=['sent_1']),
	#             ...
	#             ]

	# sentences here can also be considered as document
	# for document with > 1 sentence, the input is the sequence of words in document
	
	labelledSubjSentences = createLabelledSentences(subjSentences)

	# doc2vec model
	d2v_subj_ls1, d2v_subj_ls2, d2v_subj_ls3, d2v_subj_ls_embed1, d2v_subj_ls_embed2, d2v_subj_ls_embed3 = docEmbedding(labelledSubjSentences, subject_vocab, 200, 50)

	
	d2v_subj_ls1.save('d2v_subj_ls1')
	d2v_subj_ls2.save('d2v_subj_ls2')
	d2v_subj_ls3.save('d2v_subj_ls3')
	savePickle(d2v_subj_ls_embed1,'d2v_subj_ls_embed1')
	savePickle(d2v_subj_ls_embed2,'d2v_subj_ls_embed2')
	savePickle(d2v_subj_ls_embed3,'d2v_subj_ls_embed3')
