# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "14.06.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
sys.path.insert(0,'..')
import numpy as np
from text_preprocessing import *
from language_models import *
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder



PATH = '/home/inimah/git/Neural-Language-Models/scripts/prepdata/spamassasin'


if __name__ == '__main__':


	# reading stored pre-processed (in pickle format)

	# Final vocabulary list after being reduced from less frequent words (links, noises)
	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))

	# Final tokenized documents with labels 
	subject = readPickle(os.path.join(PATH,'spamas_fin_labelled_subj'))


	tokenized_docs = []
	class_labels = []
	for i, data in enumerate(subject):
		class_labels.append(data[0])
		tokenized_docs.append(data[1])


	# doc2vec model of mail subject
	# labelling sentences with tag sent_id - since gensim doc2vec has different format of input as follows:
	# sentences = [
	#             labelledSentences(words=[u're', u':', u'2', u'.', u'882', u's', u'-', u'>', u'np', u'np'], tags=['sent_0']),
	#             labelledSentences(words=[u'job', u'-', u'university', u'of', u'utah'], tags=['sent_1']),
	#             ...
	#             ]

	# sentences here can also be considered as document
	# for document with > 1 sentence, the input is the sequence of words in document

	########################################################
	# Without sampling
	# without stopword removing and stemming
	
	labelledSubjSentences = createLabelledSentences(tokenized_docs)

	# doc2vec model
	d2v_subj_spamas1, d2v_subj_spamas2, d2v_subj_spamas3, d2v_subj_spamas_embed1, d2v_subj_spamas_embed2, d2v_subj_spamas_embed3, d2v_vocab1, d2v_vocab2, d2v_vocab3, d2v_wv1, d2v_wv2, d2v_wv3 = docEmbedding(labelledSubjSentences, subject_vocab, 50, 50)

	
	d2v_subj_spamas1.save('d2v_subj_spamas1')
	d2v_subj_spamas2.save('d2v_subj_spamas2')
	d2v_subj_spamas3.save('d2v_subj_spamas3')
	
	savePickle(d2v_subj_spamas_embed1,'d2v_subj_spamas_embed1')
	savePickle(d2v_subj_spamas_embed2,'d2v_subj_spamas_embed2')
	savePickle(d2v_subj_spamas_embed3,'d2v_subj_spamas_embed3')

	savePickle(d2v_vocab1,'d2v_vocab1')
	savePickle(d2v_vocab2,'d2v_vocab2')
	savePickle(d2v_vocab3,'d2v_vocab3')

	savePickle(d2v_wv1,'d2v_wv1')
	savePickle(d2v_wv2,'d2v_wv2')
	savePickle(d2v_wv3,'d2v_wv3')
