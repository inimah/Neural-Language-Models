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
PATH2 = '/home/inimah/git/Neural-Language-Models/scripts/train_spamas/subj/d2v'


if __name__ == '__main__':


	# reading stored pre-processed (in pickle format)

	# Vocabulary
	subject_vocab = readPickle(os.path.join(PATH,'spamas_reducedVocab'))

	'''
	subjectSW_vocab = readPickle(os.path.join(PATH,'spamas_reducedvocabSW'))

	
	########################################################
	# Without sampling

	tokenized_docs = readPickle(os.path.join(PATH2,'spamas_w2v_tokenized_docs'))
	class_labels = readPickle(os.path.join(PATH2,'spamas_w2v_class_labels'))


	# For tokenized document with Stopword elimination and stemming
	tokenized_docsSW = readPickle(os.path.join(PATH2,'spamas_w2v_tokenized_docsSW'))
	class_labelsSW = readPickle(os.path.join(PATH2,'spamas_w2v_class_labelsSW'))


	########################################################
	# With sampling

	tokenized_docs500 = readPickle(os.path.join(PATH2,'spamas_w2v_tokenized_docs500'))
	class_labels500 = readPickle(os.path.join(PATH2,'spamas_w2v_class_labels500'))


	# For tokenized document with Stopword elimination and stemming
	tokenized_docsSW500 = readPickle(os.path.join(PATH2,'spamas_w2v_tokenized_docsSW500'))
	class_labelsSW500 = readPickle(os.path.join(PATH2,'spamas_w2v_class_labelsSW500'))
	

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
	d2v_subj_spamas1, d2v_subj_spamas2, d2v_subj_spamas3, d2v_subj_spamas_embed1, d2v_subj_spamas_embed2, d2v_subj_spamas_embed3 = docEmbedding(labelledSubjSentences, subject_vocab, 50, 50)

	
	d2v_subj_spamas1.save('d2v_subj_spamas1')
	d2v_subj_spamas2.save('d2v_subj_spamas2')
	d2v_subj_spamas3.save('d2v_subj_spamas3')
	savePickle(d2v_subj_spamas_embed1,'d2v_subj_spamas_embed1')
	savePickle(d2v_subj_spamas_embed2,'d2v_subj_spamas_embed2')
	savePickle(d2v_subj_spamas_embed3,'d2v_subj_spamas_embed3')

	# with stopword removing and stemming

	labelledSubjSentencesSW = createLabelledSentences(tokenized_docsSW)

	# doc2vec model
	d2v_subj_spamas1SW, d2v_subj_spamas2SW, d2v_subj_spamas3SW, d2v_subj_spamas_embed1SW, d2v_subj_spamas_embed2SW, d2v_subj_spamas_embed3SW = docEmbedding(labelledSubjSentencesSW, subjectSW_vocab, 50, 50)

	
	d2v_subj_spamas1SW.save('d2v_subj_spamas1SW')
	d2v_subj_spamas2SW.save('d2v_subj_spamas2SW')
	d2v_subj_spamas3SW.save('d2v_subj_spamas3SW')
	savePickle(d2v_subj_spamas_embed1SW,'d2v_subj_spamas_embed1SW')
	savePickle(d2v_subj_spamas_embed2SW,'d2v_subj_spamas_embed2SW')
	savePickle(d2v_subj_spamas_embed3SW,'d2v_subj_spamas_embed3SW')


	########################################################
	# With sampling
	# without stopword removing and stemming
	
	labelledSubjSentences_500 = createLabelledSentences(tokenized_docs500)

	# doc2vec model
	d2v_subj_spamas1_500, d2v_subj_spamas2_500, d2v_subj_spamas3_500, d2v_subj_spamas_embed1_500, d2v_subj_spamas_embed2_500, d2v_subj_spamas_embed3_500 = docEmbedding(labelledSubjSentences_500, subject_vocab, 50, 50)

	
	d2v_subj_spamas1_500.save('d2v_subj_spamas1_500')
	d2v_subj_spamas2_500.save('d2v_subj_spamas2_500')
	d2v_subj_spamas3_500.save('d2v_subj_spamas3_500')
	savePickle(d2v_subj_spamas_embed1_500,'d2v_subj_spamas_embed1_500')
	savePickle(d2v_subj_spamas_embed2_500,'d2v_subj_spamas_embed2_500')
	savePickle(d2v_subj_spamas_embed3_500,'d2v_subj_spamas_embed3_500')

	# with stopword removing and stemming

	labelledSubjSentencesSW_500 = createLabelledSentences(tokenized_docsSW500)

	# doc2vec model
	d2v_subj_spamas1SW_500, d2v_subj_spamas2SW_500, d2v_subj_spamas3SW_500, d2v_subj_spamas_embed1SW_500, d2v_subj_spamas_embed2SW_500, d2v_subj_spamas_embed3SW_500 = docEmbedding(labelledSubjSentencesSW_500, subjectSW_vocab, 50, 50)

	
	d2v_subj_spamas1SW_500.save('d2v_subj_spamas1SW_500')
	d2v_subj_spamas2SW_500.save('d2v_subj_spamas2SW_500')
	d2v_subj_spamas3SW_500.save('d2v_subj_spamas3SW_500')
	savePickle(d2v_subj_spamas_embed1SW_500,'d2v_subj_spamas_embed1SW_500')
	savePickle(d2v_subj_spamas_embed2SW_500,'d2v_subj_spamas_embed2SW_500')
	savePickle(d2v_subj_spamas_embed3SW_500,'d2v_subj_spamas_embed3SW_500')

	'''


	labelled_subj = readPickle(os.path.join(PATH,'spamas_fin_labelled_subj'))
	tokenized_docs = []
	class_labels = []
	for i, data in enumerate(labelled_subj):
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
	d2v_subj_spamas1, d2v_subj_spamas2, d2v_subj_spamas3, d2v_subj_spamas_embed1, d2v_subj_spamas_embed2, d2v_subj_spamas_embed3 = docEmbedding(labelledSubjSentences, subject_vocab, 50, 50)

	
	d2v_subj_spamas1.save('d2v_subj_spamas1')
	d2v_subj_spamas2.save('d2v_subj_spamas2')
	d2v_subj_spamas3.save('d2v_subj_spamas3')
	savePickle(d2v_subj_spamas_embed1,'d2v_subj_spamas_embed1')
	savePickle(d2v_subj_spamas_embed2,'d2v_subj_spamas_embed2')
	savePickle(d2v_subj_spamas_embed3,'d2v_subj_spamas_embed3')
