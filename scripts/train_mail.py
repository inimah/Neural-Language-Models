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
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from text_preprocessing import *
from language_models import *
from keras.callbacks import Callback
from bs4 import BeautifulSoup
import gc



if __name__ == '__main__':

	subj_sent = readPickle('subject_sentences')
	subj_vocab = readPickle('subject_vocab')

	# word2vec model of mail subjects
	w2v_subj_sg, w2v_subj_cbow, w2v_subj_embed_sg, w2v_subj_embed_cbow = wordEmbedding(subj_sent, subj_vocab, 200, 50)

	w2v_subj_sg.save('w2v_subj_sg')
	w2v_subj_cbow.save('w2v_subj_cbow')

	savePickle(w2v_subj_embed_sg,'w2v_subj_embed_sg')
	savePickle(w2v_subj_embed_cbow,'w2v_subj_embed_cbow')


	# create document representation of word vectors (sentence embedding)

	# By averaging word vectors
	avg_subj_embed1 = averageWE(w2v_subj_sg, subj_sent)
	avg_subj_embed2 = averageWE(w2v_subj_cbow, subj_sent)

	savePickle(avg_subj_embed1,'avg_subj_embed1')
	savePickle(avg_subj_embed2,'avg_subj_embed2')

	
	w2v_subj_sg = readPickle('w2v_subj_sg')
	w2v_subj_cbow = readPickle('w2v_subj_cbow')
	
	# By averaging with IDF weights
	
	avgIDF_subj_embed1 = averageIdfWE(w2v_subj_sg, subj_sent)
	avgIDF_subj_embed2 = averageIdfWE(w2v_subj_cbow, subj_sent)

	savePickle(avgIDF_subj_embed1,'avgIDF_subj_embed1')
	savePickle(avgIDF_subj_embed2,'avgIDF_subj_embed2')

	'''

	# By doc2vec

	labelledSentences = createLabelledSentences(subj_sent)
	d2v_subj1, d2v_subj2, d2v_subj3, d2v_subj_embed1, d2v_subj_embed2, d2v_subj_embed3 = docEmbedding(labelledSentences, subj_vocab, 200, 50)

	savePickle(d2v_subj1,'d2v_subj1')
	savePickle(d2v_subj2,'d2v_subj2')
	savePickle(d2v_subj3,'d2v_subj3')
	savePickle(d2v_subj1,'d2v_subj_embed1')
	savePickle(d2v_subj2,'d2v_subj_embed2')
	savePickle(d2v_subj3,'d2v_subj_embed3')



	# By sequential model
	'''

	
