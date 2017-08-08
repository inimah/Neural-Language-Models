# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "01.08.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function

import os
import sys
import numpy as np
from text_preprocessing import *
from language_models import *


if __name__ == '__main__':

	#examples_1 = ["The cat in the hat disabled", "A cat is a fine pet ponies.", "Dogs and cats make good pets.","I haven't got a hat."]
	#tokenized_docs, vocab = generateSentVocab(examples_1)

	examples_2 = ["Machine learning is super fun", "Python is super, super cool", "Statistics is cool, too", "Data science is fun", "Python is great for machine learning", "I like football", "Football is great to watch"]
	tokenized_docs, vocab = generateSentVocab(examples_2)

	n_docs = len(examples_2)

	# word2vec model of mail subjects
	w2v_1, w2v_2, w2v_embed1, w2v_embed2 = wordEmbedding(tokenized_docs, vocab, n_docs, 10)

	w2v_1.save('w2v_1')
	w2v_2.save('w2v_2')
	savePickle(w2v_embed1,'w2v_embed1')
	savePickle(w2v_embed2,'w2v_embed2')


	# create document representation of word vectors

	# By averaging word vectors
	avg_embed1 = averageWE(w2v_1, tokenized_docs)
	avg_embed2 = averageWE(w2v_2, tokenized_docs)

	savePickle(avg_embed1,'avg_embed1')
	savePickle(avg_embed2,'avg_embed2')


	# By averaging and idf weights of word vectors
	avgIDF_embed1 = averageIdfWE(w2v_1, vocab, tokenized_docs)
	avgIDF_embed2 = averageIdfWE(w2v_2, vocab, tokenized_docs)

	savePickle(avgIDF_embed1,'avgIDF_embed1')
	savePickle(avgIDF_embed2,'avgIDF_embed2')




	
