# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "31.08.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function

import sys
import numpy as np
from text_preprocessing import *
from vector_space import VectorSpace
from tfidf import TFIDF
from lsa import LSA
from tokenizer import Tokenizer
from lsa import LDA


def main():
	
	import optparse
	import vocabulary
	parser = optparse.OptionParser()
	parser.add_option("-f", dest="filename", help="corpus filename")
	parser.add_option("-c", dest="corpus", help="using range of Brown corpus' files(start:end)")
	parser.add_option("--alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
	parser.add_option("--beta", dest="beta", type="float", help="parameter beta", default=0.5)
	parser.add_option("-k", dest="K", type="int", help="number of topics", default=20)
	parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
	parser.add_option("-s", dest="smartinit", action="store_true", help="smart initialize of parameters", default=False)
	parser.add_option("--stopwords", dest="stopwords", help="exclude stop words", action="store_true", default=False)
	parser.add_option("--seed", dest="seed", type="int", help="random seed")
	parser.add_option("--df", dest="df", type="int", help="threshold of document freaquency to cut words", default=0)
	(options, args) = parser.parse_args()
	if not (options.filename or options.corpus): parser.error("need corpus filename(-f) or corpus range(-c)")

	if options.filename:
		corpus = vocabulary.load_file(options.filename)
	else:
		corpus = vocabulary.load_corpus(options.corpus)
		if not corpus: parser.error("corpus range(-c) forms 'start:end'")
	if options.seed != None:
		numpy.random.seed(options.seed)

	voca = vocabulary.Vocabulary(options.stopwords)
	docs = [voca.doc_to_ids(doc) for doc in corpus]
	if options.df > 0: docs = voca.cut_low_freq(docs, options.df)

	lda = LDA(options.K, options.alpha, options.beta, docs, voca.size(), options.smartinit)
	print ("corpus=%d, words=%d, K=%d, a=%f, b=%f" % (len(corpus), len(voca.vocas), options.K, options.alpha, options.beta))

	#import cProfile
	#cProfile.runctx('lda_learning(lda, options.iteration, voca)', globals(), locals(), 'lda.profile')
	lda_learning(lda, options.iteration, voca)

if __name__ == "__main__":
	main()