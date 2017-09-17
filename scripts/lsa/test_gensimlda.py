# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "31.08.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"


## this code is for testing lda function from gensim library


from __future__ import division, print_function

import sys
import numpy as np
from text_preprocessing import *
from vector_space import VectorSpace
from tokenizer import Tokenizer
from tfidf import TFIDF
#import lda
import gensim 


if __name__ == '__main__':


	doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father."
	doc2 = "My father spends a lot of time driving my sister around to dance practice."
	doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
	doc4 = "Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better."
	doc5 = "Health experts say that Sugar is not good for your lifestyle."

	example = [doc1, doc2, doc3, doc4, doc5]

	titles = []
	for i in range(len(example)):
		titles.append('doc_%s' %i)

	tokenized_docs = []
	docwords = []

	tokenizer = Tokenizer()


	for text in example:
		
		tmp = tokenizer.tokenise_and_remove_stop_words(text)
		tokenized_docs.append(tmp)

	docwords = sum(tokenized_docs,[])

	vocab = tokenizer._vocab(docwords)


	# save data
	#savePickle(tokenized_docs,'tokenized_docs')
	#savePickle(vocab,'vocab')

	vs = VectorSpace(tokenized_docs,vocab)

	
	# save matrices: BOW, sub-linear BOW, 
	#savePickle(vs.td_bow,'td_bow')
	#savePickle(vs.td_bow_sublin,'td_bow_sublin')


	tfidf = TFIDF(vs.td_bow)
	td_tfidf = tfidf.transform()

	# save tf-idf matrix transformed from BOW 
	#savePickle(td_tfidf,'td_tfidf')

	td_bow = np.array(vs.td_bow)
	td_sublinbow = np.array(vs.td_bow_sublin)

	# need to transform standard matrix into (term ID, term frequency) pairs as such document vector for doc_0 is  [(0, 2), (1, 1), (2, 2), (3, 2), (4, 1), (5, 1)] 

	# for BOW term-document matrix
	ndocs, nwords = td_bow.shape
	bow_data = []
	for i in range(ndocs):
		tmp1 = []
		
		for j in range(nwords):
			tmp2 = (j,td_bow[i][j])
			tmp1.append(tmp2)

		bow_data.append(tmp1)

	# for sublinear BOW term-document matrix

	ndocs, nwords = td_sublinbow.shape
	sublinbow_data = []
	for i in range(ndocs):
		tmp1 = []
		
		for j in range(nwords):
			tmp2 = (j,td_bow[i][j])
			tmp1.append(tmp2)

		sublinbow_data.append(tmp1)

	# for TF IDF term-document matrix

	ndocs, nwords = td_tfidf.shape
	tfidf_data = []
	for i in range(ndocs):
		tmp1 = []
		
		for j in range(nwords):
			tmp2 = (j,td_bow[i][j])
			tmp1.append(tmp2)

		tfidf_data.append(tmp1)

	# vocab is in dictionary format pair of (key, values)

	'''
	chunksize = (how many documents to load into memory) is decoupled from LDA batch size
	can process the training corpus with chunksize=10000, but with update_every=2, the maximization step of EM is done once every 2*10000=20000 documents. 
	This is the same as chunksize=20000, but uses less memory. 

	By default, update_every=1, so that the update happens after each batch of `chunksize` documents. 

	For example, if the training corpus has 50,000 documents, chunksize is 
	10,000, passes is 2, then online training is done in 10 updates: 

	#1 documents 0-9,999 
	#2 documents 10,000-19,999 
	#3 documents 20,000-29,999 
	#4 documents 30,000-39,999 
	#5 documents 40,000-49,999 
	#6 documents 0-9,999 
	#7 documents 10,000-19,999 
	#8 documents 20,000-29,999 
	#9 documents 30,000-39,999 
	#10 documents 40,000-49,999 

	*) In case running the distributed version, be aware that `update_every` refers to one worker: 
	with chunksize=10000, update_every=1 and 4 nodes, the model update is done once every 10000*1*4=40000 documents. 

	using serial LDA version on this node 
	running online LDA training, 100 topics, 1 passes over the supplied 
	corpus of 3931787 documents, updating model once every 10000 documents 
	`passes` is the number of training passes through the corpus. 
	
	

	'''

	num_topics = 3
	num_words = nwords
	num_docs = ndocs

	lda_bow = gensim.models.ldamodel.LdaModel(corpus=bow_data, id2word=vocab, num_topics=num_topics, chunksize=num_docs, passes=10)

	

	List = ldamodel.print_topics(num_topics, num_words)
	Topic_words =[]
	for i in range(0,len(List)):
		word_list = re.sub(r'(.\....\*)|(\+ .\....\*)', '',List[i][1])
		temp = [word for word in word_list.split()]
		Topic_words.append(temp)
		print('Topic ' + str(i) + ': ' + '\n' + str(word_list))
		print('\n' + '-'*100 + '\n')