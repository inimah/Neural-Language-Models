# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "31.08.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

## this code is for testing lda function from python library

from __future__ import division, print_function

import sys
import numpy as np
from text_preprocessing import *
from vector_space import VectorSpace
from tokenizer import Tokenizer
from tfidf import TFIDF
import lda


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

	# default iteration = 1000
	model = lda.LDA(n_topics=3, n_iter=1000, random_state=1)
	model.fit(td_bow)

	n_topics=3
	n_vocab = len(vocab)
	n_docs = len(example)
	

	topic_word = model.topic_word_

	# most prominent probability of words in word-topic matrix
	max_word_prob = topic_word.max()

	
	for n in range(n_topics):
		sum_pr = sum(topic_word[n,:])
		print("topic: {} sum: {}".format(n, sum_pr))

	# print 10 most prominent words for each topic
	# sorting in ascending way
	n = 10
	for i, topic_dist in enumerate(topic_word):
		topic_words = np.array(vocab.values())[np.argsort(topic_dist)][:-(n+1):-1]
		print('*Topic {}\n- {}'.format(i, ' '.join(topic_words)))


	topic_doc = model.doc_topic_

	# most prominent probability
	max_topic_prob = topic_doc.max()


	for n in range(len(example)):
		sum_pr = sum(topic_doc[n,:])
		print("document: {} sum: {}".format(n, sum_pr))


	for n in range(len(example)):
		topic_most_pr = topic_doc[n].argmax()
		print("doc: {} topic: {}\n{}...".format(n, topic_most_pr, titles[n]))


	import matplotlib.pyplot as plt

	# use matplotlib style sheet
	try:
		plt.style.use('ggplot')
	except:
		# version of matplotlib might not be recent
		pass

	

	f, ax= plt.subplots(n_topics, 1, figsize=(8, 6), sharex=True)
	#for i, k in enumerate([0, 5, 9, 14, 19]):
	for i, word_topic_i in enumerate(topic_word):
		ax[i].stem(word_topic_i, linefmt='b-', markerfmt='bo', basefmt='w-')
		ax[i].set_xlim(-5,n_vocab+5)
		ax[i].set_ylim(0, max_word_prob + .05)
		ax[i].set_ylabel("Prob")
		ax[i].set_title("Topic {}".format(i))


	ax[n_topics-1].set_xlabel("Word")

	plt.tight_layout()
	plt.show()

	# topic_doc.shape()

	f, ax= plt.subplots(n_docs, 1, figsize=(8, 6), sharex=True)
	#for i, k in enumerate([1, 3, 4, 8, 9]):
	for i, doc_topic_i in enumerate(topic_doc):
		ax[i].stem(doc_topic_i, linefmt='r-', markerfmt='ro', basefmt='w-')
		ax[i].set_xlim(-1, n_topics+1)
		ax[i].set_ylim(0, 1)
		ax[i].set_ylabel("Prob")
		ax[i].set_title("Document {}".format(i))

	ax[n_docs-1].set_xlabel("Topic")

	plt.tight_layout()
	plt.show()


'''

['Sugar is bad to consume. My sister likes to have sugar, but not my father.',
 'My father spends a lot of time driving my sister around to dance practice.',
 'Doctors suggest that driving may cause increased stress and blood pressure.',
 'Sometimes I feel pressure to perform well at school, but my father never seems to drive my sister to do better.',
 'Health experts say that Sugar is not good for your lifestyle.']


 vocab
Out[3]: 
{0: 'sai',
 1: 'school',
 2: 'feel',
 3: 'consum',
 4: 'mai',
 5: 'pressur',
 6: 'seem',
 7: 'expert',
 8: 'doctor',
 9: 'perform',
 10: 'suggest',
 11: 'father',
 12: 'sometim',
 13: 'sugar',
 14: 'better',
 15: 'health',
 16: 'lot',
 17: 'good',
 18: 'around',
 19: 'never',
 20: 'stress',
 21: 'lifestyl',
 22: 'blood',
 23: 'increas',
 24: 'practic',
 25: 'sister',
 26: 'like',
 27: 'well',
 28: 'drive',
 29: 'caus',
 30: 'bad',
 31: 'time',
 32: 'danc',
 33: 'spend'}


iter 10
 *Topic 0
- father sugar sister spend lifestyl feel consum expert better sai
*Topic 1
- drive pressur lot danc around never perform seem well school
*Topic 2
- increas blood health caus good mai suggest stress doctor father


iter 1000
 *Topic 0
- father drive sister spend increas school sometim danc around lot
*Topic 1
- sugar sai like good lifestyl expert health consum bad mai
*Topic 2
- pressur suggest feel caus mai seem better doctor blood perform


iter 10

doc: 0 topic: 0
doc_0...
doc: 1 topic: 0
doc_1...
doc: 2 topic: 2
doc_2...
doc: 3 topic: 1
doc_3...
doc: 4 topic: 0
doc_4...


iter 1000

doc: 0 topic: 1
doc_0...
doc: 1 topic: 0
doc_1...
doc: 2 topic: 2
doc_2...
doc: 3 topic: 0
doc_3...
doc: 4 topic: 1
doc_4...


'''