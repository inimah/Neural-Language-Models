# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "01.07.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import itertools

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import sys
from scipy import linalg,dot
from numpy.linalg import norm
import plotly.plotly as py
import plotly.graph_objs as go
from text_preprocessing import *
from gensim.models import Word2Vec, Doc2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE




#PATH = 'prepdata/lingspam'
#EMBED_PATH = 'embedding/lingspam'

if __name__ == '__main__':


	'''
	Original documents:
	["Machine learning is super fun", 
	"Python is super, super cool", 
	"Statistics is cool, too", 
	"Data science is fun", 
	"Python is great for machine learning", 
	"I like football", 
	"Football is great to watch"]
	'''

	tokenized_docs = readPickle('../../embeddings/examples_2/lsa/prep/tokenized_docs')
	vocab = readPickle('../../embeddings/examples_2/lsa/prep/vocab')
	td_bow = readPickle('../../embeddings/examples_2/lsa/prep/td_bow')
	td_bow_sublin = readPickle('../../embeddings/examples_2/lsa/prep/td_bow_sublin')
	td_tfidf = readPickle('../../embeddings/examples_2/lsa/prep/td_tfidf')
	bow_null_ind = readPickle('../../embeddings/examples_2/lsa/prep/bow_null_ind')
	bow_sublin_null_ind = readPickle('../../embeddings/examples_2/lsa/prep/bow_sublin_null_ind')
	tfidf_null_ind = readPickle('../../embeddings/examples_2/lsa/prep/tfidf_null_ind')



	#text_w = ['statistics', 'python', 'data', 'football', 'watch', 'machine', 'learning', 'science', 'fun', 'cool']
	text_w = vocab.values()

	## Word vectors from LSA-BOW approach 
	######################################

	lsa_bow_u= readPickle('../../embeddings/examples_2/lsa/prep/lsa_bow_u')
	#print("lsa_bow_u.shape: %s \n" %str(lsa_bow_u.shape))
	#print(lsa_bow_u)

	lsa_bow_sigma= readPickle('../../embeddings/examples_2/lsa/prep/lsa_bow_sigma')
	#print("lsa_bow_sigma.shape: %s \n" %str(lsa_bow_sigma.shape))
	#print(lsa_bow_sigma)

	lsa_bow_vt= readPickle('../../embeddings/examples_2/lsa/prep/lsa_bow_vt')
	lsa_bow_v = np.transpose(lsa_bow_vt)
	#print("bow_v.shape: %s \n" %str(bow_v.shape))
	#print(bow_v)

	# words similarity w.r.t document context
	# column is principal component or dimension (context) in which we can measure similarity of words in context (rows of matrix)
	lsa_bow_diag_sigma = readPickle('../../embeddings/examples_2/lsa/prep/lsa_bow_diag_sigma')
	wv_bow = dot(lsa_bow_u, lsa_bow_diag_sigma)
	#print("wv_bow.shape: %s \n" %str(wv_bow.shape))
	#print(wv_bow)

	# vocab list of BOW matrix after eliminating zeros vector (non-occur words)
	bow_vocab = {}
	i = 0
	for k,v in vocab.iteritems():
		if k not in bow_null_ind:
			bow_vocab[i]=v
			i+=1

	w_ind = []
	revert_bow_vocab = dict((v, k) for k, v in bow_vocab.iteritems())
	for w,i in revert_bow_vocab.iteritems():
		if w in text_w:
			w_ind.append(i)

	w_ind.sort()

	'''
	words: 1: 'statistics', 5: 'python', 6: 'data', 8: 'football', 9: 'watch',
	11: 'machine', 14: 'learning', 15: 'science', 16: 'fun', 18: 'cool'
	w = [1,5,6,8,9,11,14,15,16,18]
	'''
	  
	
	
	wv_bow_ = []
	for i in range(len(wv_bow)):
		if i in w_ind:
			wv_bow_.append(wv_bow[i])
	#wv_bow_
	

	# document similarity w.r.t. grouped context (principal components)
	# column is principal component or dimension (context) in which we can measure similarity of documents in context (rows of matrix)
	lsa_bow_sigma_vt = linalg.diagsvd(lsa_bow_sigma, len(lsa_bow_sigma), len(lsa_bow_vt))
	dv_bow = dot(lsa_bow_v, lsa_bow_sigma_vt)
	#print(dv_bow)

	'''
	# Dimension reduction 
	# get 2nd to n- dimension of eigenvectors
	'''

	wv_bow_= np.array(wv_bow_,dtype=object)
	wv_bow_dim = wv_bow_[:,1:]
	dv_bow_ = dv_bow[:,1:]


	pca2 = PCA(n_components=2)
	wv_bow_pca2 = pca2.fit_transform(wv_bow_dim)
	dv_bow_pca2 = pca2.fit_transform(dv_bow_)


	tsne2 = TSNE(n_components=2)
	wv_bow_tsne2 = tsne2.fit_transform(wv_bow_dim)

	layout = go.Layout(
		showlegend=False
	)


	trace_wv_bow_pca2 = go.Scatter(
	x = wv_bow_pca2[:, 0],
	y = wv_bow_pca2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_wv_bow_pca2 = [trace_wv_bow_pca2]
	

	
	fig = go.Figure(data=data_wv_bow_pca2, layout=layout)
	py.iplot(fig, filename='wv_bow_pca2_prep')

	trace_wv_bow_tsne2 = go.Scatter(
	x = wv_bow_tsne2[:, 0],
	y = wv_bow_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_wv_bow_tsne2 = [trace_wv_bow_tsne2]
	
	
	fig = go.Figure(data=data_wv_bow_tsne2, layout=layout)
	py.iplot(fig, filename='wv_bow_tsne2_prep')
	


	
	## Word vectors from LSA-Sublinear BOW approach  
	######################################

	lsa_bow_sublin_u= readPickle('../../embeddings/examples_2/lsa/prep/lsa_bow_sublin_u')
	#print("lsa_bow_sublin_u.shape: %s \n" %str(lsa_bow_sublin_u.shape))
	#print(lsa_bow_sublin_u)

	lsa_bow_sublin_sigma= readPickle('../../embeddings/examples_2/lsa/prep/lsa_bow_sublin_sigma')
	#print("lsa_bow_sublin_sigma.shape: %s \n" %str(lsa_bow_sublin_sigma.shape))
	#print(lsa_bow_sublin_sigma)

	lsa_bow_sublin_vt= readPickle('../../embeddings/examples_2/lsa/prep/lsa_bow_sublin_vt')
	#print("lsa_bow_sublin_vt.shape: %s \n" %str(lsa_bow_sublin_vt.shape))
	#print(lsa_bow_sublin_vt)
	lsa_bow_sublin_v = np.transpose(lsa_bow_sublin_vt)
	#print("\nlsa_bow_sublin_v.shape: %s \n" %str(lsa_bow_sublin_v.shape))
	#print(lsa_bow_sublin_v)

	# words similarity w.r.t document context
	# column is principal component or dimension (context) in which we can measure similarity of words in context (rows of matrix)
	lsa_bow_sublin_diag_sigma = readPickle('../../embeddings/examples_2/lsa/prep/lsa_bow_sublin_diag_sigma')
	wv_bow_sublin = dot(lsa_bow_sublin_u, lsa_bow_sublin_diag_sigma)
	#print("wv_bow_sublin.shape: %s \n" %str(wv_bow_sublin.shape))
	#print(wv_bow_sublin)

	'''
	words: 1: 'statistics', 5: 'python', 6: 'data', 8: 'football', 9: 'watch',
	11: 'machine', 14: 'learning', 15: 'science', 16: 'fun', 18: 'cool'
	'''
	
	wv_bow_sublin_ = []
	for i in range(len(wv_bow_sublin)):
		if i in w_ind:
			wv_bow_sublin_.append(wv_bow_sublin[i])
	#wv_bow_sublin_

	

	# document similarity w.r.t. grouped context (principal components)
	# column is principal component or dimension (context) in which we can measure similarity of documents in context (rows of matrix)
	lsa_bow_sublin_sigma_vt = linalg.diagsvd(lsa_bow_sublin_sigma, len(lsa_bow_sublin_sigma), len(lsa_bow_sublin_vt))
	dv_bow_sublin = dot(lsa_bow_sublin_v, lsa_bow_sublin_sigma_vt)
	#print(dv_bow_sublin)

	'''
	# Dimension reduction 
	# get 2nd to n- dimension of eigenvectors
	'''

	wv_bow_sublin_=np.array(wv_bow_sublin_,dtype=object)
	wv_bow_sublin_dim = wv_bow_sublin_[:,1:]
	pca2 = PCA(n_components=2)
	wv_bow_sublin_pca2 = pca2.fit_transform(wv_bow_sublin_dim)
	tsne2 = TSNE(n_components=2)
	wv_bow_sublin_tsne2 = tsne2.fit_transform(wv_bow_sublin_dim)


	trace_wv_bow_sublin_pca2 = go.Scatter(
	x = wv_bow_sublin_pca2[:, 0],
	y = wv_bow_sublin_pca2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_wv_bow_sublin_pca2 = [trace_wv_bow_sublin_pca2]
	
	
	fig = go.Figure(data=data_wv_bow_sublin_pca2, layout=layout)
	py.iplot(fig, filename='wv_bow_sublin_pca2_prep')

	trace_wv_bow_sublin_tsne2 = go.Scatter(
	x = wv_bow_sublin_tsne2[:, 0],
	y = wv_bow_sublin_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_wv_bow_sublin_tsne2 = [trace_wv_bow_sublin_tsne2]
	
	
	fig = go.Figure(data=data_wv_bow_sublin_tsne2, layout=layout)
	py.iplot(fig, filename='wv_bow_sublin_tsne2_prep')
	

	

	## Word vectors from LSA-TF IDF approach  
	######################################

	lsa_tfidf_u= readPickle('../../embeddings/examples_2/lsa/prep/lsa_tfidf_u')
	#print("lsa_tfidf_u.shape: %s \n" %str(lsa_tfidf_u.shape))
	#print(lsa_tfidf_u)

	lsa_tfidf_sigma= readPickle('../../embeddings/examples_2/lsa/prep/lsa_tfidf_sigma')
	#print("lsa_tfidf_sigma.shape: %s \n" %str(lsa_tfidf_sigma.shape))
	#print(lsa_tfidf_sigma)

	lsa_tfidf_vt= readPickle('../../embeddings/examples_2/lsa/prep/lsa_tfidf_vt')
	#print("lsa_tfidf_vt.shape: %s \n" %str(lsa_tfidf_vt.shape))
	#print(lsa_tfidf_vt)
	lsa_tfidf_v = np.transpose(lsa_tfidf_vt)																																																																																																																																	
	#print("\nlsa_tfidf_v.shape: %s \n" %str(lsa_tfidf_v.shape))
	#print(lsa_tfidf_v)

	# words similarity w.r.t document context
	# column is principal component or dimension (context) in which we can measure similarity of words in context (rows of matrix)
	lsa_tfidf_diag_sigma = readPickle('../../embeddings/examples_2/lsa/prep/lsa_tfidf_diag_sigma')
	wv_tfidf = dot(lsa_tfidf_u, lsa_tfidf_diag_sigma)
	#print("wv_tfidf.shape: %s \n" %str(wv_tfidf.shape))
	#print(wv_tfidf)

	'''
	words: 1: 'statistics', 5: 'python', 6: 'data', 8: 'football', 9: 'watch',
	11: 'machine', 14: 'learning', 15: 'science', 16: 'fun', 18: 'cool'
	'''
	  
	
	wv_tfidf_ = []
	for i in range(len(wv_tfidf)):
		if i in w_ind:
			wv_tfidf_.append(wv_tfidf[i])
	#wv_tfidf_

	

	# document similarity w.r.t. grouped context (principal components)
	# column is principal component or dimension (context) in which we can measure similarity of documents in context (rows of matrix)
	lsa_tfidf_sigma_vt = linalg.diagsvd(lsa_tfidf_sigma, len(lsa_tfidf_sigma), len(lsa_tfidf_vt))
	dv_tfidf = dot(lsa_tfidf_v, lsa_tfidf_sigma_vt)
	#print(dv_tfidf)

	'''
	# Dimension reduction 
	# get 2nd to n- dimension of eigenvectors

	'''

	wv_tfidf_=np.array(wv_tfidf_,dtype=object)
	wv_tfidf_dim = wv_tfidf_[:,1:]
	pca2 = PCA(n_components=2)
	wv_tfidf_pca2 = pca2.fit_transform(wv_tfidf_dim)
	tsne2 = TSNE(n_components=2)
	wv_tfidf_tsne2 = tsne2.fit_transform(wv_tfidf_dim)


	trace_wv_tfidf_pca2 = go.Scatter(
	x = wv_tfidf_pca2[:, 0],
	y = wv_tfidf_pca2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_wv_tfidf_pca2 = [trace_wv_tfidf_pca2]
	

	
	fig = go.Figure(data=data_wv_tfidf_pca2, layout=layout)
	py.iplot(fig, filename='wv_tfidf_pca2_prep')

	trace_wv_tfidf_tsne2 = go.Scatter(
	x = wv_tfidf_tsne2[:, 0],
	y = wv_tfidf_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_wv_tfidf_tsne2 = [trace_wv_tfidf_tsne2]
	
	
	fig = go.Figure(data=data_wv_tfidf_tsne2, layout=layout)
	py.iplot(fig, filename='wv_tfidf_tsne2_prep')
	
	'''
	## Word vectors from Word2Vec  
	######################################

	# Word vectors
	#skip gram
	w2v_1 = readPickle('../../embeddings/examples_2/w2v/w2v_embed1')
	#cbow
	w2v_2 = readPickle('../../embeddings/examples_2/w2v/w2v_embed2')
	

	

	w2v1_null_ind = np.where(~w2v_1.any(axis=1))[0]

	w2v2_null_ind = np.where(~w2v_2.any(axis=1))[0]

	# vocab list after eliminating zero values (non-occured terms)
	w2v1_vocab = {}
	i = 0
	for k,v in vocab.iteritems():
		if k not in w2v1_null_ind:
			w2v1_vocab[i]=v
			i+=1


	# get index of sampled words
	w2v_ind = []
	revert_w2v1_vocab = dict((v, k) for k, v in w2v1_vocab.iteritems())
	for w,i in revert_w2v1_vocab.iteritems():
		if w in text_w:
			w2v_ind.append(i)

	w2v_ind.sort()

	# list vector of related words
	w2v_1_ = []
	for i in range(len(w2v_1)):
		if i in w2v_ind:
			w2v_1_.append(w2v_1[i])


	w2v_2_ = []
	for i in range(len(w2v_2)):
		if i in w2v_ind:
			w2v_2_.append(w2v_2[i])

	
	'''
	# Dimension reduction 
	# get 2nd to n- dimension of eigenvectors

	'''

	pca2 = PCA(n_components=2)
	w2v1_pca2 = pca2.fit_transform(w2v_1_)
	w2v2_pca2 = pca2.fit_transform(w2v_2_)
	tsne2 = TSNE(n_components=2)
	w2v1_tsne2 = tsne2.fit_transform(w2v_1_)
	w2v2_tsne2 = tsne2.fit_transform(w2v_2_)


	trace_w2v1_pca2 = go.Scatter(
	x = w2v1_pca2[:, 0],
	y = w2v1_pca2[:, 1],
	mode = 'markers',
	text = text_w
	)

	trace_w2v2_pca2 = go.Scatter(
	x = w2v2_pca2[:, 0],
	y = w2v2_pca2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_w2v1_pca2 = [trace_w2v1_pca2]

	data_w2v2_pca2 = [trace_w2v2_pca2]
	

	
	fig = go.Figure(data=data_w2v1_pca2, layout=layout)
	py.iplot(fig, filename='w2v1_pca2')

	fig = go.Figure(data=data_w2v2_pca2, layout=layout)
	py.iplot(fig, filename='w2v2_pca2')

	trace_w2v1_tsne2 = go.Scatter(
	x = w2v1_tsne2[:, 0],
	y = w2v1_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	trace_w2v2_tsne2 = go.Scatter(
	x = w2v2_tsne2[:, 0],
	y = w2v2_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_w2v1_tsne2 = [trace_w2v1_tsne2]
	data_w2v2_tsne2 = [trace_w2v2_tsne2]
	
	
	fig = go.Figure(data=data_w2v1_tsne2, layout=layout)
	py.iplot(fig, filename='w2v1_tsne2')

	fig = go.Figure(data=data_w2v2_tsne2, layout=layout)
	py.iplot(fig, filename='w2v2_tsne2')


	## plotting doc2vec

	# Document vectors

	avg_embed1 = readPickle('../../embeddings/examples_2/w2v/avg_embed1')
	avg_embed2 = readPickle('../../embeddings/examples_2/w2v/avg_embed2')

	
	pca2 = PCA(n_components=2)
	dv1_pca2 = pca2.fit_transform(avg_embed1)
	dv2_pca2 = pca2.fit_transform(avg_embed2)

	
	trace_dv1_pca2 = go.Scatter(
	x = dv1_pca2[:, 0],
	y = dv1_pca2[:, 1],
	mode = 'markers',
	text = text_w
	)

	trace_dv2_pca2 = go.Scatter(
	x = dv2_pca2[:, 0],
	y = dv2_pca2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_dv1_pca2 = [trace_dv1_pca2]
	data_dv2_pca2 = [trace_dv2_pca2]
		
	fig = go.Figure(data=data_dv1_pca2, layout=layout)
	py.iplot(fig, filename='dv1_pca2')

	fig = go.Figure(data=data_dv2_pca2, layout=layout)
	py.iplot(fig, filename='dv2_pca2')

	tsne2 = TSNE(n_components=2)
	dv1_tsne2 = tsne2.fit_transform(avg_embed1)
	dv2_tsne2 = tsne2.fit_transform(avg_embed2)

	trace_dv1_tsne2 = go.Scatter(
	x = dv1_tsne2[:, 0],
	y = dv1_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	trace_dv2_tsne2 = go.Scatter(
	x = dv2_tsne2[:, 0],
	y = dv2_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_dv1_tsne2 = [trace_dv1_tsne2]
	data_dv2_tsne2 = [trace_dv2_tsne2]
	
	
	fig = go.Figure(data=data_dv1_tsne2, layout=layout)
	py.iplot(fig, filename='dv1_tsne2')

	fig = go.Figure(data=data_dv2_tsne2, layout=layout)
	py.iplot(fig, filename='dv2_tsne2')

	avgIDF_embed1 = readPickle('../../embeddings/examples_2/w2v/avgIDF_embed1')
	avgIDF_embed2 = readPickle('../../embeddings/examples_2/w2v/avgIDF_embed2')


	dv1_idf_pca2 = pca2.fit_transform(avgIDF_embed1)
	dv2_idf_pca2 = pca2.fit_transform(avgIDF_embed2)

	trace_dv1_idf_pca2 = go.Scatter(
	x = dv1_idf_pca2[:, 0],
	y = dv1_idf_pca2[:, 1],
	mode = 'markers',
	text = text_w
	)

	trace_dv2_idf_pca2 = go.Scatter(
	x = dv2_idf_pca2[:, 0],
	y = dv2_idf_pca2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_dv1_idf_pca2 = [trace_dv1_idf_pca2]
	data_dv2_idf_pca2 = [trace_dv2_idf_pca2]
		
	fig = go.Figure(data=data_dv1_idf_pca2, layout=layout)
	py.iplot(fig, filename='dv1_idf_pca2')

	fig = go.Figure(data=data_dv2_idf_pca2, layout=layout)
	py.iplot(fig, filename='dv2_idf_pca2')


	dv1_idf_tsne2 = tsne2.fit_transform(avgIDF_embed1)
	dv2_idf_tsne2 = tsne2.fit_transform(avgIDF_embed2)

	trace_dv1_idf_tsne2 = go.Scatter(
	x = dv1_idf_tsne2[:, 0],
	y = dv1_idf_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	trace_dv2_idf_tsne2 = go.Scatter(
	x = dv2_idf_tsne2[:, 0],
	y = dv2_idf_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_dv1_idf_tsne2 = [trace_dv1_idf_tsne2]
	data_dv2_idf_tsne2 = [trace_dv2_idf_tsne2]
	
	
	fig = go.Figure(data=data_dv1_idf_tsne2, layout=layout)
	py.iplot(fig, filename='dv1_idf_tsne2')

	fig = go.Figure(data=data_dv2_idf_tsne2, layout=layout)
	py.iplot(fig, filename='dv2_idf_tsne2')


	'''




	


	

	