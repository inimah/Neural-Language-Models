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
from vector_space import VectorSpace
from tfidf import TFIDF
from lsa import LSA
from tokenizer import Tokenizer
from gensim.models import Word2Vec, Doc2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# location: ~/anaconda2/lib/python2.7/site-packages/plotly/tools.py
from plotly.graph_objs import *
import plotly.figure_factory as ff
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage




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

	tokenized_docs = readPickle('../../embeddings/examples_2/lsa/nonprep/tokenized_docs')
	vocab = readPickle('../../embeddings/examples_2/lsa/nonprep/vocab')
	td_bow = readPickle('../../embeddings/examples_2/lsa/nonprep/td_bow')
	td_bow_sublin = readPickle('../../embeddings/examples_2/lsa/nonprep/td_bow_sublin')
	td_tfidf = readPickle('../../embeddings/examples_2/lsa/nonprep/td_tfidf')
	bow_null_ind = readPickle('../../embeddings/examples_2/lsa/nonprep/bow_null_ind')
	bow_sublin_null_ind = readPickle('../../embeddings/examples_2/lsa/nonprep/bow_sublin_null_ind')
	tfidf_null_ind = readPickle('../../embeddings/examples_2/lsa/nonprep/tfidf_null_ind')

	text_w = ['machine', 'great', 'like', 'python', 'data', 'watch', 'football', 'statistics', 'learning', 'fun', 'science', 'super', 'cool']

	## Word vectors from LSA-BOW approach 
	######################################

	lsa_bow_u= readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_bow_u')
	#print("lsa_bow_u.shape: %s \n" %str(lsa_bow_u.shape))
	#print(lsa_bow_u)

	lsa_bow_sigma= readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_bow_sigma')
	#print("lsa_bow_sigma.shape: %s \n" %str(lsa_bow_sigma.shape))
	#print(lsa_bow_sigma)

	lsa_bow_vt= readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_bow_vt')
	lsa_bow_v = np.transpose(lsa_bow_vt)
	#print("bow_v.shape: %s \n" %str(bow_v.shape))
	#print(bow_v)

	# words similarity w.r.t document context
	# column is principal component or dimension (context) in which we can measure similarity of words in context (rows of matrix)
	lsa_bow_diag_sigma = readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_bow_diag_sigma')
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

	

	wv_bow_= np.array(wv_bow_,dtype=object)
	wv_bow_dim = wv_bow_[:,1:]
	dv_bow_ = dv_bow[:,1:]

	# without dimension reduction

	
	dim = dv_bow.shape[1]
	cols = [i for i in range(1,dim)]

	# word vector
	df_wv_bow = pd.DataFrame(wv_bow_dim,columns=list(cols))
	df_wv_bow['terms']=text_w
	savePickle(df_wv_bow,'df_wv_bow')


	# document vector
	df_dv_bow = pd.DataFrame(dv_bow_,columns=list(cols))
	savePickle(df_dv_bow,'df_dv_bow')

	sim = float(dot(dv_bow_[0],dv_bow_[1]) / (norm(dv_bow_[0]) * norm(dv_bow_[1])))



	'''
	# Dimension reduction 
	# get 2nd to n- dimension of eigenvectors
	'''


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
	py.iplot(fig, filename='wv_bow_pca2')

	trace_wv_bow_tsne2 = go.Scatter(
	x = wv_bow_tsne2[:, 0],
	y = wv_bow_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_wv_bow_tsne2 = [trace_wv_bow_tsne2]
	
	
	fig = go.Figure(data=data_wv_bow_tsne2, layout=layout)
	py.iplot(fig, filename='wv_bow_tsne2')
	


	
	## Word vectors from LSA-Sublinear BOW approach  
	######################################

	lsa_bow_sublin_u= readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_bow_sublin_u')
	#print("lsa_bow_sublin_u.shape: %s \n" %str(lsa_bow_sublin_u.shape))
	#print(lsa_bow_sublin_u)

	lsa_bow_sublin_sigma= readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_bow_sublin_sigma')
	#print("lsa_bow_sublin_sigma.shape: %s \n" %str(lsa_bow_sublin_sigma.shape))
	#print(lsa_bow_sublin_sigma)

	lsa_bow_sublin_vt= readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_bow_sublin_vt')
	#print("lsa_bow_sublin_vt.shape: %s \n" %str(lsa_bow_sublin_vt.shape))
	#print(lsa_bow_sublin_vt)
	lsa_bow_sublin_v = np.transpose(lsa_bow_sublin_vt)
	#print("\nlsa_bow_sublin_v.shape: %s \n" %str(lsa_bow_sublin_v.shape))
	#print(lsa_bow_sublin_v)

	# words similarity w.r.t document context
	# column is principal component or dimension (context) in which we can measure similarity of words in context (rows of matrix)
	lsa_bow_sublin_diag_sigma = readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_bow_sublin_diag_sigma')
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

	

	wv_bow_sublin_=np.array(wv_bow_sublin_,dtype=object)
	wv_bow_sublin_dim = wv_bow_sublin_[:,1:]
	dv_bow_sublin_ = dv_bow_sublin[:,1:]

	# without dimension reduction

	
	dim = dv_bow_sublin.shape[1]
	cols = [i for i in range(1,dim)]

	# word vector
	df_wv_bow_sublin = pd.DataFrame(wv_bow_sublin_dim,columns=list(cols))
	df_wv_bow_sublin['terms']=text_w
	savePickle(df_wv_bow_sublin,'df_wv_bow_sublin')


	# document vector
	df_dv_bow_sublin = pd.DataFrame(dv_bow_sublin_,columns=list(cols))
	savePickle(df_dv_bow_sublin,'df_dv_bow_sublin')

	sim = float(dot(dv_bow_sublin_[0],dv_bow_sublin_[1]) / (norm(dv_bow_sublin_[0]) * norm(dv_bow_sublin_[1])))


	'''
	# Dimension reduction 
	# get 2nd to n- dimension of eigenvectors
	'''


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
	py.iplot(fig, filename='wv_bow_sublin_pca2')

	trace_wv_bow_sublin_tsne2 = go.Scatter(
	x = wv_bow_sublin_tsne2[:, 0],
	y = wv_bow_sublin_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_wv_bow_sublin_tsne2 = [trace_wv_bow_sublin_tsne2]
	
	
	fig = go.Figure(data=data_wv_bow_sublin_tsne2, layout=layout)
	py.iplot(fig, filename='wv_bow_sublin_tsne2')
	

	

	## Word vectors from LSA-TF IDF approach  
	######################################

	lsa_tfidf_u= readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_tfidf_u')
	#print("lsa_tfidf_u.shape: %s \n" %str(lsa_tfidf_u.shape))
	#print(lsa_tfidf_u)

	lsa_tfidf_sigma= readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_tfidf_sigma')
	#print("lsa_tfidf_sigma.shape: %s \n" %str(lsa_tfidf_sigma.shape))
	#print(lsa_tfidf_sigma)

	lsa_tfidf_vt= readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_tfidf_vt')
	#print("lsa_tfidf_vt.shape: %s \n" %str(lsa_tfidf_vt.shape))
	#print(lsa_tfidf_vt)
	lsa_tfidf_v = np.transpose(lsa_tfidf_vt)																																																																																																																																	
	#print("\nlsa_tfidf_v.shape: %s \n" %str(lsa_tfidf_v.shape))
	#print(lsa_tfidf_v)

	# words similarity w.r.t document context
	# column is principal component or dimension (context) in which we can measure similarity of words in context (rows of matrix)
	lsa_tfidf_diag_sigma = readPickle('../../embeddings/examples_2/lsa/nonprep/lsa_tfidf_diag_sigma')
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

	

	wv_tfidf_=np.array(wv_tfidf_,dtype=object)
	wv_tfidf_dim = wv_tfidf_[:,1:]

	
	dv_tfidf_ = dv_tfidf[:,1:]

	# without dimension reduction

	
	dim = dv_tfidf.shape[1]
	cols = [i for i in range(1,dim)]

	# word vector
	df_wv_tfidf = pd.DataFrame(wv_tfidf_dim,columns=list(cols))
	df_wv_tfidf['terms']=text_w
	savePickle(df_wv_tfidf,'df_wv_tfidf')


	# document vector
	df_dv_tfidf = pd.DataFrame(dv_tfidf_,columns=list(cols))
	savePickle(df_dv_tfidf,'df_dv_tfidf')

	sim = float(dot(dv_tfidf_[0],dv_tfidf_[1]) / (norm(dv_tfidf_[0]) * norm(dv_tfidf_[1])))


	'''
	# Dimension reduction 
	# get 2nd to n- dimension of eigenvectors

	'''


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
	py.iplot(fig, filename='wv_tfidf_pca2')

	trace_wv_tfidf_tsne2 = go.Scatter(
	x = wv_tfidf_tsne2[:, 0],
	y = wv_tfidf_tsne2[:, 1],
	mode = 'markers',
	text = text_w
	)

	data_wv_tfidf_tsne2 = [trace_wv_tfidf_tsne2]
	
	
	fig = go.Figure(data=data_wv_tfidf_tsne2, layout=layout)
	py.iplot(fig, filename='wv_tfidf_tsne2')
	

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



##################################################
# Ranking document with cosine similarity
# Generating document similarity matrix
#################################################
	
vs = VectorSpace(tokenized_docs,vocab)

# For document matrix from BOW
arr_bow = np.array(td_bow, dtype = object)
# create similarity matrix
sim_matrix_bow = np.zeros(shape=(arr_bow.shape[0], arr_bow.shape[0]), dtype='float32')
for i in range(arr_bow.shape[0]):
	sim_matrix_bow[i] = [vs._cosine(arr_bow[i], dv) for dv in arr_bow]

savePickle(sim_matrix_bow,'sim_matrix_bow')





# For document matrix from Sub-linear BOW
arr_bow_sublin = np.array(td_bow_sublin, dtype = object)
# create similarity matrix
sim_matrix_bow_sublin = np.zeros(shape=(arr_bow_sublin.shape[0], arr_bow_sublin.shape[0]), dtype='float32')
for i in range(arr_bow_sublin.shape[0]):
	sim_matrix_bow_sublin[i] = [vs._cosine(arr_bow_sublin[i], dv) for dv in arr_bow_sublin]

savePickle(sim_matrix_bow_sublin,'sim_matrix_bow_sublin')



# For document matrix from TFIDF
arr_tfidf = np.array(td_tfidf, dtype = object)
# create similarity matrix
sim_matrix_tfidf = np.zeros(shape=(arr_tfidf.shape[0], arr_tfidf.shape[0]), dtype='float32')
for i in range(arr_tfidf.shape[0]):
	sim_matrix_tfidf[i] = [vs._cosine(arr_tfidf[i], dv) for dv in arr_tfidf]

savePickle(sim_matrix_tfidf,'sim_matrix_tfidf')



# For document matrix from SVD + BOW
# create similarity matrix
sim_matrix_svdbow = np.zeros(shape=(dv_bow.shape[0], dv_bow.shape[0]), dtype='float32')
for i in range(dv_bow.shape[0]):
	sim_matrix_svdbow[i] = [vs._cosine(dv_bow[i], dv) for dv in dv_bow]

savePickle(sim_matrix_svdbow,'sim_matrix_svdbow')



# For document matrix from SVD + Sub-linear BOW
# create similarity matrix
sim_matrix_svdbow_sublin = np.zeros(shape=(dv_bow_sublin.shape[0], dv_bow_sublin.shape[0]), dtype='float32')
for i in range(dv_bow_sublin.shape[0]):
	sim_matrix_svdbow_sublin[i] = [vs._cosine(dv_bow_sublin[i], dv) for dv in dv_bow_sublin]

savePickle(sim_matrix_svdbow_sublin,'sim_matrix_svdbow_sublin')



# For document matrix from SVD + TFIDF 
# create similarity matrix
sim_matrix_svdtfidf = np.zeros(shape=(dv_tfidf.shape[0], dv_tfidf.shape[0]), dtype='float32')
for i in range(dv_tfidf.shape[0]):
	sim_matrix_svdtfidf[i] = [vs._cosine(dv_tfidf[i], dv) for dv in dv_tfidf]

savePickle(sim_matrix_svdtfidf,'sim_matrix_svdtfidf')


# For document matrix from Average Word2Vec
sim_matrix_avg_w2v = np.zeros(shape=(avg_embed1.shape[0], avg_embed1.shape[0]), dtype='float32')
for i in range(avg_embed1.shape[0]):
	sim_matrix_avg_w2v[i] = [vs._cosine(avg_embed1[i], dv) for dv in avg_embed1]

savePickle(sim_matrix_avg_w2v,'sim_matrix_avg_w2v')


# For document matrix from Average-IDF Word2Vec
sim_matrix_avgIDF_w2v = np.zeros(shape=(avgIDF_embed1.shape[0], avgIDF_embed1.shape[0]), dtype='float32')
for i in range(avgIDF_embed1.shape[0]):
	sim_matrix_avgIDF_w2v[i] = [vs._cosine(avgIDF_embed1[i], dv) for dv in avgIDF_embed1]

savePickle(sim_matrix_avgIDF_w2v,'sim_matrix_avgIDF_w2v')

##################################################
# Hierarchical clustering
# 
#################################################


# For document matrix from BOW
#################################################
names_bow = [str(i) for i in range(arr_bow.shape[0])]

fig = ff.create_dendrogram(arr_bow, orientation='bottom', color_threshold = 0.6, labels=names_bow, linkagefun=lambda x: linkage(arr_bow, 'single', metric='cosine'))
py.iplot(fig, filename='dendrogram_bow')

'''
to check / inspect the results
l_bow = linkage(arr_bow, method="single", metric="cosine")

In [313]: l_bow
Out[313]: 
array([[  0.        ,   4.        ,   0.45227744,   2.        ],
	   [  1.        ,   7.        ,   0.52565835,   3.        ],
	   [  2.        ,   8.        ,   0.52565835,   4.        ],
	   [  3.        ,   9.        ,   0.5527864 ,   5.        ],
	   [  6.        ,  10.        ,   0.63485163,   6.        ],
	   [  5.        ,  11.        ,   0.74180111,   7.        ]])


In [315]: pd.DataFrame(1-sim_matrix_bow)
Out[315]: 
		  0         1         2         3         4         5         6
0  0.000000  0.525658  0.800000  0.552786  0.452277  1.000000  0.800000
1  0.525658  0.000000  0.525658  0.823223  0.711325  1.000000  0.841886
2  0.800000  0.525658  0.000000  0.776393  0.817426  1.000000  0.800000
3  0.552786  0.823223  0.776393  0.000000  0.795876  1.000000  0.776393
4  0.452277  0.711325  0.817426  0.795876  0.000000  1.000000  0.634852
5  1.000000  1.000000  1.000000  1.000000  1.000000  0.000000  0.741801
6  0.800000  0.841886  0.800000  0.776393  0.634852  0.741801  0.000000

# Dendogram by matplotlib


'''

# Initialize figure by creating upper dendrogram
figure = ff.create_dendrogram(arr_bow, orientation='bottom', labels=names_bow, linkagefun=lambda x: linkage(arr_bow, 'single', metric='cosine'))
for i in range(len(figure['data'])):
	figure['data'][i]['yaxis'] = 'y2'

# Create Side Dendrogram
dendro_side = ff.create_dendrogram(arr_bow, orientation='right', linkagefun=lambda x: linkage(arr_bow, 'single', metric='cosine'))
for i in range(len(dendro_side['data'])):
	dendro_side['data'][i]['xaxis'] = 'x2'

# Add Side Dendrogram Data to Figure
figure['data'].extend(dendro_side['data'])

# Create Heatmap
dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
dendro_leaves = list(map(int, dendro_leaves))
heat_data = 1-sim_matrix_bow
heat_data = heat_data[dendro_leaves,:]
heat_data = heat_data[:,dendro_leaves]

heatmap = Data([
	Heatmap(
		x = dendro_leaves,
		y = dendro_leaves,
		z = heat_data,
		colorscale = 'YIGnBu'
	)
])

heatmap[0]['x'] = figure['layout']['xaxis']['tickvals']
heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

# Add Heatmap Data to Figure
figure['data'].extend(Data(heatmap))

# Edit Layout
figure['layout'].update({'width':800, 'height':800,
						 'showlegend':False, 'hovermode': 'closest',
						 })
# Edit xaxis
figure['layout']['xaxis'].update({'domain': [.15, 1],
								  'mirror': False,
								  'showgrid': False,
								  'showline': False,
								  'zeroline': False,
								  'ticks':""})
# Edit xaxis2
figure['layout'].update({'xaxis2': {'domain': [0, .15],
								   'mirror': False,
								   'showgrid': False,
								   'showline': False,
								   'zeroline': False,
								   'showticklabels': False,
								   'ticks':""}})

# Edit yaxis
figure['layout']['yaxis'].update({'domain': [0, .85],
								  'mirror': False,
								  'showgrid': False,
								  'showline': False,
								  'zeroline': False,
								  'showticklabels': False,
								  'ticks': ""})
# Edit yaxis2
figure['layout'].update({'yaxis2':{'domain':[.825, .975],
								   'mirror': False,
								   'showgrid': False,
								   'showline': False,
								   'zeroline': False,
								   'showticklabels': False,
								   'ticks':""}})

# Plot!
py.iplot(figure, filename='heatmap_bow')



# For document matrix from Sub-linear BOW
#################################################

names_bow_sublin = [str(i) for i in range(arr_bow_sublin.shape[0])]
fig = ff.create_dendrogram(arr_bow_sublin, orientation='bottom', labels=names_bow_sublin, linkagefun=lambda x: linkage(arr_bow_sublin, 'single', metric='cosine'))
py.iplot(fig, filename='dendrogram_bow_sublin')

'''
to check / inspect the results
l_bow_sublin = linkage(arr_bow_sublin, method="single", metric="cosine")

In [357]: l_bow_sublin
Out[357]: 
array([[  0.        ,   4.        ,   0.45227744,   2.        ],
       [  1.        ,   2.        ,   0.4880109 ,   2.        ],
       [  7.        ,   8.        ,   0.54037933,   4.        ],
       [  3.        ,   9.        ,   0.5527864 ,   5.        ],
       [  6.        ,  10.        ,   0.63485163,   6.        ],
       [  5.        ,  11.        ,   0.74180111,   7.        ]])

       In [358]: pd.DataFrame(1-sim_matrix_bow_sublin)
Out[358]: 
          0         1         2         3         4         5         6
0  0.000000  0.540379  0.800000  0.552786  0.452277  1.000000  0.800000
1  0.540379  0.000000  0.488011  0.809193  0.688413  1.000000  0.829337
2  0.800000  0.488011  0.000000  0.776393  0.817426  1.000000  0.800000
3  0.552786  0.809193  0.776393  0.000000  0.795876  1.000000  0.776393
4  0.452277  0.688413  0.817426  0.795876  0.000000  1.000000  0.634852
5  1.000000  1.000000  1.000000  1.000000  1.000000  0.000000  0.741801
6  0.800000  0.829337  0.800000  0.776393  0.634852  0.741801  0.000000



'''

# For document matrix from TFIDF
#################################################

names_arr_tfidf = [str(i) for i in range(arr_tfidf.shape[0])]
fig = ff.create_dendrogram(arr_tfidf, orientation='bottom', labels=names_arr_tfidf, linkagefun=lambda x: linkage(arr_tfidf, 'single', metric='cosine'))
py.iplot(fig, filename='dendrogram_arr_tfidf')

'''
to check / inspect the results
l_tfidf = linkage(arr_tfidf, method="single", metric="cosine")

In [380]: l_tfidf
Out[380]: 
array([[  0.        ,   4.        ,   0.58947372,   2.        ],
       [  1.        ,   7.        ,   0.61991643,   3.        ],
       [  2.        ,   8.        ,   0.689304  ,   4.        ],
       [  3.        ,   9.        ,   0.77308928,   5.        ],
       [  5.        ,   6.        ,   0.81755161,   2.        ],
       [ 10.        ,  11.        ,   0.83066876,   7.        ]])



In [381]: pd.DataFrame(1-sim_matrix_tfidf)
Out[381]: 
          0         1         2         3         4         5         6
0  0.000000  0.619916  0.996230  0.773089  0.589474  1.000000  0.996230
1  0.619916  0.000000  0.689304  0.996876  0.843256  1.000000  0.997147
2  0.996230  0.689304  0.000000  0.996625  0.996919  1.000000  0.996918
3  0.773089  0.996876  0.996625  0.000000  0.996625  1.000000  0.996625
4  0.589474  0.843256  0.996919  0.996625  0.000000  1.000000  0.830669
5  1.000000  1.000000  1.000000  1.000000  1.000000  0.000000  0.817552
6  0.996230  0.997147  0.996918  0.996625  0.830669  0.817552  0.000000

'''

# For document matrix from SVD + BOW
#################################################
names_dv_bow = [str(i) for i in range(dv_bow.shape[0])]

fig = ff.create_dendrogram(dv_bow, orientation='bottom', labels=names_dv_bow, linkagefun=lambda x: linkage(dv_bow, 'single', metric='cosine'))
py.iplot(fig, filename='dendrogram_dv_bow')



'''
to check / inspect the results
l_tfidf = linkage(arr_tfidf, method="single", metric="cosine")

In [380]: l_tfidf
Out[380]: 
array([[  0.        ,   4.        ,   0.58947372,   2.        ],
       [  1.        ,   7.        ,   0.61991643,   3.        ],
       [  2.        ,   8.        ,   0.689304  ,   4.        ],
       [  3.        ,   9.        ,   0.77308928,   5.        ],
       [  5.        ,   6.        ,   0.81755161,   2.        ],
       [ 10.        ,  11.        ,   0.83066876,   7.        ]])



In [381]: pd.DataFrame(1-sim_matrix_tfidf)
Out[381]: 
          0         1         2         3         4         5         6
0  0.000000  0.619916  0.996230  0.773089  0.589474  1.000000  0.996230
1  0.619916  0.000000  0.689304  0.996876  0.843256  1.000000  0.997147
2  0.996230  0.689304  0.000000  0.996625  0.996919  1.000000  0.996918
3  0.773089  0.996876  0.996625  0.000000  0.996625  1.000000  0.996625
4  0.589474  0.843256  0.996919  0.996625  0.000000  1.000000  0.830669
5  1.000000  1.000000  1.000000  1.000000  1.000000  0.000000  0.817552
6  0.996230  0.997147  0.996918  0.996625  0.830669  0.817552  0.000000

'''

# For document matrix from SVD Sub-linear BOW
#################################################

names_dv_bow_sublin = [str(i) for i in range(dv_bow_sublin.shape[0])]

fig = ff.create_dendrogram(dv_bow_sublin, orientation='bottom', labels=names_dv_bow_sublin, linkagefun=lambda x: linkage(dv_bow_sublin, 'single', metric='cosine'))
py.iplot(fig, filename='dendrogram_dv_bow_sublin')



# For document matrix from SVD + TFIDF
#################################################

names_dv_tfidf = [str(i) for i in range(dv_tfidf.shape[0])]
fig = ff.create_dendrogram(dv_tfidf, orientation='bottom', labels=names_dv_tfidf, linkagefun=lambda x: linkage(dv_tfidf, 'single', metric='cosine'))
py.iplot(fig, filename='dendrogram_dv_tfidf')


# For document matrix from Average Word2Vec
#################################################

names_avg_w2v = [str(i) for i in range(avg_embed1.shape[0])]
fig = ff.create_dendrogram(avg_embed1, orientation='bottom', labels=names_avg_w2v, linkagefun=lambda x: linkage(avg_embed1, 'single', metric='cosine'))
py.iplot(fig, filename='dendrogram_avg_w2v')




# For document matrix from Average Word2Vec
#################################################

names_avgIDF_w2v = [str(i) for i in range(avgIDF_embed1.shape[0])]
fig = ff.create_dendrogram(avgIDF_embed1, orientation='bottom', labels=names_avgIDF_w2v, linkagefun=lambda x: linkage(avgIDF_embed1, 'single', metric='cosine'))
py.iplot(fig, filename='dendrogram_avgIDF_w2v')


	

	