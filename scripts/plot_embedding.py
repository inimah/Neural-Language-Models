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
from text_preprocessing import *
from gensim.models import Word2Vec, Doc2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


PATH = 'prepdata/lingspam'
EMBED_PATH = 'embedding/lingspam'

if __name__ == '__main__':

	#w2v_model = Word2Vec.load(os.path.join(EMBED_PATH,'w2v_cont_ls1'))
	#d2v_model1 = Doc2Vec.load(os.path.join(EMBED_PATH,'d2v_cont_ls1'))
	#d2v_model2 = Doc2Vec.load(os.path.join(EMBED_PATH,'d2v_cont_ls2'))
	#d2v_model3 = Doc2Vec.load(os.path.join(EMBED_PATH,'d2v_cont_ls3'))

	avg_embed1 = readPickle(os.path.join(EMBED_PATH,'avg_contls_embed1'))
	avg_embed2 = readPickle(os.path.join(EMBED_PATH,'avg_contls_embed2'))
	avgIDF_embed1 = readPickle(os.path.join(EMBED_PATH,'avgIDF_contls_embed1'))
	avgIDF_embed2 = readPickle(os.path.join(EMBED_PATH,'avgIDF_contls_embed2'))
	#d2v_embed1 = readPickle(os.path.join(EMBED_PATH,'d2v_cont_ls_embed1'))
	#d2v_embed2 = readPickle(os.path.join(EMBED_PATH,'d2v_cont_ls_embed2'))
	#d2v_embed3 = readPickle(os.path.join(EMBED_PATH,'d2v_cont_ls_embed3'))


	ls_classLabel = readPickle(os.path.join(PATH,'ls_classLabel'))


	pca1 = PCA(n_components=50)
	vec50_dim1 = pca1.fit_transform(avg_embed1)
	tsne1 = TSNE(n_components=2)
	vec2_dim1 = tsne1.fit_transform(vec50_dim1)

	plt.figure()
	plt.scatter(vec2_dim1[:, 0], vec2_dim1[:, 1])
	plt.savefig('unlabelled_avg_embed1.png')
	plt.clf()

	pca2 = PCA(n_components=50)
	vec50_dim2 = pca2.fit_transform(avg_embed2)
	tsne2 = TSNE(n_components=2)
	vec2_dim2 = tsne2.fit_transform(vec50_dim2)

	plt.figure()
	plt.scatter(vec2_dim2[:, 0], vec2_dim2[:, 1])
	plt.savefig('unlabelled_avg_embed2.png')
	plt.clf()

	pca1_idf = PCA(n_components=50)
	vec50_dim1_idf = pca1_idf.fit_transform(avgIDF_embed1)
	tsne1_idf = TSNE(n_components=2)
	vec2_dim1_idf = tsne1_idf.fit_transform(vec50_dim1_idf)

	plt.figure()
	plt.scatter(vec2_dim1_idf[:, 0], vec2_dim1_idf[:, 1])
	plt.savefig('unlabelled_avgIDF_embed1.png')
	plt.clf()

	pca2_idf = PCA(n_components=50)
	vec50_dim2_idf = pca2_idf.fit_transform(avgIDF_embed2)
	tsne2_idf = TSNE(n_components=2)
	vec2_dim2_idf = tsne2_idf.fit_transform(vec50_dim2_idf)

	plt.figure()
	plt.scatter(vec2_dim2_idf[:, 0], vec2_dim2_idf[:, 1])
	plt.savefig('unlabelled_avgIDF_embed2.png')
	plt.clf()


	labelled_vec1 = zip(ls_classLabel,vec2_dim1)
	labelled_vec2 = zip(ls_classLabel,vec2_dim2)
	labelled_vec3 = zip(ls_classLabel,vec2_dim1_idf)
	labelled_vec4 = zip(ls_classLabel,vec2_dim2_idf)

	plt.figure()
	fig, ax = plt.subplots()
	for doc, vec2_dim1 in labelled_vec1:
		ax.scatter(vec2_dim1[0], vec2_dim1[1], color=('r' if doc == 'spam' else 'b'))
	plt.savefig('labelled_avg_embed1.png')
	plt.clf()

	plt.figure()
	fig, ax = plt.subplots()
	for doc, vec2_dim2 in labelled_vec2:
		ax.scatter(vec2_dim2[0], vec2_dim2[1], color=('r' if doc == 'spam' else 'b'))
	plt.savefig('labelled_avg_embed2.png')
	plt.clf()

	plt.figure()
	fig, ax = plt.subplots()
	for doc, vec2_dim1_idf in labelled_vec3:
		ax.scatter(vec2_dim1_idf[0], vec2_dim1_idf[1], color=('r' if doc == 'spam' else 'b'))
	plt.savefig('labelled_avgIDF_embed1.png')
	plt.clf()

	plt.figure()
	fig, ax = plt.subplots()
	for doc, vec2_dim2_idf in labelled_vec4:
		ax.scatter(vec2_dim2_idf[0], vec2_dim2_idf[1], color=('r' if doc == 'spam' else 'b'))
	plt.savefig('labelled_avgIDF_embed2.png')
	plt.clf()

