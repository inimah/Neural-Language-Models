# -*- coding: utf-8 -*-
#__author__ = "@tita [Iftitahu Ni'mah]"
#__date__ = "06.05.2017"
#__version__ = "1.0.1"
#__maintainer__ = "@tita [Iftitahu Ni'mah]"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function


import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tarfile
import zipfile
import pandas as pd
import numpy as np
import math
import random
import codecs
import matplotlib.pyplot as plt
import nltk
import itertools
import cPickle
import h5py
import collections
from collections import defaultdict
import argparse
import logging
from gensim.models import Word2Vec
import deepdish as dd

np.random.seed([3,1415])


ENRON_PATH = 'data/enron/maildata'
POLYGOT_PATH = 'data/multilingual/polygot'
EUROPARL1_PATH = 'data/multilingual/europarl/txt'
EUROPARL2_PATH = 'data/multilingual/europarl/nl-en'
TED_PATH = 'data/multilingual/ted'
TEST_PATH = 'data/multilingual/test'



'''
# Regex pattern. Currently is not used, replaced by nltk.word_tokenize library that has covered everything declared here...
pattern = r"""
 (?x)                   # set flag to allow verbose regexps
 (?:[A-Z]\.)+           # abbreviations, e.g. U.S.A.
 |\$?\d+(?:\.?,\d+)?%?       # numbers, incl. currency and percentages
 |\w+(?:[-']\w+)*       # words w/ optional internal hyphens/apostrophe
 |(?:[+/\-@&*])         # special characters with meanings
 """
'''



# listing files inside folder and subfolders
################################################
def listData(pathname):
	filenames = []
	for path, subdirs, files in os.walk(pathname):
		for name in files:
			filenames.append(os.path.join(path,name))
	return filenames


# get filename and extension
# to distinguish compressed file (*.tar, *.zip) and uncompressed file (*.txt, *.csv) 
################################################
def getNameExtension(filenames):
	filepath = os.path.basename(filenames)
	fileName, fileExtension = os.path.splitext(filepath)

	return fileName, fileExtension


# generating pairs of class labelname and path of file for supervised learning
# the text data is located in a folder with class / label name. e.g. spam, ham, category of documents
# array list of (filename, labelname) --> dictionary list of of { labelname: [filenames,...] }
################################################
def getClassLabel(filenames):
	allfiles = []
	for path in filenames:
		dir = os.path.dirname(path)
		labelname = os.path.basename(dir)
		allfiles.append([path,labelname])
	# this dictionary list {filepath: label,...} is not yet grouped by class label, and contains duplicate labels
	filedict = dict(allfiles)
	# grouping the duplicate keys and reverse the order of dictionary to {label:[filepaths,...]}
	datadict = defaultdict(list)
	for key, value in sorted(filedict.iteritems()):
		datadict[value].append(key)

	return datadict

# reading each text document
################################################
def readDocs(filename):
	infile = open(filename) 
	content = infile.read()
	infile.close()
	
	return content

# reading ta file without un-tar-ing it 
################################################
def readTarDocs(filetar):
	tar = tarfile.open(filetar)
	for member in tar.getmembers():
		f = tar.extractfile(member)
		content = f.read()
	tar.close()
	
	return content

def readZipDocs(filezip):
	with zipfile.ZipFile(fileZip) as z:
		for filename in z.namelist():
			if not filename.endswith('/'):
				with z.open(filename) as f:
					content = f.read()
	return content



# saving file into pickle format
################################################
def savePickle(dataToWrite,pickleFilename):
	f = open(pickleFilename, 'wb')
	cPickle.dump(dataToWrite, f)
	f.close()

# reading file in pickle format
################################################
def readPickle(pickleFilename):
	f = open(pickleFilename, 'rb')
	obj = cPickle.load(f)
	f.close()
	return obj

# saving file into hdf5 format
# only works for array with same length
################################################
def saveH5File(h5Filename,datasetName,dataToWrite):
	# h5Filename should be in format 'name-of-file.h5'
	# datasetName in String "" format
	with h5py.File(h5Filename, 'w') as hf:
		hf.create_dataset(datasetName,  data=dataToWrite)

# reading file in hdf5 format
# only works for array with same length
################################################
def readH5File(h5Filename,datasetName):
	# h5Filename should be in format 'name-of-file.h5'
	# datasetName in String "" format
	with h5py.File(h5Filename, 'r') as hf:
		data = hf[datasetName][:]
	return data

# using deepdish library
def saveH5Dict(h5Filename,datadict):
	dd.io.save(h5Filename, datadict)

# using deepdish library
def readH5Dict(h5Filename):
	data = dd.io.load(h5Filename)
	return data

# encoding unicode data (u'') in nested array or dictionary to string
################################################
def convert(data):
	if isinstance(data, basestring):
		return str(data)
	elif isinstance(data, collections.Mapping):
		return dict(map(convert, data.iteritems()))
	elif isinstance(data, collections.Iterable):
		return type(data)(map(convert, data))
	else:
		return data

# convert sentences (in words form) to array list of numbers
# vocab here is dictionary pairs of (index, word)
# word sentence is original sentence (array of word tokens in a document ['','',''])
################################################
def wordsToIndex(vocab,wordSentence):
	revertVocab = dict((v, k) for k, v in vocab.iteritems())
	numSentence = [revertVocab[i] for i in wordSentence]

	return numSentence


# revert back array list of numbers to original sentences
# vocab here is dictionary pairs of (index, word)
################################################
def indexToWords(vocab,numSentence):
	revertSentence = [vocab[i] for i in numSentence]

	return revertSentence

# creating word vocabulary and training sets for bi-lingual corpora pairs
# text documents need to be stored in their corresponding language folders (eg: en, nl)
################################################

def generatePairset(datadictionary):
	
	alltokens = dict()
	alldocs = dict()
	langDocs = []
	tokens = []
	worddocs_freq = []
	vocab = dict()

	# datadictionary is dictionary containing of filepaths
	# k denotes corpora associated to certain language 
	for k in datadictionary:
		docs=[]
		docwords=[]
		words=[]
		for filename in datadictionary[k]:
			docs.append(readDocs(filename))
		for doc in docs:
			# tokenize, but preserve punctuations,n-grams, currency, percentage
			docwords.append(nltk.word_tokenize(doc.decode('utf-8','ignore')))

		# encoding unicode data back to string
		words = convert(docwords)
		# new dictionary to store all information {language label: [[..,..],[..,..]]}
		alltokens[k] = words
		# also store in array
		langDocs.append(words)

	# generating vocabulary list from each language corpora
	# the length of array is number of distinct language corpora
	for i in range(len(langDocs)):
		tmp = sum(langDocs[i],[])
		tokens.append(tmp)

    for i in range(len(tokens)): 
    	# frequency of word across document corpus for each language corpora
    	# nltk creates pair word - frequency in dictionary format
    	tfDict = nltk.FreqDist(tokens[i])
    	worddocs_freq.append(tfDict)
    	terms = tfDict.keys()
    	# add 'zero' as the first vocab and 'UNK' as unknown words
    	terms.insert(0,'zero')
    	terms.append('UNK')
    	vocab[i] = dict([(j,terms[j]) for j in range(len(terms))])

	
	# save vocabulary list
	savePickle(vocab,'vocabulary')
	# alternative - saving as h5 file
	#saveH5File('vocabulary.h5','vocabulary',vocab)

	cnt = 0
	for k in alltokens:
		numbersInDoc = []
		for i in range(len(alltokens[k])):
			numbersInDoc.append(wordsToIndex(vocab[cnt],alltokens[k][i]))
		alldocs[k] = numbersInDoc
		cnt += 1

	# save all dictionary of all documents (in numerical list format)
	savePickle(alldocs,'documents')
	# alternative - saving as h5 file
	saveH5Dict('documents.h5',alldocs)

	   

	return tokens, worddocs_freq, vocab, alltokens, alldocs


# creating word vocabulary and training sets
# for monolingual corpora
################################################

def generateTrainset(datadictionary):
	
	alltokens = dict()
	alldocs = dict()

	# datadictionary is dictionary containing of filepaths
	for k in datadictionary:
		docs=[]
		docwords=[]
		words=[]
		for filename in datadictionary[k]:
			docs.append(readDocs(filename))
		for doc in docs:
			# tokenize, but preserve punctuations,n-grams, currency, percentage
			docwords.append(nltk.word_tokenize(doc.decode('utf-8','ignore')))

		# encoding unicode data back to string
		words = convert(docwords)
		# new dictionary to store all information {label: [[..,..],[..,..]]}
		alltokens[k] = words
	# generating vocabulary from all document tokens
	tokens = [value for key, value in alltokens.items()]
	mergedTokens = sum(tokens,[])
	# frequency of word across document corpus
	worddocs_freq = nltk.FreqDist(itertools.chain(*mergedTokens)) 
	uniqueWords = worddocs_freq.keys()
	# add 'zero' as the first vocab and 'UNK' as unknown words
	uniqueWords.insert(0,'zero')
	uniqueWords.append('UNK')
	# indexing word vocabulary : pairs of (index,word)
	vocab=dict([(i,uniqueWords[i]) for i in range(len(uniqueWords))])
	# save vocabulary list
	savePickle(vocab,'vocabulary')
	# alternative - saving as h5 file
	#saveH5File('vocabulary.h5','vocabulary',vocab)

	for k in alltokens:
		numbersInDoc = []
		for i in range(len(alltokens[k])):
			numbersInDoc.append(wordsToIndex(vocab,alltokens[k][i]))
		alldocs[k] = numbersInDoc

	# save all dictionary of all documents (in numerical list format)
	savePickle(alldocs,'documents')
	# alternative - saving as h5 file
	saveH5Dict('documents.h5',alldocs)

	   

	return mergedTokens, worddocs_freq, vocab, alltokens, alldocs

# characters for end-of-sentence markers
################################################
def endOfSentence(i, docTokens):
	
	return docTokens[i] in ('.','?','!') and (i == len(docTokens) - 1 or not docTokens[i+1] in ('.','?','!'))




# splitting document into sentences
################################################
def splitSentences(docTokens):
	"""Split sentences, returns sentences as a list of lists of tokens
	 (each sentence is a list of tokens)"""
	sentences = []
	begin = 0
	for i, token in enumerate(docTokens):
		if endOfSentence(i, docTokens): 
			sentences.append(docTokens[begin:i+1])
			begin = i+1
			
	#If the last sentence does not end nicely with a EOS-marker
	#we would miss it. Add the last sentence if our 'begin' cursor
	#isn't at the end yet:    
	if begin < len(docTokens):
		sentences.append(docTokens[begin:])        
	return sentences


# maximum, minimum, and average word length per class label
# return dictionary format of scalar values (n keys = n classes)
def classMinMaxAvgLength(datadict):
	minlength = dict()
	maxlength = dict()
	avglength = dict()
	for key in datadict:
		ndocs = (len(datadict[key]))
		minArr = (min(len(l) for l in datadict[key]))
		maxArr = (max(len(l) for l in datadict[key]))
		avgArr = (sum(len(l) for l in datadict[key]))
		minlength[key] = minArr
		maxlength[key] = maxArr
		avglength[key] = avgArr/ndocs
	
	return minlength, maxlength, avglength


# maximum, minimum, and average word length for whole document corpus
# return scalar values 
def getMinMaxAvgLength(datadict):
	minArr = []
	maxArr = []
	avgArr = []
	ndocs = []
	for key in datadict:
		ndocs.append(len(datadict[key]))
		minArr.append(min(len(l) for l in datadict[key]))
		maxArr.append(max(len(l) for l in datadict[key]))
		avgArr.append(sum(len(l) for l in datadict[key]))
	n = sum(ndocs)
	minlength = min(minArr)
	maxlength = max(maxArr)
	avglength = sum(avgArr)/n

	return minlength, maxlength, avglength

# return array of sentences / documents w.r.t classes
# the length of array represents number of classes in document corpus

##example:
##number of class label is 2
##In []: len(sentences2)
##Out[]: 2

##number of documents in class '0' is 13
##In []: len(sentences2[0])
##Out[]: 13

##document[0] in class '0' has 2875 sentences
##In []: len(sentences2[0][0])
##Out[]: 2875

##the first sentence in corresponding document consists of 33 words
##In []: len(sentences2[0][0][0])
##Out[]: 33

##word[0] in corresponding sentence has 6 characters
##In []: len(sentences2[0][0][0][0])
##Out[]: 6


## data dictonary here is a dictionary of word tokens (not numeric tokens!)
def getSentencesClass(datadict):
	sentences = []
	for key in datadict:
		tmp = []
		for i in range(len(datadict[key])):
			tmp.append(splitSentences(datadict[key][i]))
		sentences.append(tmp)
	return sentences

## transform to numeric form of sentences
def sentenceToNum(sentences):
	numSentences = []
	
	# number of language/class
	for i in range(len(sentences)):
		sent = []
	    sentdoc = []
		# number of documents in corresponding language / class
		for j in range(len(sentences[i])):
			# for each document, transform sentences to numeric sentences
			for k in range(len(sentences[i][j])):
				sent.append(wordsToIndex(vocab[i], sentences[i][j][k]))
			sentdoc.append(sent)
		numSentences.append(sentdoc)



# return array of sentences / documents in a corpus
# the length of array represents number of documents in corpus 
def getSentencesAll(datadict):
	npArr = dictToArray(datadict)
	dataArr = sum(npArr,[])
	sentences = []
	for i in range(len(dataArr)):
		sentences.append(splitSentences(dataArr[i]))

	return sentences


def printSentencesClass(classid, docid, sentid, sentences):
	for i in range(len(sentences)):
		if i == classid:
			for j in range(len(sentences[i])):
				if j == docid:
					for k in range(len(sentences[i][j])):
						if k == sentid:
							for txt in sentences[i][j][k]:
								print('%s ' %txt,end='') 


# get statistics (number of sentences per document, number of words per sentence) for each class 
def getStatClass(sentences):
	nSentences = dict()
	nWords = dict()
	# class -nth
	for i in range(len(sentences)):
		n1 = []
		n2 = [] 
		# document -nth
		for j in range(len(sentences[i])):
			n3 = []
			n1.append(len(sentences[i][j]))
			for k in range(len(sentences[i][j])):
				n3.append(len(sentences[i][j][k]))
			n2.append(n3)
		nSentences[i] = n1
		nWords[i] = n2 

	minSent = []
	maxSent = []
	sumSent = []
	avgSent = []

	minWords = []
	maxWords = []
	sumWords = []
	avgWords = []

	for n in range(len(nSentences)):
		minSent.append(min(nSentences[n]))
		maxSent.append(max(nSentences[n]))
		sumSent.append(sum(nSentences[n]))
		tmp = sum(nSentences[n]) / len(nSentences[n])
		avgSent.append(tmp)

	for m in range(len(nWords)):
		arr1 = []
		arr2 = []
		arr3 = []
		arr4 = []
		for n in range(len(nWords[m])):
			arr1.append(min(nWords[m][n]))
			arr2.append(max(nWords[m][n]))
			tmp = sum(nWords[m][n])
			arr3.append(tmp)
			arr4.append(tmp/len(nWords[m][n]))
		minWords.append(arr1)
		maxWords.append(arr2)
		sumWords.append(arr3)
		avgWords.append(arr4)


	return nSentences, nWords, minSent, maxSent, sumSent, avgSent, minWords, maxWords, sumWords, avgWords

# get statistics (number of sentences, number of words) from all document corpus
def getStatDocs():
	return 0 


# matrix vectorization for sentences in a document (3D matrix, i: n sentences in a document, j: )
def sentenceMatrixVectorization(sentenceInDoc, nVocab, nTimeSequence):
	sequences = np.zeros((len(word_sentences), nTimeSequence, nVocab))
	for i, sentence in enumerate(word_sentences):
		for j, word in enumerate(sentence):
			sequences[i, j, word] = 1



# transform string class label to nominal 
def stringToNominalClass(datadict):
	numericdict = dict()
	
	i=0
	for k in datadict:
		numericdict[i]=datadict[k]
		i +=1
	return numericdict 

# return dictionary to nested array
def dictToArray(numericdict):
	list1 = []
	for key in numericdict:
		list2 = []
		for i in range(len(numericdict[key])):
			list2.append(numericdict[key][i])
		list1.append(list2)

	npArr = np.array(list1)
	return npArr

# data dictionary here is in nominal class label format
# the resulting array is a mixed class label data sets
def dictMrgArray(numericdict):
	xlist = []
	ylist = []
	for key in numericdict:
		for j in range(len(numericdict[key])):
			xlist.append(numericdict[key][j])
			ylist.append(key)

	npx = np.array(xlist)
	npy = np.array(ylist)

	return npx, npy



# splitting data in dictionary format (with nominal class)
def splitDataDict(datadict, train_percent=.6, validate_percent=.2, seed=None):
	np.random.seed([3,1415])
	train = dict()
	validate = dict()
	test = dict()

	# count length of data dictionary (number of documents)
	numdoc = []
	for key in datadict:
		numdoc.append(len(datadict[key]))

	# permutation (sampling) per class
	perm = []
	train_end = []
	validate_end = []

	for l in range(len(numdoc)):
		perm.append(np.random.permutation(numdoc[l]))
		tmp = int(train_percent * numdoc[l])
		train_end.append(tmp)
		validate_end.append(int(validate_percent * numdoc[l]) + tmp)


	for lkey in range(len(datadict)):
		trainArr = []
		validateArr = []
		testArr = []
		key = datadict.keys()[lkey]
		ind = perm[lkey]
		indTrain = train_end[lkey]
		indVal = validate_end[lkey]

		for i in ind[:indTrain]:
			trainArr.append(datadict[key][i])
		train[key] = trainArr

		for j in ind[indTrain:indVal]:
			validateArr.append(datadict[key][j])
		validate[key] = validateArr

		for k in ind[indVal:]:
			testArr.append(datadict[key][k])
		test[key] = testArr

	savePickle(train,'trainset')
	savePickle(validate,'validateset')
	savePickle(test,'testset')
	# alternative - saving as h5 file
	saveH5Dict('trainset.h5',train)
	saveH5Dict('validate.h5',validate)
	saveH5Dict('test.h5',test)

	return train, validate, test


# splitting data in arraylist format
# not yet tested, will probably raise errors
def splitDataArr(arrdata, train_percent=.6, validate_percent=.2, seed=None):
	np.random.seed(seed)
	train = []
	validate = []
	test = []
	m = len(arrdata)
	perm = np.random.permutation(m)
	train_end = int(train_percent * m)
	validate_end = int(validate_percent * m) + train_end
	for i in perm[:train_end]:
		train.append(arrdata[i])
	for j in perm[train_end:validate_end]:
		validate.append(arrdata[j])
	for k in perm[validate_end:]:
		test.append(arrdata[k])

	savePickle(train,'trainset')
	savePickle(validate,'validateset')
	savePickle(test,'testset')
	# alternative - saving as h5 file
	saveH5File('trainset.h5','trainset',train)
	saveH5File('validate.h5','validateset',validate)
	saveH5File('test.h5','testset',test)

	return train, validate, test


# generate embedding of the corresponding training data sets, 
################################################
def generateEmbedding(documents, vocab, argsize, argiter):

	model = Word2Vec(documents, size=argsize, min_count=5, window=5, sg=1, iter=argiter)
	weights = model.wv.syn0
	d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
	embedding = np.zeros(shape=(len(vocab)+1, argsize), dtype='float32')

	for i, w in vocab.items():
		if w not in d:continue
		embedding[i, :] = weights[d[w], :]
	savePickle(embedding,'embedding')
	# alternative - saving as h5 file
	#saveH5File('embedding.h5','embedding',embedding)
	return embedding


# check whether monolingual corpus has same number of documents, if not print the differences 
################################################
#def checkParalelCorpus():
#	return 0

# expand word embedding 
################################################
#def expandEmbedding():
#	return 0


# 
################################################
 

#if __name__ == '__main__':

#   get list of data files
#	filenames = listData(TEXT_PATH)

#	datadict = getClassLabel(filenames)
#	mergedTokens, worddocs_freq, vocab, alltokens, alldocs  = generateTrainset(datadict)

#   pretrain word embedding from document corpus with word2vec
#	embedding = generateEmbedding(mergedTokens, vocab, 100, 50)
#	numericdict = stringToNominalClass(alldocs)
#	minlength, maxlength, avglength = getMinMaxAvgLength(numericdict)
#   sentences = getSentencesClass(alltokens)
#   try printing a sample of sentence
#   printSentencesClass(0, 0, 0, sentences)
#   transform to numerical sentenes
#   numSentences = sentenceToNum(sentences)


#   splitting training, validation, test sets (in dictionary format)
#	train, validate, test = splitDataDict(numericdict, train_percent=.6, validate_percent=.2, seed=None)
	

#	xTrain, yTrain = dictToArray(train)
#	xValidate, yValidate = dictToArray(validate)
#	xTest, yTest = dictToArray(test)
