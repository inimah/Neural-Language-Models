# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "20.05.2017"
#__update__ = "06.05.2017"
#__version__ = "1.0.1"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function


import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import tarfile
import zipfile
import re
import pandas as pd
import numpy as np
import math
import random
import codecs
#import matplotlib.pyplot as plt
import nltk
import itertools
import cPickle
import h5py
import collections
from collections import defaultdict
from collections import namedtuple
import argparse
import logging
from bs4 import BeautifulSoup
import html2text  # another alternative of bs4, perserving url link
from lsa.tokenizer import Tokenizer
from gensim.models import Word2Vec
#import deepdish as dd

np.random.seed([3,1415])



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


# reading data from csv / structured format 
def extractData(fileDat):
	data = pd.read_table(fileDat,sep=',',index_col=None,header=0,error_bad_lines=False,dtype = unicode)
	return data


def writeData(df,filename):
	file = open(filename, 'w')
	file.write(df)
	file.close()


def split_at(s, c, n):
	words = s.split(c)
	return c.join(words[:n]), c.join(words[n:])

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

def readZipDocs(fileZip):
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

def textToTokens(text):

	tokenizer = Tokenizer()

	tokens = nltk.word_tokenize(text.decode('utf-8','ignore'))
	

	# remove stopwords and stemming
	tokenSw = tokenizer._remove_stop_words(tokens)
	tokenStem = tokenizer._stemming(tokenSw)

	return tokens, tokenStem


# input here is tokenized document in dictionary format
def indexVocab(datadictionary):

	# generate tokens from subject title
	arrTokens = [value for key, value in datadictionary.items()]
	wordTokens = sum(arrTokens,[])

	# frequency of word across document corpus
	tf = nltk.FreqDist(itertools.chain(*wordTokens)) 
	wordIndex = tf.keys()
	# add 'zero' as the first vocab and 'UNK' as unknown words
	wordIndex.insert(0,'SOF')
	wordIndex.append('EOF')
	wordIndex.append('UNK')
	# indexing word vocabulary : pairs of (index,word)
	vocab=dict([(i,wordIndex[i]) for i in range(len(wordIndex))])

	return vocab, tf 


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


# in different server setting, the library may requires soup = BeautifulSoup(text, "lxml") or BeautifulSoup(text, "html5lib")
def htmlToTextBS(text):
	#soup = BeautifulSoup(text, "lxml")
	soup = BeautifulSoup(text,'html.parser')

	for script in soup(["script", "style"]):
		script.extract()

	txt = soup.get_text()
	# eliminate multiple line break (if anys)
	txt =  re.sub(r'\n\s*\n', '\n\n', txt)

	# split into raw sentence according to line breaks
	# this probably need to be checked further if one line include more than 1 sentence 
	arrayTxt = txt.encode("ascii","ignore").split("\n") 

	rawSentences=[]
	for text in arrayTxt:
		tmp = text.strip()
		if tmp != "":
			rawSentences.append(tmp)

	return rawSentences

# we don't use this function
def htmlToText(text):

	# this will return text, including URL links

	txt = html2text.html2text(text)

	return txt



# tokenized unlabelled corpus
# and generate vocabulary word list
# for short text document (single sentence)
################################################
def generateSentVocab(documents):

	docwords = []
	alltokens = []
	# tokenized words per sentence 
	for text in documents:
		txt = str(text).lower()
		tmp = nltk.word_tokenize(txt.decode('utf-8','ignore'))
		docwords.append(tmp)

	words = convert(docwords)
	# merge all tokens into one list
	alltokens = sum(words,[])

	# frequency occurence of terms
	tf = nltk.FreqDist(alltokens)
	# terms / unique words
	terms = tf.keys()
	terms.insert(0,'zerostart')
	terms.append('EOF')
	terms.append('UNK')
	vocab = dict([(i,terms[i]) for i in range(len(terms))])


	return words, vocab


# tokenized unlabelled corpus
# and generate vocabulary word list
# for longer text document (multiple sentences)
################################################
def generateDocVocab(documents):

	
	docwords = []
	docsents = []

	# tokenized words per document
	for text in documents:
		txt = str(text).lower()
		tmp = nltk.word_tokenize(txt.decode('utf-8','ignore'))
		docwords.append(tmp)

	# splitting document into sentences
	for text in docwords:
		tmp = splitSentences(text)
		docsents.append(tmp)

	words = convert(docwords)
	# merge all tokens into one list
	alltokens = sum(docwords,[])

	# frequency occurence of terms
	tf = nltk.FreqDist(alltokens)
	# terms / unique words
	terms = tf.keys()
	terms.insert(0,'zerostart')
	terms.append('EOF')
	terms.append('UNK')
	vocab = dict([(i,terms[i]) for i in range(len(terms))])


	return words, docsents, vocab

def generateDocMail(documents):

	
	subj = dict()
	cont = dict()

	allSubj = dict()
	allCont = dict()
	

	for k in datadictionary:
		subject = []
		content = []
		for mail in datadictionary[k]:
			subjTitle = []
			contWords = []
			
			with open(mail) as f:
				for i,line in enumerate(f):		
					if i == 0:
						tmpSubj = line
						# get tokenized subject title
						lSubj = tmpSubj.lower()
						keyword = 'subject:'
						# get title after keyword
						beforeKey, inKey, afterKey = lSubj.partition(keyword)
						
						#afterKey.encode('ascii', 'ignore')
						# tokenize
						subjTitle = nltk.word_tokenize(afterKey.decode('utf-8', 'ignore'))
					else:
						tmpCont = line.lower()
						
						#tmpCont.encode('ascii', 'ignore')
						# get tokenized content
						contWords = nltk.word_tokenize(tmpCont.decode('utf-8', 'ignore')) 

			subject.append(subjTitle)
			content.append(contWords)

		# escape unicode characters if anys - transform to string format of text
		strSubj = convert(subject)
		strCont = convert(content)

		# these are tokenized words for subject and content of mail
		subj[k] = strSubj
		cont[k] = strCont

	# generate tokens from subject title
	subjTokens = [value for key, value in subj.items()]
	subjAllTokens = sum(subjTokens,[])

	# frequency of word across document corpus
	subjFreq = nltk.FreqDist(itertools.chain(*subjAllTokens)) 
	subjUnique = subjFreq.keys()
	# add 'zero' as the first vocab and 'UNK' as unknown words
	subjUnique.insert(0,'zerostart')
	subjUnique.append('EOF')
	subjUnique.append('UNK')
	# indexing word vocabulary : pairs of (index,word)
	subjVocab=dict([(i,subjUnique[i]) for i in range(len(subjUnique))])
	
	# generate tokens from mail contents
	contentTokens = [value for key, value in cont.items()]
	contentAllTokens = sum(contentTokens,[])

	# frequency of word across document corpus
	contFreq = nltk.FreqDist(itertools.chain(*contentAllTokens)) 
	contUnique = contFreq.keys()
	# add 'zero' as the first vocab and 'UNK' as unknown words
	contUnique.insert(0,'zerostart')
	contUnique.append('EOF')
	contUnique.append('UNK')
	# indexing word vocabulary : pairs of (index,word)
	contVocab=dict([(i,contUnique[i]) for i in range(len(contUnique))])
	

	# encode tokenized of words in document into its index/numerical val. in vocabulary list
	# for subject
	for k in subj:
		numericSubj =[]
		for i in range(len(subj[k])):
			numericSubj.append(wordsToIndex(subjVocab,subj[k][i]))
		allSubj[k] = numericSubj


	# for content of mail 
	for k in cont:
		numericCont =[]
		for i in range(len(cont[k])):
			numericCont.append(wordsToIndex(contVocab,cont[k][i]))
		allCont[k] = numericCont


	return subjVocab, contVocab, subj, cont, allSubj, allCont

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
			doc = doc.lower()
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
		terms.insert(0,'zerostart')
		terms.append('UNK')
		vocab[i] = dict([(j,terms[j]) for j in range(len(terms))])

	
	# save vocabulary list
	savePickle(vocab,'vocabulary')
	# alternative - saving as h5 file
	#saveH5Dict('vocabulary.h5',vocab)

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
	#saveH5Dict('documents.h5',alldocs)

	   

	return tokens, worddocs_freq, vocab, alltokens, alldocs


# creating word vocabulary and training sets
# for monolingual corpora
# general text
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
			doc = doc.lower()
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
	uniqueWords.insert(0,'zerostart')
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
	#saveH5Dict('documents.h5',alldocs)
	   

	return mergedTokens, worddocs_freq, vocab, alltokens, alldocs

# creating word vocabulary and training sets
# for mail data
# specifically data with class label (e.g binary spam - nonspam label)
# text contents of mail also include subject and original content --> need to be separated
################################################
def generateMailVocab(datadictionary):

	subj = dict()
	cont = dict()

	allSubj = dict()
	allCont = dict()
	

	for k in datadictionary:
		subject = []
		content = []
		for mail in datadictionary[k]:
			subjTitle = []
			contWords = []
			
			with open(mail) as f:
				for i,line in enumerate(f):		
					if i == 0:
						tmpSubj = line
						# get tokenized subject title
						lSubj = tmpSubj.lower()
						keyword = 'subject:'
						# get title after keyword
						beforeKey, inKey, afterKey = lSubj.partition(keyword)
						
						#afterKey.encode('ascii', 'ignore')
						# tokenize
						subjTitle = nltk.word_tokenize(afterKey.decode('utf-8', 'ignore'))
					else:
						tmpCont = line.lower()
						
						#tmpCont.encode('ascii', 'ignore')
						# get tokenized content
						contWords = nltk.word_tokenize(tmpCont.decode('utf-8', 'ignore')) 

			subject.append(subjTitle)
			content.append(contWords)

		# escape unicode characters if anys - transform to string format of text
		strSubj = convert(subject)
		strCont = convert(content)

		# these are tokenized words for subject and content of mail
		subj[k] = strSubj
		cont[k] = strCont

	# generate tokens from subject title
	subjTokens = [value for key, value in subj.items()]
	subjAllTokens = sum(subjTokens,[])

	# frequency of word across document corpus
	subjFreq = nltk.FreqDist(itertools.chain(*subjAllTokens)) 
	subjUnique = subjFreq.keys()
	# add 'zero' as the first vocab and 'UNK' as unknown words
	subjUnique.insert(0,'zerostart')
	subjUnique.append('EOF')
	subjUnique.append('UNK')
	# indexing word vocabulary : pairs of (index,word)
	subjVocab=dict([(i,subjUnique[i]) for i in range(len(subjUnique))])
	
	# generate tokens from mail contents
	contentTokens = [value for key, value in cont.items()]
	contentAllTokens = sum(contentTokens,[])

	# frequency of word across document corpus
	contFreq = nltk.FreqDist(itertools.chain(*contentAllTokens)) 
	contUnique = contFreq.keys()
	# add 'zero' as the first vocab and 'UNK' as unknown words
	contUnique.insert(0,'zerostart')
	contUnique.append('EOF')
	contUnique.append('UNK')
	# indexing word vocabulary : pairs of (index,word)
	contVocab=dict([(i,contUnique[i]) for i in range(len(contUnique))])
	

	# encode tokenized of words in document into its index/numerical val. in vocabulary list
	# for subject
	for k in subj:
		numericSubj =[]
		for i in range(len(subj[k])):
			numericSubj.append(wordsToIndex(subjVocab,subj[k][i]))
		allSubj[k] = numericSubj


	# for content of mail 
	for k in cont:
		numericCont =[]
		for i in range(len(cont[k])):
			numericCont.append(wordsToIndex(contVocab,cont[k][i]))
		allCont[k] = numericCont


	return subjVocab, contVocab, subj, cont, allSubj, allCont


# characters for end-of-sentence markers
################################################
def endOfSentence(i, docTokens):
	
	return docTokens[i] in ('.','?','!', ';',':') and (i == len(docTokens) - 1 or not docTokens[i+1] in ('.','?','!', ';', ':'))




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
# the input is datadictionary
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


# the input is aray list of documents (with single sentence or sequence of words)
# either sentences in numerical or text string format 
def calcSentencesStats(arraySentences):

	minArr = []
	maxArr = []
	avgArr = []
	ndocs = []
	n_sentences = len(arraySentences)
	
	minlength = min(len(sent) for sent in arraySentences)
	maxlength = max(len(sent) for sent in arraySentences)
	avglength = sum((len(sent) for sent in arraySentences)) / n_sentences
	

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


## For encoded document (in numeric sequence)
def getSentencesClassNum(datadict):
	sentences = []
	for key in datadict:
		tmp = []
		for i in range(len(datadict[key])):
			tmp=(splitSentences(datadict[key][i]))
		sentences.append(tmp)
	return sentences

## transform to numeric form of sentences
# for monolingual WE corpus
def sentToNum(sentences, vocab):
	numSentences = []
	
	
	# for number of classes in document corpus (e.g. spam and non-spam)
	for i in range(len(sentences)):
		sentdoc = []
		# number of documents within class member
		for j in range(len(sentences[i])):
			sent = []
			# number of sentences within a document
			for k in range(len(sentences[i][j])):
				sent.append(wordsToIndex(vocab, sentences[i][j][k]))
			sentdoc.append(sent)
		numSentences.append(sentdoc)

	return numSentences

## transform to numeric form of sentences
# for Bi-lingual WE corpus
def sentToNumBi(sentences, vocab):
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

	return numSentences

## revert stored numeric sequence of sentences to word sequence format
# for the purpose of splitting document into sentences
# for Mono-lingual WE corpus
def sentToWords(doc, vocab):
	sentences = dict()

	# number of class. e.g: spam vs ham (binary class)
	for i in doc:
		sent = []	
		# number of documents in corresponding class
		for j in range(len(doc[i])):
			# for each document, revert numeric to word sequence
			sent.append(indexToWords(vocab, doc[i][j]))
	
		sentences[i] = sent

	return sentences



## revert stored numeric sequence of sentences to word sequence format
# for the purpose of splitting document into sentences
# for Bi-lingual WE corpus
def sentToWordsBi(doc, vocab):
	sentences = dict()
	_n = 0
	# number of language/class
	for i in doc:

		sent = []
	
		# number of documents in corresponding language / class
		for j in range(len(doc[i])):
			# for each document, revert numeric to word sequence
			sent.append(indexToWords(vocab[_n], doc[i][j]))

		_n += 1
			
		sentences[i] = sent

	return sentences


# return array of sentences / documents in a corpus
# the length of array represents number of documents in corpus 
def getSentencesAll(datadict):
	npArr = dictToArray(datadict)
	dataArr = sum(npArr,[])
	sentences = []
	for i in range(len(dataArr)):
		sentences.append(splitSentences(dataArr[i]))

	return sentences

# for doc2vector input
# creating list of sentences with tags
def createLabelledSentences(wordSentences):

	# labelled sentences with sent_id
	labelledSentences = []
	taggedDocument = namedtuple('labelledSentences', 'words tags')
	for i,text in enumerate(wordSentences):
		tag = ['sent_%s' %i]
		labelledSentences.append(taggedDocument(text,tag))


	return labelledSentences

# transform / join comma separated word sequence to a single string of sentence
# e.g. as input of sklearn TfIdfVectorized  
def sequenceToStr(wordSentences):
	strSentences = []
	for i,text in enumerate(wordSentences):

		# clean text from utf8 character 'u'
		cleanTxt = convert(text)
		sent = ' '.join(cleanTxt)
		strSentences.append(sent)

	return strSentences
	

# for tokenized string text
def printDoc(sentences):

	print("%.2f")

	return joinedStr




def printSentencesClass(classid, docid, sentid, sentences):
	for i in range(len(sentences)):
		if i == classid:
			for j in range(len(sentences[i])):
				if j == docid:
					for k in range(len(sentences[i][j])):
						if k == sentid:
							for txt in sentences[i][j][k]:
								print('%s ' %txt,end='') 





# get number of words per document
# for data training without splitting into sentences 
# to provide time steps parameter for RNN
# e.g. wordStats['nWords'] = 173
def getWordStats(arraySentences):

	wordStats = []
	# ignore null content mail
	wordStats=[len(sent) for sent in arraySentences if len(sent) != 0]

	

	return wordStats


# get statistics (number of sentences per document, number of words per sentence) 
def getDocStats(arraySentences):

	docStats = {}
	#docStats['nSentences']=[len(sent) for sent in numSentences]


	return docStats

# Shuffling sentences ( in every epoch to avoid local minima )
def shuffleSentences(numericSentences):
	sentences = np.array(numericSentences)
	indices = np.arange(len(sentences))
	np.random.shuffle(indices)
	sentences = sentences[indices]

	return sentences


# return a numeric sequence of words in a sentence  into one-hot vector
# this is for last layer with time distributed (3D dimension)
# the text input is whole text of documents (without splitting into sentences)
def matrixVectorization3D(docs,timesteps,vocab_length):
	# create zero matrix
	sequences = np.zeros((len(docs), timesteps, vocab_length))
	# the encoding here is also the index of word in vocabulary list as such 
	# it is the index of one hot vector with '1' value
	for i, text in enumerate(docs):
		for j, word_index in enumerate(text):
			sequences[i, j, word_index] = 1
	return sequences



# transform string class label to nominal 
def stringToNominalClass(datadict):
	numericdict = dict()
	
	i=0
	for k in datadict:
		numericdict[i]=datadict[k]
		i +=1
	return numericdict 


#
def dictToArray(datadict):

	list1 = []
	for key in datadict:
		list1.append(datadict[key])

	npArr = np.array(list1)
	return npArr


# return dictionary to nested array
# this is for datadictionary (sequence of words and sentences in document corpus)
def docDictToArray(numericdict):
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
# this is for datadictionary (sequence of words and sentences in document corpus)
def docDictMrgArray(numericdict):
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

# find saved weights
def findWeights(folder):
	findFile = [f for f in os.listdir(folder) if 'weights' in f]
	if len(findFile) == 0:
		return []
	modified_time = [os.path.getmtime(f) for f in findFile]
	lastEpoch = findFile[np.argmax(modified_time)]
	return lastEpoch


