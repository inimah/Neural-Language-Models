# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "20.05.2017"
#__update__ = "09.09.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
sys.path.insert(0,'..')
import string
import numpy as np
import pandas as pd
from itertools import groupby
from string import punctuation
from text_preprocessing import *
from bs4 import BeautifulSoup
from lsa.tokenizer import Tokenizer



SPAMAS_PATH = '/home/inimah/exp/data/maildata/spamassasin'

#SPAMAS_PATH = '../data/spamassasin'

'''
main function : _tokenizedText(), _indexVocab() is in text_preprocessing
'''


def _encodeText(tokenized_docs, vocab):

	# encode tokenized of words in document into its index/numerical value in vocabulary list
	# the input is in array list tokenized documents

	encoded_docs = []

	for i, arrTokens in enumerate(tokenized_docs):
		encoded_docs.append(wordsToIndex(vocab,arrTokens))

	return encoded_docs

def _encodeLabelledText(tokenized_docs, vocab):

	# encode tokenized of words in document into its index/numerical value in vocabulary list
	# the input is in array list tokenized documents

	encoded_docs = []

	for i, data in enumerate(tokenized_docs):
		encoded_docs.append((data[0],wordsToIndex(vocab,data[1])))

	return encoded_docs

# from labelled tokenized documents
def _countWord(tokenized_docs):
	count_words = []
	for i,data in enumerate(tokenized_docs):
		count_ = len(data[1])
		count_words.append((data[0],count_))
	return count_words


def extractSubject(datadictionary):

	subjectMail = dict()
	for k in datadictionary:
		subject = []
		for mail in datadictionary[k]:			
			with open(mail) as f:
				for i,line in enumerate(f):
					# extract subject title
					if "Subject:" in line:
						tmpSubj = line
						# get tokenized subject title
						lSubj = tmpSubj.lower()
						keyword = 'subject:'
						# get title after keyword
						beforeKey, inKey, afterKey = lSubj.partition(keyword)

				subject.append(afterKey)
				
		subjectMail[k] = (subject)
		

	return subjectMail


def extractContent(datadictionary):

	
	contentMail = dict()

	for k in datadictionary:
		content = []
		
		for mail in datadictionary[k]:
			textContent = ""
			lineCont = 0
			
			with open(mail) as f:
				for i,line in enumerate(f):

					if len(line.strip()) == 0 :

						lineCont = i 
						break
			with open(mail) as f:
				for i,line in enumerate(f):
					if i >= lineCont:
						textContent += line.lower()
				
					# extract the content of mail
					
		
				content.append(textContent)
		contentMail[k] = (content)

	return contentMail

# remove punctuation except the ones that identified EOS 
def cleanPunctuations(text):

	eos_punct = ['.','?','!', ';',':','/']
	punc = set(punctuation) - set(p for p in eos_punct)
	newtext = ""
	for word in text:
		if word not in punc:
			newtext += word

	return newtext


# clean text /line from repeated consecutive punctuation (removes duplicates)
def removeDuplicatePunctuations(text):

	punc = set(punctuation) - set('.')
	newtext = []
	for k, g in groupby(text):
		if k in punc:
			newtext.append(k)
		else:
			newtext.extend(g)

	textClean = ''.join(newtext)

	return textClean

def cleanHtmlEntities(contentMail):

	cleanContent = dict()

	for k in contentMail:
		cont = []
		for l, text in enumerate(contentMail[k]):
			rawSentences = htmlToTextBS(text)
			cont.append(rawSentences)

		cleanContent[k] = cont

	return cleanContent



def unionList(list1,list2):
	for (a,b) in list2:
		if (a,b) not in list1: 
			list1.append((a,b))


	return list1

# this will return documents as whole tokenized words from aray of sentences 
def mergeSentences(tokenized_docs):

	
	mergeTokens = dict()
	# merge splitted tokens of sentences into one

	# mail content without stopword elimination and stemming
	for i in tokenized_docs:
		arrTokens = []
		for j, arrSentences in enumerate(tokenized_docs[i]):
			mergeArr = sum(arrSentences,[])
			arrTokens.append(mergeArr)
		mergeTokens[i] = arrTokens

	return mergeTokens



def reduceVocab(term_frequency_vocab):
	lessFreq_vocab=[]  
	wordIndex = []
	for k,v in term_frequency_vocab.iteritems():
		if v<3:
			lessFreq_vocab.append((k,v))
		else:
			wordIndex.append(k)
	wordIndex.insert(0,'SOF')
	wordIndex.append('EOF')
	wordIndex.append('UNK')
	reduced_vocab=dict([(i,wordIndex[i]) for i in range(len(wordIndex))])

	return reduced_vocab, lessFreq_vocab

# remove less frequent vocab from documents
def removeLessFreqVocab(tokenized_docs,reduced_vocab):

	reduced_docs= dict()
	for i in tokenized_docs:
		tmp_docs = []
		for j, tokens in enumerate(tokenized_docs[i]):
			tmp_tokens = []
			for k, word in enumerate(tokens):
				if word in reduced_vocab.values():
					tmp_tokens.append(word)
			tmp_docs.append(tmp_tokens)
		reduced_docs[i] = tmp_docs

	return reduced_docs

# for documents with splitted sentences
def removeLessFreqSent(tokenized_docs,reduced_vocab):

	reduced_docs= dict()
	for i in tokenized_docs:
		tmp_docs = []
		for j, arrSentences in enumerate(tokenized_docs[i]):
			tmp_sent = []
			for k, tokens in enumerate(arrSentences):
				tmp_tokens = []
				for l, word in enumerate(tokens):
					if word in reduced_vocab.values():
						tmp_tokens.append(word)
				tmp_sent.append(tmp_tokens)
			tmp_docs.append(tmp_sent)
		reduced_docs[i] = tmp_docs

	return reduced_docs


# discard less frequent words in tokenized documents based on new vocabulary list
# and ignore/discard documents with length of sentences > 100 sentences
# and discard array list of empty sentence and < 3 word tokens (preserve sentences with 4~25 word tokens)
def removeSentences(content_sent,vocab):

	doc_index = []
	sent_index = []
	new_docs = dict()
	for i in content_sent:
		tmp_docs = []
		for j, arrSentences in enumerate(content_sent[i]):
			# discard documents with > 100 sentences
			if len(arrSentences) <= 100:
				tmp_sent = []
				for k, tokens in enumerate(arrSentences):
					if (len(tokens) > 3 and len(tokens) <= 25):
						tmp_tokens = []
						for l, word in enumerate(tokens):
							if word in vocab.values():
								tmp_tokens.append(word)
						tmp_sent.append(tmp_tokens)
					else:
						sent_index.append((i,j,k))
				tmp_docs.append(tmp_sent)
			else:
				doc_index.append((i,j))

		new_docs[i] = tmp_docs

	return new_docs, sent_index, doc_index

# the input is in dictionary format of tokenized documents
# for short text input (without splitting into sentences)
def generateTrainingSets(tokenized_docs):

	labelled_data = []
	for i in tokenized_docs:
		for j, tokens in enumerate(tokenized_docs[i]):
			labelled_data.append((i,tokens))

	
	return labelled_data



if __name__ == '__main__':

	contVocabTF=readPickle('spamas_contVocabTF')
	contSWVocabTF=readPickle('spamas_contSWVocabTF')
	newTokenCont = readPickle('spamas_tokenCont')
	newTokenContSW = readPickle('spamas_tokenContSW')





	# discard words with frequency less than 5
	# also discard less frequent words in tokenized documents based on new vocabulary list

	reduced_contvocab, lessFreq_contvocab = reduceVocab(contVocabTF)
	reduced_cont = removeLessFreqSent(newTokenCont,reduced_contvocab)

	# length of vocab after being discarded
	
	# len(reduced_contvocab) : 39389
	# len(spamas_contVocabTF) : 106361

	
	reduced_contvocabSW, lessFreq_contvocabSW = reduceVocab(contSWVocabTF)
	reduced_contSW = removeLessFreqSent(newTokenContSW,reduced_contvocabSW)


	
	#In [6]: len(vocabSW)
	#Out[6]: 2467

	savePickle(reduced_contvocab,'spamas_reduced_contvocab')
	savePickle(reduced_contvocabSW,'spamas_reduced_contvocabSW')
	savePickle(lessFreq_contvocab,'spamas_lessFreq_contvocab')
	savePickle(lessFreq_contvocabSW,'spamas_lessFreq_contvocabSW')
	savePickle(reduced_cont,'spamas_reduced_cont')
	savePickle(reduced_contSW,'spamas_reduced_contSW')

	# remove documents with > 100 sentences and preserve sentence with 4~25 words - otherwise discards

	fin_cont, removed_sent_index, removed_doc_index = removeSentences(reduced_cont,reduced_contvocab)
	fin_contSW, removed_sent_indexSW, removed_doc_indexSW = removeSentences(reduced_contSW,reduced_contvocabSW)

	savePickle(fin_cont,'spamas_fin_cont')
	savePickle(fin_contSW,'spamas_fin_contSW')
	savePickle(removed_sent_index,'spamas_fin_removed_sent_index')
	savePickle(removed_sent_indexSW,'spamas_fin_removed_sent_indexSW')
	savePickle(removed_doc_index,'spamas_fin_removed_doc_index')
	savePickle(removed_doc_indexSW,'spamas_fin_removed_doc_indexSW')

	labelled_cont= generateTrainingSets(fin_cont)
	labelled_contSW= generateTrainingSets(fin_contSW)

	savePickle(labelled_cont,'spamas_fin_labelled_cont')
	savePickle(labelled_contSW,'spamas_fin_labelled_contSW')





	


