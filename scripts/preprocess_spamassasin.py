# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "20.05.2017"
#__update__ = "09.09.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
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


# this function only applies for subject tokenized text	
def _encodeText(dataDictionary, vocab):

	# encode tokenized of words in document into its index/numerical val. in vocabulary list
	# the input is in dictionary format

	encodedText = dict()

	for k in dataDictionary:
		numericText =[]
		for i, arrTokens in enumerate(dataDictionary[k]):
			numericText.append(wordsToIndex(vocab,arrTokens))
		encodedText[k] = numericText

	return encodedText

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

if __name__ == '__main__':

	# get list of data files
	filenames = listData(SPAMAS_PATH)
	# grouped by class
	datadict = getClassLabel(filenames)

	'''
	Statistic of Spamassasin data sets
	len(filenames) = 9349
	len(datadict['spam']) = 2398
	len(datadict['easy_ham']) = 6451
	len(datadict['hard_ham']) = 500

	'''


	# return text of subject and content of mail (since Spamassasin data is raw ones and contain alot of noises)
	subjectMail = extractSubject(datadict)
	contentMail = extractContent(datadict)

	# clean content from HTML entities, css, javascript
	# this is in a form of array list of sentences as values of dictionary 
	contentMailCleaned = cleanHtmlEntities(contentMail)

	##########################
	# Tokenize subject title
	##########################

	# 1. clean from consecutive punctuations
	# 2. remove punctuations other than EOS characters (if anys)
	# 3. tokenize

	newSubjects = dict()
	newSubjectSW = dict()
	for i in subjectMail:
		textTokens = []
		textTokenSW = []
		for j, text in enumerate(subjectMail[i]):
			tmp1 = removeDuplicatePunctuations(text)
			tmp2 = cleanPunctuations(tmp1)
			tokens, tokenSW = textToTokens(tmp2)
			textTokens.append(tokens)
			textTokenSW.append(tokenSW)
		newSubjects[i] = textTokens
		newSubjectSW[i] = textTokenSW


	##########################
	# Tokenize content of mail
	##########################

	
	# clean from consecutive punctuations
	# for each class (spam, ham, ...)
	newContent = dict()
	for i in contentMailCleaned:
		newSent = []
		# for each mail content (array list of sentences)
		for j, arrSentences in enumerate(contentMailCleaned[i]):
			newText = []
			for k,text in enumerate(arrSentences):
				newText.append(removeDuplicatePunctuations(text))
			newSent.append(newText)
		newContent[i] = newSent

			
	

	# 1. remove punctuations other than EOS characters (if anys)
	# 2. and tokenize documents (list of sentences in this case)
	# 3. and check whether the tokenized array of sentences contain more than 1 sentence (by checking the occurrence of EOS tokens in list)
	

	newTokenized = dict()
	newTokenizedSW = dict()
	for i in newContent:
		newTokens = []
		newTokenSW = []
		for j, arrSentences in enumerate(newContent[i]):
			newText = []
			newTextSW = []
			for k, text in enumerate(arrSentences):
				# clean from punctuations
				tmp = cleanPunctuations(text)
				# if sentence only contain punctuation as such it becomes an empty line
				if len(tmp) == 0:
					continue
				else:
					# tokenize sentences
					tokens, tokenSW = textToTokens(tmp)

					# check whether the token list consist of more than 1 sentence
					# if anys, split the list again into sub sentences

					textTokens = []
					textTokenSW = []
					begin = 0
					beginSW = 0

					# without stopwords elimination and stemming
					for l, word in enumerate(tokens):
						if(endOfSentence(l,tokens)):
							textTokens.append(tokens[begin:l+1])
							begin = l+1
					if begin < len(tokens):
						textTokens.append(tokens[begin:])

					# with stopwords elimination and stemming
					for m, word in enumerate(tokenSW):
						if(endOfSentence(m,tokenSW)):
							textTokenSW.append(tokenSW[beginSW:m+1])
							beginSW = m+1
					if beginSW < len(tokenSW):
						textTokenSW.append(tokenSW[beginSW:])

				newText.extend(textTokens)
				newTextSW.extend(textTokenSW)

			newTokens.append(newText)
			newTokenSW.append(newTextSW)

		newTokenized[i] = newTokens
		newTokenizedSW[i] = newTokenSW


	
	# check whether there is empty subject title or content
	# often due to the mail only contains html tags or stopwords in title
	newDict = dict()
	newTokenSubj = dict()
	newTokenSubjSW = dict()
	newTokenCont = dict()
	newTokenContSW = dict()

	indSubject = []
	indSubjectSW = []
	# without stopword elimination and stemming
	for k in newSubjects:
		for i,text in enumerate(newSubjects[k]):
			if len(text) == 0:
				indSubject.append((k,i))

	# with stopword elimination and stemming
	for k in newSubjectSW:
		for i,text in enumerate(newSubjectSW[k]):
			if len(text) == 0:
				indSubjectSW.append((k,i))


	# the content of mail is array list of tokenized sentences
	# as such the depth of loops is more than tokenized subject title

	indContent = []
	indContentSW = []
	# without stopword elimination and stemming
	for i in newTokenized:
		for j,arrSentences in enumerate(newTokenized[i]):
			if len(arrSentences) == 0:
				indContent.append((i,j))

	# with stopword elimination and stemming
	for i in newTokenizedSW:
		for j,arrSentences in enumerate(newTokenizedSW[i]):
			if len(arrSentences) == 0:
				indContentSW.append((i,j))

	# update information in data dictionary - as such we remove empty subjects and content
	newIndSubj = unionList(indSubject,indSubjectSW)
	newIndCont = unionList(indContent,indContentSW)
	newIndex = unionList(newIndSubj,newIndCont)
	newIndex.sort()

	if len(newIndex) != 0:

		# updating subjects (without stopword elimination and stemming)
		for i in newSubjects:
			arrItem = []
			for j in range(len(newSubjects[i])):
				if (i,j) not in newIndex:
					arrItem.append(newSubjects[i][j])
			newTokenSubj[i] = convert(arrItem)

		# updating subjects (without stopword elimination and stemming)
		for i in newSubjectSW:
			arrItem = []
			for j in range(len(newSubjectSW[i])):
				if (i,j) not in newIndex:
					arrItem.append(newSubjectSW[i][j])
			newTokenSubjSW[i] = convert(arrItem)

		# updating mail content (without stopword elimination and stemming)
		for i in newTokenized:
			arrItem = []
			for j in range(len(newTokenized[i])):
				if (i,j) not in newIndex:
					arrItem.append(newTokenized[i][j])
			newTokenCont[i] = convert(arrItem)

		# updating mail content (with stopword elimination and stemming)
		for i in newTokenizedSW:
			arrItem = []
			for j in range(len(newTokenizedSW[i])):
				if (i,j) not in newIndex:
					arrItem.append(newTokenizedSW[i][j])
			newTokenContSW[i] = convert(arrItem)

		# updating data dictionary
		# information about training and test sets (original documents)
		for i in datadict:
			arrItem = []
			for j in range(len(datadict[i])):
				if (i,j) not in newIndex:
					arrItem.append(datadict[i][j])
			newDict[i] = arrItem

	else:
		newTokenSubj = newSubjects
		newTokenSubjSW = newSubjectSW
		newTokenCont = newTokenized
		newTokenContSW = newTokenizedSW
		newDict = datadict

	'''

	In [165]: len(newTokenSubj['easy_ham'])
	Out[165]: 6449

	In [166]: len(newTokenSubj['hard_ham'])
	Out[166]: 500

	In [167]: len(newTokenSubj['spam'])
	Out[167]: 2379
	'''

	####################################################
	# Generate vocab from subject title
	####################################################

	subjVocab, subjVocabTF = indexVocab(newTokenSubj)
	subjSWVocab, subjSWVocabTF = indexVocab(newTokenSubjSW)

	'''
	In [17]: len(subjVocab)
	Out[17]: 7328

	In [21]: len(subjVocabTF)
	Out[21]: 7325

	In [23]: len(subjSWVocab)
	Out[23]: 6182

	In [24]: len(subjSWVocabTF)
	Out[24]: 6179



	'''

	####################################################
	# Generate vocab from mail content
	####################################################

	mergeCont = dict()
	mergeContSW = dict()
	# merge splitted tokens of sentences into one

	# mail content without stopword elimination and stemming
	for i in newTokenCont:
		arrTokens = []
		for j, arrSentences in enumerate(newTokenCont[i]):
			mergeArr = sum(arrSentences,[])
			arrTokens.append(mergeArr)
		mergeCont[i] = arrTokens

	# mail content with stopword elimination and stemming
	for i in newTokenContSW:
		arrTokens = []
		for j, arrSentences in enumerate(newTokenContSW[i]):
			mergeArr = sum(arrSentences,[])
			arrTokens.append(mergeArr)
		mergeContSW[i] = arrTokens


	# vocab for tokens without stopword elimination and stemming
	contVocab, contVocabTF = indexVocab(mergeCont)

	# vocab for tokens with stopword elimination and stemming
	contSWVocab, contSWVocabTF = indexVocab(mergeContSW)

	'''
	In [36]: len(contVocab)
	Out[36]: 120774

	In [37]: len(contVocabTF)
	Out[37]: 120771

	In [39]: len(contSWVocab)
	Out[39]: 105089

	In [40]: len(contSWVocabTF)
	Out[40]: 105086


	'''


	'''

	####################################################
	# Encoding tokenized subject title into numbers
	####################################################


	# encode tokenized documents into sequence of numbers

	# without stopword eliminating and stemming
	encodedSubject = _encodeText(newTokenSubj, subjVocab)
	# with stopword eliminating and stemming
	encodedSubjectSW = _encodeText(newTokenSubjSW, subjSWVocab)


	####################################################
	# Encoding tokenized mail content into numbers
	####################################################

	encodedContent = dict()
	encodedContentSW = dict()

	for i in newTokenCont:
		textSentences = []
		for j, arrSentences in enumerate(newTokenCont[i]):
			textTokens = []
			for k, arrTokens in enumerate(arrSentences):
				textTokens.append(wordsToIndex(contVocab,arrTokens))
			textSentences.append(textTokens)
		encodedContent[i] = textSentences	

	for i in newTokenContSW:
		textSentences = []
		for j, arrSentences in enumerate(newTokenContSW[i]):
			textTokens = []
			for k, arrTokens in enumerate(arrSentences):
				textTokens.append(wordsToIndex(contSWVocab,arrTokens))
			textSentences.append(textTokens)
		encodedContentSW[i] = textSentences

	# save encoded version of tokenized documents
	savePickle(encodedSubject,'spamas_encodedSubject')
	savePickle(encodedSubjectSW,'spamas_encodedSubjectSW')
	savePickle(encodedContent,'spamas_encodedContent')
	savePickle(encodedContentSW,'spamas_encodedContentSW')

	'''


	# save pre-processed text of mail's subject title and content
	# original data sets
	savePickle(subjectMail,'spamas_textSubjects')
	savePickle(contentMailCleaned,'spamas_textContent')

	# store information - updating data after reduction
	savePickle(newIndex,'spamas_emptyIndex')
	savePickle(newDict,'spamas_dataDict')


	# save tokenized documents 
	# for subject	
	savePickle(newTokenSubj,'spamas_tokenSubj')
	savePickle(newTokenSubjSW,'spamas_tokenSubjSW')
	# For mail content, 2 versions are stored:
	# content without splitting into sentences
	savePickle(mergeCont,'spamas_mergeTokenCont')
	savePickle(mergeContSW,'spamas_mergeTokenContSW')
	# content with splitting into tokenized sentences
	savePickle(newTokenCont,'spamas_tokenCont')
	savePickle(newTokenContSW,'spamas_tokenContSW')


	# save vocabulary list
	# for subject
	savePickle(subjVocab,'spamas_subjVocab')
	savePickle(subjVocabTF,'spamas_subjVocabTF')
	savePickle(subjSWVocab,'spamas_subjSWVocab')
	savePickle(subjSWVocabTF,'spamas_subjSWVocabTF')

	# for content
	savePickle(contVocab,'spamas_contVocab')
	savePickle(contVocabTF,'spamas_contVocabTF')
	savePickle(contSWVocab,'spamas_contSWVocab')
	savePickle(contSWVocabTF,'spamas_contSWVocabTF')

	'''

	In [33]: len(spamas_contVocab)
	Out[33]: 120774


	In [38]: len(spamas_subjVocab[0])
	Out[38]: 7328

	In [39]: len(spamas_subjVocab[1])
	Out[39]: 7325



	'''


	


