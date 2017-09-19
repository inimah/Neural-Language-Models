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

# the input is in dictionary format of tokenized documents
# for short text input (without splitting into sentences)
def generateTrainingSets(tokenized_docs):

	labelled_data = []
	for i in tokenized_docs:
		for j, tokens in enumerate(tokenized_docs[i]):
			labelled_data.append((i,tokens))

	
	x_data = []
	y_data = []
	for i, data in enumerate(labelled_data):
		y_data.append(data[0])
		x_data.append(data[1])

	return labelled_data, x_data, y_data 

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

	

	# save pre-processed text of mail's subject title and content
	# original data sets
	savePickle(subjectMail,'spamas_textSubjects')
	

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

	# clean content from HTML entities, css, javascript
	# this is in a form of array list of sentences as values of dictionary 
	contentMailCleaned = cleanHtmlEntities(contentMail)
	savePickle(contentMailCleaned,'spamas_textContent')

	
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


	# store information - updating data after reduction
	savePickle(newIndex,'spamas_emptyIndex')
	savePickle(newDict,'spamas_dataDict')


	# save tokenized documents 
	# for subject	
	# this is before reducing by TF, number of sentences, and number of words per sentence
	savePickle(newTokenSubj,'spamas_tokenSubj')
	savePickle(newTokenSubjSW,'spamas_tokenSubjSW')

	# content with splitting into tokenized sentences
	savePickle(newTokenCont,'spamas_tokenCont')
	savePickle(newTokenContSW,'spamas_tokenContSW')

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

	# save vocabulary list
	# for subject
	savePickle(subjVocab,'spamas_subjVocab')
	savePickle(subjVocabTF,'spamas_subjVocabTF')
	savePickle(subjSWVocab,'spamas_subjSWVocab')
	savePickle(subjSWVocabTF,'spamas_subjSWVocabTF')


	
	# discard words with frequency less than 5
	# also discard less frequent words in tokenized documents based on new vocabulary list

	reduced_vocab, lessFreq_vocab = reduceVocab(subjVocabTF)
	reduced_subject = removeLessFreqVocab(newTokenSubj,reduced_vocab)

	# length of vocab after being discarded
	
	#In [5]: len(vocab)
	#Out[5]: 2814
	
	reduced_vocabSW, lessFreq_vocabSW = reduceVocab(subjSWVocabTF)
	reduced_subjectSW = removeLessFreqVocab(newTokenSubjSW,reduced_vocabSW)


	
	#In [6]: len(vocabSW)
	#Out[6]: 2467

	savePickle(reduced_vocab,'spamas_reducedVocab')
	savePickle(reduced_vocabSW,'spamas_reducedVocabSW')
	savePickle(lessFreq_vocab,'spamas_lessFreq_vocab')
	savePickle(lessFreq_vocabSW,'spamas_lessFreq_vocabSW')
	savePickle(reduced_subject,'spamas_reducedTokenSubj')
	savePickle(reduced_subjectSW,'spamas_reducedTokenSubjSW')


	# create labelled instances for training sets

	labelled_subj= generateTrainingSets(reduced_subject)
	labelled_subjSW= generateTrainingSets(reduced_subjectSW)

	savePickle(labelled_subj,'spamas_labelled_subj')
	savePickle(labelled_subjSW,'spamas_labelled_subjSW')


	# check the statistics of data as such documents with longer sentences (with noises) will be discarded

	'''
	count_words = _countWord(labelled_subj)
	# w1 = 'spam'
	w1 = []
	# w2 = 'easy_ham'
	w2 = []
	# w3 = 'hard_ham'
	w3 = []
	for i, data in enumerate(count_words):
		if data[0] == 'spam':
			w1.append(data[1])
		elif data[0] == 'easy_ham':
			w2.append(data[1])
		elif data[0] == 'hard_ham':
			w3.append(data[1])

	# for spam
	max_spam = max(w1)
	avg_spam = sum(w1)/len(w1)

	# max_spam = 2621
	# avg_spam = 6

	# for easy ham
	max_easyham = max(w2)
	avg_easyham = sum(w2)/len(w2)
	# max_easyham = 22
	# avg_easyham = 6

	# for hard ham
	max_hardham = max(w3)
	avg_hardham = sum(w3)/len(w3)
	# max_hardham = 13
	# avg_hardham = 5


	aw1 = sns.distplot(w1)
	fig_w1 = aw1.get_figure()
	fig_w1.savefig('spam_words_per_subj.png')
	fig_w1.clf()
	aw2 = sns.distplot(w2)
	fig_w2 = aw2.get_figure()
	fig_w2.savefig('easyham_words_per_subj.png')
	fig_w2.clf()
	aw3 = sns.distplot(w3)
	fig_w3 = aw3.get_figure()
	fig_w3.savefig('hardham_words_per_subj.png')
	fig_w3.clf()


	'''


	# discard subjects with number of words > 25 (as being seen in statistics of subject title)
	fin_subjects = []
	for i, data in enumerate(labelled_subj):
		if len(data[1]) <= 25:
			fin_subjects.append((data[0],data[1]))

	# save reduced versioned of labelled tokenized documents
	savePickle(fin_subjects,'spamas_fin_labelled_subj')

	# Encode text into numerical tokenized format
	fin_encoded_subj = _encodeLabelledText(fin_subjects,reduced_vocab)
	savePickle(fin_encoded_subj,'spamas_fin_encoded_subj')

	# check statistic of each class (maximum - average number of words per class)
	fin_count_subjwords = _countWord(fin_encoded_subj)
	savePickle(fin_count_subjwords,'spamas_fin_count_subjwords')

	fin_subjectsSW = []
	for i, data in enumerate(labelled_subjSW):
		if len(data[1]) <= 25:
			fin_subjectsSW.append((data[0],data[1]))

	# save reduced versioned of labelled tokenized documents
	savePickle(fin_subjectsSW,'spamas_fin_labelled_subjSW')

	# Encode text into numerical tokenized format
	fin_encoded_subjSW = _encodeLabelledText(fin_subjectsSW,reduced_vocabSW)
	savePickle(fin_encoded_subjSW,'spamas_fin_encoded_subjSW')

	# check statistic of each class (maximum - average number of words per class)
	fin_count_subjwordsSW = _countWord(fin_encoded_subjSW)
	savePickle(fin_count_subjwordsSW,'spamas_fin_count_subjwordsSW')


	####################################################
	# Generate vocab from mail content
	####################################################

	# merge splitted tokens of sentences into one

	# mail content without stopword elimination and stemming
	mergeCont = mergeSentences(newTokenCont)

	# mail content with stopword elimination and stemming
	mergeContSW = mergeSentences(newTokenContSW)
	


	# vocab for tokens without stopword elimination and stemming
	contVocab, contVocabTF = indexVocab(mergeCont)

	# vocab for tokens with stopword elimination and stemming
	contSWVocab, contSWVocabTF = indexVocab(mergeContSW)



	# For mail content, 2 versions are stored:
	# content without splitting into sentences
	savePickle(mergeCont,'spamas_mergeTokenCont')
	savePickle(mergeContSW,'spamas_mergeTokenContSW')
	

	# vocabulary for content
	savePickle(contVocab,'spamas_contVocab')
	savePickle(contVocabTF,'spamas_contVocabTF')
	savePickle(contSWVocab,'spamas_contSWVocab')
	savePickle(contSWVocabTF,'spamas_contSWVocabTF')


	# discard words with frequency less than 5
	# also discard less frequent words in tokenized documents based on new vocabulary list

	reduced_vocab, lessFreq_vocab = reduceVocab(subjVocabTF)
	reduced_subject = removeLessFreqVocab(newTokenSubj,reduced_vocab)

	# length of vocab after being discarded
	
	#In [5]: len(vocab)
	#Out[5]: 2814
	
	reduced_vocabSW, lessFreq_vocabSW = reduceVocab(subjSWVocabTF)
	reduced_subjectSW = removeLessFreqVocab(newTokenSubjSW,reduced_vocabSW)


	
	#In [6]: len(vocabSW)
	#Out[6]: 2467

	savePickle(reduced_vocab,'spamas_reducedVocab')
	savePickle(reduced_vocabSW,'spamas_reducedVocabSW')
	savePickle(lessFreq_vocab,'spamas_lessFreq_vocab')
	savePickle(lessFreq_vocabSW,'spamas_lessFreq_vocabSW')
	savePickle(reduced_subject,'spamas_reducedTokenSubj')
	savePickle(reduced_subjectSW,'spamas_reducedTokenSubjSW')




	


