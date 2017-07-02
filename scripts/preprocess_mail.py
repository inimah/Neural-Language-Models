# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "28.06.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import os
import sys
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from text_preprocessing import *
from language_models import *
from keras.callbacks import Callback
from bs4 import BeautifulSoup

nltk.data.path.append('/opt/data/nltk_data')


PATH = '/opt/data/phishing_mails/content'


# to check length of body between 3 attributes of body
def body_len(text1, text2):
	if len(text1) > 3:
		return text1
	else:
		return text2

def clean_html_javascript(text):
	soup = BeautifulSoup(text, "lxml")

	for script in soup(["script", "style"]):
		script.extract()

	txt = soup.get_text()
	# break into lines
	lines = (line.strip() for line in txt.splitlines())
	# break multi-headlines into a line
	chunks = (phrase.strip() for line in lines for phrase in line.split(" "))

	txt = '\n'.join(chunk for chunk in chunks if chunk)

	return txt

if __name__ == '__main__':

	# get list of data files
	filenames = listData(PATH)

	data={}

	for path in filenames:
		filepath = os.path.basename(path)
		fileName, fileExtension = os.path.splitext(filepath)
		fname, fdate = split_at(fileName, '_', 2)
		data[fdate] = extractData(path)	


	ind = 0 
	for i in data:
		ind += len(data[i])


	mail = pd.DataFrame()
	yearmonth = []
	subject = []
	body1 = []
	body2 = []
	sender = []

	for i in data:	
		for j in range(len(data[i])):
			yearmonth.append(i)
			subject.append(data[i]['SUBJECT2'][j])
			sender.append(data[i]['SENDER_EMAIL'][j])
			body1.append(data[i]['BODY'][j])
			body2.append(data[i]['BODY_HTML'][j])
		

	mail['yearmonth'] = yearmonth
	mail['subject'] = subject
	mail['sender'] = sender
	mail['body1'] = body1
	mail['body2'] = body2

	

	mail['body'] = mail.apply(lambda x: body_len(str(x['body1']),str(x['body2'])), axis=1) 

	cleanmail = mail[(mail['subject'] != '') & (mail['body'] != '')]

	cleanmail['clean1']= cleanmail['body'].apply(lambda x: clean_html_javascript(str(x)))
	cleanmail['clean2']= cleanmail['clean1'].apply(lambda x: x.encode('utf-8').replace('\r', ' ').replace('\n', ' '))
	cleanmail['txtbody']= cleanmail['clean2'].apply(lambda x:x.decode("utf-8").encode("ascii", "ignore"))


	maildata = cleanmail[['yearmonth', 'sender', 'subject','txtbody']] 
	maildata = maildata.reset_index(drop=True)
	maildata.to_csv('maildata.csv', index=True, header = True, sep='\t')

	timeset = set(yearmonth)
	timegrp = list(sorted(timeset))

	allmails = {}

	for i in timegrp:
		allmails[i] = maildata[maildata['yearmonth']==i]
		allmails[i].to_csv('mail_%s.csv' %i, index=True, header = True, sep='\t')

	subjects = maildata['subject']
	senders = maildata['sender']
	contents = maildata['txtbody']

	'''
	savePickle(subjects, 'allSubjects')
	savePickle(senders, 'allSenders')
	savePickle(contents, 'allContent')

	####################################
	# for mail subjects

	# generating vocab
	subject_sent, subject_vocab = generateSentVocab(subjects)
	savePickle(subject_vocab,'subject_vocab')
	savePickle(subject_sent,'subject_sentences')


	# word2vec model of mail subjects
	w2v_subj_sg, w2v_subj_cbow, w2v_subj_embed_sg, w2v_subj_embed_cbow = wordEmbedding(subject_sent, subject_vocab, 200, 50)


	# create document representation of word vectors (sentence embedding)

	# By averaging word vectors
	avg_subj_embed1 = averageWE(w2v_subj_sg, subject_sent)
	avg_subj_embed2 = averageWE(w2v_subj_cbow, subject_sent)

	# By sequential model : shallow encoder
	

	# By sequential model : hierarchical encoder

	####################################
	# for mail contents

	content_words, content_sents, content_vocab = generateDocVocab(contents)
	savePickle(content_vocab,'content_vocab')

	# word2vec model of mail subjects
	w2v_cont_sg, w2v_cont_cbow, w2v_cont_embed_sg, w2v_cont_embed_cbow = wordEmbedding(content_sents, content_vocab, 200, 50)


	# create document representation of word vectors (sentence embedding)

	# By averaging word vectors
	avg_cont_embed1 = averageWE(w2v_cont_sg, content_sents)
	avg_cont_embed2 = averageWE(w2v_cont_cbow, content_sents)

	'''



