# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "13.06.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function

import gzip
import os
import sys
import wget
import tarfile
import zipfile
from scripts.text_preprocessing import *


# destination_path : name of directory. e.g data
MAILDATA_URL = "https://storage.googleapis.com/trl_data/maildata/raw/maildata.zip"
NOVELS_URL = "https://storage.googleapis.com/trl_data/novels/novel_datasets.zip"
SHAKESPEARE_URL = "https://storage.googleapis.com/trl_data/shakespeare/shakespeare_text.zip"
EUROPARL_URL = "https://storage.googleapis.com/trl_data/multilingual/europarl/nl-en.zip"
NLWIKI_URL = "https://storage.googleapis.com/trl_data/multilingual/wiki_polygot/raw/nl_wiki_text.tar.lzma"
ENWIKI_URL = "https://storage.googleapis.com/trl_data/multilingual/wiki_polygot/raw/en_wiki_text.tar.lzma"

import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-url', default='europarl')
args = vars(ap.parse_args())
URL = args['url']





def getData(url, destination_path):

	wget.download(url, destination_path)
	#filename =  os.path.basename(url)
    filepath = listData(destination_path)

	return filepath

if __name__ == '__main__':

	# make sure destination path exists

	# download raw europarl data

	europarl_path = getData(EUROPARL_URL,data/europarl)

	# download raw mail data sets (enron, lingspam, spamassasin)

	mail_path = getData(MAILDATA_URL,data/maildata)


