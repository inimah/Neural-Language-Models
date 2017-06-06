# Cross-Language Embeddings for Document Classification

The series of experiments focus on classifying mail data sets using Mono-Language and Cross-Language Embeddings. The main purpose for this cross-language experiment is to transfer the learning from resource rich language (labelled English mail data sets) to resource-poor language (unlabelled Dutch mail data sets - suspected as phising mails).

version 1.0:
- For sequence-to-sequence learning of bi-lingual documents, a parallel corpus of English and Dutch document is used http://www.statmt.org/europarl/. A compact version of pre-processed data (python dictionary format) will be shared in data/.
- The resulting trained weights from bi-lingual sequence model (machine translation task) will be shared in weights/ for further analysis purpose.
- labelled mono-language mail datasets (English language) used in this experiment: Enron mail data set, Lingspam mail data set, Spamassasin mail data set. A compact pre-processed labelled data in python dictionary format will be shared in data/ for reproducible research.
- unlabelled mono-language mail data sets (Dutch language) is sampled from raw mail data suspected as phising emails. The data won't be publicly available, but the author will use another sample set for tutorial purposes (if necessary).
