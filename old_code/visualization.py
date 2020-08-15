'''
A file for visualizing LDA output from topic_modeling.py

Turns out pyLDAvis is deprecated
'''

import gensim
import pickle
import pyLDAvis.gensim


dictionary = gensim.corpora.Dictionary.load('dictionary_title_081420.gensim')
corpus = gensim.corpora.MmCorpus('corpus_title_081420.mm') #pickle.load(open('corpus.pkl', 'rb')) # probably the way I saved this is to blame
lda = gensim.models.ldamodel.LdaModel.load('model10_title_081420.gensim')

topics = lda.print_topics(num_words=10)

pyLDAvis.gensim.prepare(lda, corpus, dictionary)

# for topic in topics:
#     print(topic)

#pyLDAvis.enable_notebook()
#lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
#pyLDAvis.display(lda_display)