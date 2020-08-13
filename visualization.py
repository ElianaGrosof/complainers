'''
A file for visualizing LDA output from topic_modeling.py
'''

import gensim
import pickle
import pyLDAvis.gensim


dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb')) # probably the way I saved this is to blame
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

topics = lda.print_topics(num_words=10)

# for topic in topics:
#     print(topic)

pyLDAvis.enable_notebook()
lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary)
pyLDAvis.display(lda_display)