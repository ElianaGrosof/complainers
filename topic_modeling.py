'''
What Do People Complain About on the Internet?
This is a script to find topics in the r/complaints subreddit.
@author Eliana Grosof
August 13, 2020

help from: https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21

how do I do all the documents at once?
'''

'''
Functions for cleaning text data, including tokenizing and lemma-ing. 
Uses spaCY and NLTK packages. 
Will probably also combine title and body into one document.
'''

# import and load tools to get tokens of words
from spacy.lang.en import English
parser = English() # parser is same as nlp variable in spaCY documentation

# import and load tools to get root words
# import nltk
# nltk.download('omw')
# from nltk.stem.wordnet import WordNetLemmatizer

# tokenize text
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens

# get root word (get lemma)
import spacy
sp = spacy.load('en_core_web_sm')
def get_lemma(word):
    spacy_word = sp(word)
    for w in spacy_word:
        if w:
            return w.lemma_

# get stopwords
import nltk
nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

# prepare a single document for topic modeling
def process_document(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 3] # only keep a token if it has more than 3 characters
    tokens = [token for token in tokens if token not in en_stop] # take out the stopwords
    tokens = [get_lemma(token) for token in tokens]
    return tokens

# open csv file containing documents, clean document, add to all_docs list
all_docs = []
with open('complaints_title_only.csv') as f:
    for line in f:
        tokens = process_document(line)
        all_docs.append(tokens)
'''
LDA with Gensim
'''

from gensim import corpora
dictionary = corpora.Dictionary(all_docs) # turn all data into a dictionary mappping of normalized words and their integer ids
# convert each document, called text, into bag-of-words representation (list of (token_id, token_count) tuples)
corpus = []
for doc in all_docs:
    corpus.append(dictionary.doc2bow(doc))

# save dictionary and corpus for future use
import pickle
pickle.dump(corpus, open('corpus_title.pkl', 'wb'))
dictionary.save('dictionary_title.gensim')

# Train LDA model
from gensim.models import LdaModel
num_topics = 10 # find this number of topics in the data
passes = 15

ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=passes)
ldamodel.save('model10_title.gensim')
topics = ldamodel.print_topics(num_words=5)

for topic in topics:
    print(topic)

