'''
What Do People Complain About on the Internet?
This is a script to find topics in the r/complaints subreddit.
@author Eliana Grosof
August 13, 2020

help from: https://towardsdatascience.com/topic-modelling-in-python-with-nltk-and-gensim-4ef03213cd21
'''

'''
Functions for cleaning text data, including tokenizing and lemma-ing. 
Uses spaCY and NLTK packages. 
Will probably also combine title and body into one document.
'''

# import and load tools to get tokens of words
from spacy.lang.en import English
parser = English() # parser is same as nlp variable in spaCY documentation

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


'''
LDA with Gensim
'''
def lda(clean_docs, model_name, topics):
    from gensim import corpora
    dictionary = corpora.Dictionary(clean_docs) # turn all data into a dictionary mappping of normalized words and their integer ids
    # convert each document, called text, into bag-of-words representation (list of (token_id, token_count) tuples)
    corpus = []
    for doc in clean_docs:
        corpus.append(dictionary.doc2bow(doc))

    # serialize version: save dictionary and corpus for future use
    from gensim.corpora import MmCorpus
    MmCorpus.serialize('corpus_'+model_name+'.mm', corpus)
    dictionary.save('dictionary_'+model_name+'.gensim')

    '''
    # pickle version: save dictionary and corpus for future use
    import pickle
    pickle.dump(corpus, open('corpus_body.pkl', 'wb'))
    dictionary.save('dictionary_body.gensim')
    '''

    # Train LDA model
    from gensim.models import LdaModel
    num_topics = topics # find this number of topics in the data
    passes = 15

    ldamodel = LdaModel(corpus, num_topics = num_topics, id2word=dictionary, passes=passes)
    ldamodel.save('model3_'+model_name+'.gensim')
    topics = ldamodel.print_topics(num_words=5)

    for topic in topics:
        print(topic)

def main():
    # open csv file containing documents, clean document, add to all_docs list
    all_docs = []
    model_name = 'body_title_081420'
    topics = 3
    with open('data/complaints_body_title.csv') as f:
        for line in f:
            tokens = process_document(line)
            all_docs.append(tokens)

    lda(all_docs, model_name, topics)

main()