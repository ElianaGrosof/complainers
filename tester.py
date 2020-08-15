import gensim

dictionary = gensim.corpora.Dictionary.load('dictionary_title_081420.gensim')
corpus = gensim.corpora.MmCorpus('corpus_title_081420.mm')
lda = gensim.models.ldamodel.LdaModel.load('model10_title_081420.gensim')

data = []
with open('data/tokenized_body_edited.csv') as f:
    for line in f:
        split_line = line.split(",")

        tokens = []
        for token in split_line:
            if len(token) > 0 and token != "\n":
                tokens.append(token)

        data.append(tokens)

print(data)
print(len(data))