# Complainers
What do people complain about on Reddit?  
In this project, I use NLP topic modeling techniques
to find the topics that people complain about on the subreddit _r/complaints_.

## Getting Started
This project uses:  
* Python 3.8.1
* gensim 3.8.3
* spaCY 2.3.2
* NLTK 3.5  

To run visualization:  
* jupyter-notebook 6.0.3
* Pandas 1.0.3
* Seaborn 0.10.1

## To Run
1. Get most recent reddit data using scrape.py. *See existing scrape in data/complaints_new_13082020.csv.*
2. Run and save models using topic_modeling.py. (All models are LDA unless specified otherwise.)
3. Visualize using jupyter notebook _visualization.ipynb_

## Results
I found that a 5-topic LDA model worked best.   
Redditors complained about:
1. People, in general
2. Posts people make on reddit
3. Insurance companies
4. Games that are played
5. Work, especially issues pertaining to time

The LSI model returned very similar results, except that it found that redditors complained about 
car problems rather than insurance companies. 

I checked with the subreddit, and confirmed that the results were at least somewhat accurate.

Note that indexing in project starts at 0, so just subtract 1 when looking at visualization.ipynb.