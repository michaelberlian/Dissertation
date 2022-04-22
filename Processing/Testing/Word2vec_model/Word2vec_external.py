import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import pandas as pd
from nltk.tokenize import sent_tokenize
import numpy as np
import gensim

# load existing corpus "text8" using api from gensim 
corpus = api.load('text8')

# build the model using corpus "text8" with tested parameter
model = Word2Vec(sentences=corpus,vector_size=150,
                window=8,
                min_count=1,
                workers=4,
                sg=1,)

# save the model
model.save("../../../Model/Testing/Word2vec/word2vec_text8.model")

##########################################################################################

# using the model with corpus "text8" continue the learning with our dataset from ../Cleaning.py
model = Word2Vec.load("../../../Model/Testing/Word2vec/word2vec_text8.model")

# processing our data from list of paragraph to list of sentences
df = pd.read_csv('../../../Data/Processed/Data.csv')

content_list = df['content'].tolist()
arr = [] 
counter = 0 
for content in content_list:
    try:
        sentences = sent_tokenize(content)
        arr += sentences
    except:
        counter += 1

# simple preprocess
df = pd.DataFrame([arr],index=['content'],columns=np.arange(len(arr)))
df = df.transpose()
df.drop_duplicates(inplace=True)
contents = df['content'].apply(gensim.utils.simple_preprocess)
print (contents)

# update the text8 model with our own data
model.build_vocab(contents, update=True)
model.train(contents, total_examples=1, epochs=1)

# save the model
model.save("../../../Model/Testing/Word2vec/word2vec_text8_added.model")

##########################################################################################

# download pre-trained model of "word2vec-google-news-300"
model = gensim.downloader.load('word2vec-google-news-300')
model.save("../../../Model/Testing/Word2vec/word2vec_word2vec-google-news-300.model")

##########################################################################################

# load existing corpus "20-newsgroup" using api from gensim 
corpus = api.load('20-newsgroups')

# build the model using corpus "newsgroup" with tested parameter
model = Word2Vec(sentences=corpus,vector_size=150,
                window=8,
                min_count=1,
                workers=4,
                sg=1,)

# save the model
model.save("../../../Model/Testing/Word2vec/word2vec_20-newsgroups.model")

#########################################################################################

# build word2vec model using downloaded data of simpsons dataset from kaggle : https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/data
df = pd.read_csv('../../../Data/Raw/simpsons_dataset.csv')


# processing our data from list of paragraph to list of sentences
content_list = df['spoken_words'].tolist()
arr = [] 
counter = 0 
for content in content_list:
    try:
        sentences = sent_tokenize(content)
        arr += sentences
    except:
        counter += 1

df = pd.DataFrame([arr],index=['content'],columns=np.arange(len(arr)))
df = df.transpose()
df.drop_duplicates(inplace=True)
contents = df['content'].apply(gensim.utils.simple_preprocess)
print (contents)

# build and save the model using tested parameters
model = Word2Vec(
    vector_size=150,
    window=8,
    min_count=1,
    workers=4,
    sg=1,
)
model.build_vocab(contents)
model.train(contents, total_examples=model.corpus_count, epochs=model.epochs)
model.save("../../../Model/Testing/Word2vec/word2vec_simpsons.model")