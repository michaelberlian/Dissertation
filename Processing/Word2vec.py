import pandas as pd 
import gensim
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec

# load data that had been cleaned and random sampled from Cleaning.py
df = pd.read_csv('../Data/Processed/Data.csv')

# create a list of sentence from list of content
content_list = df['content'].tolist()
arr = [] 
counter = 0 
for content in content_list:
    try:
        sentences = sent_tokenize(content)
        arr += sentences
    except:
        counter += 1

# dropping duplicates and use simple preprocess to the sentences provided by gensim
df = pd.DataFrame([arr],index=['content'],columns=np.arange(len(arr)))
df = df.transpose()
df.drop_duplicates(inplace=True)
contents = df['content'].apply(gensim.utils.simple_preprocess)

# print (contents)

# contents = df['content']

# initiated the Word2vec model from gensim library with parameters that has been tested on ../Testing/Word2vec_model
model = Word2Vec(
    vector_size=150,
    window=8,
    min_count=1,
    workers=4,
    sg=1,
)
# build the vocab and train/fit the model to the processed data
model.build_vocab(contents)
model.train(contents, total_examples=model.corpus_count, epochs=model.epochs)
# save the model
model.save("../Model/Word2vec/word2vec.model")